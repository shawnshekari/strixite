#include "serve/server_config.hpp"

#include "common/args.hpp"
#include "common/check.hpp"
#include "serve/http_server.hpp"
#include "serve/prompt_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace strix {

namespace {

std::string trim(const std::string &s) {
    const char *ws = " \t\r";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string fmt_number(double v) {
    // Whole numbers as integers (600, not 6e+02); otherwise the shortest form that reads back the same.
    char s[64];
    if (v == (double)(long long)v && v > -1e15 && v < 1e15) {
        std::snprintf(s, sizeof s, "%lld", (long long)v);
        return s;
    }
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(s, sizeof s, "%.*g", p, v);
        if (std::strtod(s, nullptr) == v) break;
    }
    return s;
}

const char *placeholder(ConfigOption::Kind k) {
    switch (k) {
    case ConfigOption::Kind::Int: return "N";
    case ConfigOption::Kind::Number: return "X";
    case ConfigOption::Kind::OnOff: return "on|off";
    case ConfigOption::Kind::Path: return "PATH";
    case ConfigOption::Kind::PathOrOff: return "PATH|off";
    case ConfigOption::Kind::Text: return "TEXT";
    }
    return "?";
}

}  // namespace

ServerSettings::ServerSettings(std::vector<ConfigOption> options) : options_(std::move(options)) {
    STRIX_CHECK(!options_.empty(), "ServerSettings: empty option table");
    settings_.reserve(options_.size());
    for (size_t i = 0; i < options_.size(); ++i) {
        const ConfigOption &o = options_[i];
        STRIX_CHECK(!o.name.empty() && o.name.find_first_of(" \t=#") == std::string::npos && o.name.rfind("--", 0) != 0,
                    "ServerSettings: option ", i, " has a bad name '", o.name, "' (expected a flag name without --)");
        STRIX_CHECK(o.name != "config" && o.name != "print-config" && o.name != "help",
                    "ServerSettings: option name '", o.name, "' is reserved for the command line");
        for (size_t j = 0; j < i; ++j)
            STRIX_CHECK(options_[j].name != o.name, "ServerSettings: option '", o.name, "' is in the table twice");
        if (o.kind == ConfigOption::Kind::Int || o.kind == ConfigOption::Kind::Number)
            STRIX_CHECK(o.lo <= o.hi, "ServerSettings: option '", o.name, "' has range [", o.lo, ", ", o.hi, "]");
        STRIX_CHECK(!o.help.empty(), "ServerSettings: option '", o.name, "' has no help text");
        validate(i, o.default_value, "the default of '" + o.name + "'");
        settings_.push_back({o.default_value, "default"});
    }
}

size_t ServerSettings::index(const std::string &name, const char *caller) const {
    for (size_t i = 0; i < options_.size(); ++i)
        if (options_[i].name == name) return i;
    STRIX_FAIL("ServerSettings::", caller, ": no option '", name, "' in the table");
}

void ServerSettings::validate(size_t i, const std::string &value, const std::string &where) const {
    const ConfigOption &o = options_[i];
    const std::string what = where + ": " + o.name;
    switch (o.kind) {
    case ConfigOption::Kind::Int: parse_int(value, what, (long long)o.lo, (long long)o.hi); break;
    case ConfigOption::Kind::Number: parse_double(value, what, o.lo, o.hi); break;
    case ConfigOption::Kind::OnOff:
        STRIX_CHECK(value == "on" || value == "off", what, " = '", value, "'; expected on or off");
        break;
    case ConfigOption::Kind::Path:
    case ConfigOption::Kind::PathOrOff:
    case ConfigOption::Kind::Text:
        STRIX_CHECK(!value.empty(), what, " is empty; expected ",
                    o.kind == ConfigOption::Kind::Text ? "text" : o.kind == ConfigOption::Kind::Path ? "a path" : "a path or off");
        break;
    }
}

void ServerSettings::load_file(const std::string &path) {
    STRIX_CHECK(!path.empty(), "ServerSettings::load_file: empty path");
    std::ifstream f(path);
    STRIX_CHECK(f.good(), "config file '", path, "': can't open it for reading");
    std::vector<std::string> seen_at(options_.size());  // "LINE" where each key was set
    std::string raw;
    for (int line = 1; std::getline(f, raw); ++line) {
        const std::string where = "config file " + path + ":" + std::to_string(line);
        const size_t hash = raw.find('#');
        const std::string s = trim(hash == std::string::npos ? raw : raw.substr(0, hash));
        if (s.empty()) continue;
        const size_t eq = s.find('=');
        STRIX_CHECK(eq != std::string::npos, where, ": '", s, "' has no '='; expected key = value");
        const std::string key = trim(s.substr(0, eq)), value = trim(s.substr(eq + 1));
        STRIX_CHECK(!key.empty(), where, ": no key before '='; expected key = value");
        size_t i = 0;
        while (i < options_.size() && options_[i].name != key) ++i;
        STRIX_CHECK(i < options_.size(), where, ": unknown key '", key, "'",
                    key.rfind("--", 0) == 0 ? " (keys are flag names without --)" : "", "; see strix_server --help");
        STRIX_CHECK(seen_at[i].empty(), where, ": duplicate key '", key, "', already set on line ", seen_at[i]);
        STRIX_CHECK(!value.empty(), where, ": ", key, " has an empty value; expected ", placeholder(options_[i].kind));
        validate(i, value, where);
        const ConfigOption::Kind k = options_[i].kind;
        if (k == ConfigOption::Kind::Path || (k == ConfigOption::Kind::PathOrOff && value != "off"))
            STRIX_CHECK(value[0] == '/', where, ": ", key, " = '", value,
                        "' is relative; expected an absolute path (a file's paths can't depend on the working directory)");
        seen_at[i] = std::to_string(line);
        settings_[i] = {value, "file:" + std::to_string(line)};
    }
    STRIX_CHECK(f.eof(), "config file '", path, "': read error");
}

void ServerSettings::load(const std::vector<std::string> &args) {
    std::vector<std::pair<size_t, std::string>> flags;
    std::vector<bool> flagged(options_.size(), false);
    bool have_config = false;
    for (size_t a = 0; a < args.size(); ++a) {
        const std::string &k = args[a];
        if (k == "--print-config") {
            STRIX_CHECK(!print_config_, "command line: --print-config given twice");
            print_config_ = true;
            continue;
        }
        if (k == "--help") {
            help_ = true;
            continue;
        }
        STRIX_CHECK(k.rfind("--", 0) == 0 && k.size() > 2, "command line: '", k,
                    "' where an option was expected (--name value); see strix_server --help");
        STRIX_CHECK(a + 1 < args.size(), "command line: ", k, " has no value");
        const std::string &v = args[++a];
        const std::string name = k.substr(2);
        if (name == "config") {
            STRIX_CHECK(!have_config, "command line: --config given twice ('", config_path_, "' and '", v, "')");
            STRIX_CHECK(!v.empty(), "command line: --config is empty; expected a path");
            have_config = true;
            config_path_ = v;
            continue;
        }
        size_t i = 0;
        while (i < options_.size() && options_[i].name != name) ++i;
        STRIX_CHECK(i < options_.size(), "command line: unknown option '", k, "'; see strix_server --help");
        STRIX_CHECK(!flagged[i], "command line: ", k, " given twice");
        validate(i, v, "command line");
        flagged[i] = true;
        flags.emplace_back(i, v);
    }
    if (have_config) load_file(config_path_);
    for (const auto &[i, v] : flags) settings_[i] = {v, "flag"};
}

const ConfigSetting &ServerSettings::setting(const std::string &name) const { return settings_[index(name, "setting")]; }

int64_t ServerSettings::integer(const std::string &name) const {
    const size_t i = index(name, "integer");
    STRIX_CHECK(options_[i].kind == ConfigOption::Kind::Int, "ServerSettings::integer: '", name, "' isn't an integer option");
    return parse_int(settings_[i].value, name, (long long)options_[i].lo, (long long)options_[i].hi);
}

double ServerSettings::number(const std::string &name) const {
    const size_t i = index(name, "number");
    STRIX_CHECK(options_[i].kind == ConfigOption::Kind::Number, "ServerSettings::number: '", name, "' isn't a number option");
    return parse_double(settings_[i].value, name, options_[i].lo, options_[i].hi);
}

bool ServerSettings::on(const std::string &name) const {
    const size_t i = index(name, "on");
    STRIX_CHECK(options_[i].kind == ConfigOption::Kind::OnOff, "ServerSettings::on: '", name, "' isn't an on|off option");
    return settings_[i].value == "on";
}

const std::string &ServerSettings::text(const std::string &name) const {
    static const std::string none;
    const size_t i = index(name, "text");
    const ConfigOption::Kind k = options_[i].kind;
    STRIX_CHECK(k == ConfigOption::Kind::Path || k == ConfigOption::Kind::PathOrOff || k == ConfigOption::Kind::Text,
                "ServerSettings::text: '", name, "' isn't a path or text option");
    if (k == ConfigOption::Kind::PathOrOff && settings_[i].value == "off") return none;
    return settings_[i].value;
}

std::string ServerSettings::usage(const std::string &program) const {
    std::ostringstream u;
    u << "usage: " << program << " [--config PATH] [--print-config] [--help] [--name VALUE]...\n"
      << "  --config PATH     settings file: key = value lines (keys: the names below), # comments\n"
      << "  --print-config    print the effective settings in the file's syntax and exit\n"
      << "  precedence: default < config file < flag\n";
    for (const ConfigOption &o : options_) {
        u << "  --" << o.name << ' ' << placeholder(o.kind);
        if (o.kind == ConfigOption::Kind::Int || o.kind == ConfigOption::Kind::Number)
            u << " [" << fmt_number(o.lo) << ", " << fmt_number(o.hi) << "]";
        u << "  (default " << o.default_value << ")\n      " << o.help << '\n';
    }
    return u.str();
}

std::string ServerSettings::render() const {
    std::ostringstream r;
    r << "# strix_server effective settings; each line's source after the value\n";
    for (size_t i = 0; i < options_.size(); ++i)
        r << options_[i].name << " = " << settings_[i].value << "  # " << settings_[i].source << '\n';
    return r.str();
}

std::vector<ConfigOption> strix_server_options(const std::string &models_dir, int64_t ngram_cache_rows_default) {
    STRIX_CHECK(!models_dir.empty(), "strix_server_options: empty models_dir");
    STRIX_CHECK(ngram_cache_rows_default >= 0, "strix_server_options: ngram_cache_rows_default ", ngram_cache_rows_default);
    using K = ConfigOption::Kind;
    const ServerConfig server;
    const PromptCache::Options cache;
    const std::string ckpt = models_dir + "/Qwen3.8-Flash-Next";
    return {
        {"weights", K::Path, 0, 0, models_dir + "/converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw",
         "converted weights (strixw); the shipping layout U - gdn_in - g128 by default"},
        {"ngram", K::Path, 0, 0, models_dir + "/converted/Qwen3.8-Flash-Next.ngram-q8/ngram.table",
         "the n-gram table (int8 rows)"},
        {"tokenizer", K::Path, 0, 0, ckpt + "/tokenizer.json", "tokenizer.json"},
        {"generation-config", K::Path, 0, 0, ckpt + "/generation_config.json",
         "generation_config.json: the sampling defaults (temperature, top_k, top_p)"},
        {"host", K::Text, 0, 0, server.host, "address to listen on"},
        {"port", K::Int, 1, 65535, std::to_string(server.port), "port to listen on"},
        {"capacity", K::Int, 4096, 1 << 20, "262144", "positions per session (at most 262144 x rope-yarn-factor)"},
        {"rope-yarn-factor", K::Number, 1, 8, "1",
         "YaRN context extension: positions as trained (262144) x this; 1 = off (RoPE as trained). A change makes "
         "the prompt cache's entries from the other setting unusable (dropped at startup)"},
        {"chunk", K::Int, 1, 16384, "512", "prefill chunk in tokens; progress and cancellation are checked per chunk"},
        {"model-id", K::Text, 0, 0, server.model_id, "the model name the API reports"},
        {"prompt-cache-dir", K::Path, 0, 0, models_dir + "/prompt-cache", "the disk prompt cache's directory"},
        {"prompt-cache-gib", K::Int, 0, 1 << 16, "128", "the disk prompt cache's cap in GiB; 0 = no prompt cache"},
        {"prompt-cache-ram-margin-gib", K::Int, 0, 1 << 12, std::to_string(cache.ram_margin >> 30),
         "RAM tier: keep MemAvailable above this many GiB"},
        {"low-memory", K::Text, 0, 0, "fail",
         "startup memory check (serve/memory_plan.hpp): fail = refuse to start when MemAvailable after loading is short "
         "of the prompt cache's margin + one full-capacity state, naming the settings that fit; adapt = start with the "
         "RAM caches turned down, each step logged"},
        {"prompt-cache-idle-s", K::Number, 1, 1e7, fmt_number(cache.idle_seconds),
         "RAM tier: an entry idle this many seconds is written to disk"},
        {"prompt-cache-write-gib-per-hour", K::Number, 0, 1e6, fmt_number(cache.write_gib_per_hour),
         "RAM tier: disk write budget in GiB per hour; 0 = no limit"},
        {"prompt-cache-disk-free-gib", K::Int, 0, 1 << 16, std::to_string(cache.disk_free_margin >> 30),
         "a cache write never leaves less than this many GiB free on the cache's filesystem"},
        {"prompt-cache-checkpoints", K::Int, 1, 1000, std::to_string(cache.checkpoints_per_chain),
         "prompt cache checkpoints kept per conversation"},
        {"prompt-cache-delta-max-tokens", K::Int, 0, 1 << 24, std::to_string(cache.delta_max_tokens),
         "a turn is saved as a delta on the whole entry it extends while it adds at most this many tokens (and a "
         "quarter of the base); 0 = every entry whole"},
        {"mtp", K::OnOff, 0, 0, "on", "MTP drafting"},
        // MTP defaults from the sweeps (bench/results.jsonl 740bdd7 + c816938-k*, real text, 10 prompts): chained
        // drafts at margin 2.0 with a 65536-id draft vocabulary. 5 drafts since 2026-10-01: teacher-forced replays of
        // real traffic, 3 repeats (7692313-draftconf-*): -0.60% ms/token on terminal-bench / mixed captures, -0.74% on
        // opencode turns vs 4.
        {"mtp-draft", K::Int, 1, 15, "5", "MTP: most drafts per verify (the head chained)"},
        {"mtp-margin", K::Number, 0, 100, "2", "MTP: draft only while the top-1 logit margin is at least this"},
        {"mtp-vocab", K::Int, 0, 1 << 30, "65536", "MTP: drafts are scored over token ids [0, N); 0 = the whole vocabulary"},
        {"ngram-cache-rows", K::Int, 0, 1 << 26, std::to_string(ngram_cache_rows_default),
         "the n-gram table's LRU row cache in rows (320 B each, plus index); 0 = none"},
        {"capture-dir", K::PathOrOff, 0, 0, "off",
         "write each request's token ids and MTP steps here (serve/capture.hpp); off = no capture"},
        {"think-nudge", K::OnOff, 0, 0, "on",
         "the thinking nudge into a think block going in circles (serve/think_nudge.hpp); off for like-for-like comparisons"},
        {"think-end-guard", K::OnOff, 0, 0, "on",
         "no end of turn while the think block or a tool call is open (Engine::Options::think_end_guard); off for like-for-like comparisons"},
        {"literal-tags", K::OnOff, 0, 0, "on",
         "a </think> / <tool_call> / </tool_call> token not right after a newline is text (OutputParser::Config::literal_tags); off for like-for-like comparisons"},
        {"think-nudge-rate", K::Number, 0.01, 1, "0.25",
         "thinking nudge: due when this share of the last 1024 thinking tokens repeat earlier 8-grams"},
        {"think-nudge-min-tokens", K::Int, 1024, 1 << 24, "3072", "thinking nudge: no rate trigger before this many thinking tokens"},
        {"think-nudge-wording", K::Text, 0, 0, "commit",
         "thinking nudge 1's text: commit (\"... commit to the most promising approach now\") or decide (\"... state my "
         "decision and the exact next command\")"},
        {"thinking", K::OnOff, 0, 0, "on",
         "thinking for requests that don't say (enable_thinking / thinking.type / reasoning_effort / a budget); off = "
         "rendered without the think block unless the request asks"},
    };
}

}  // namespace strix
