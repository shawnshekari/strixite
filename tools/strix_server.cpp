// strix-server: Qwen3.8-Flash-Next behind the OpenAI Chat Completions API. Loads the converted weights (the shipping
// layout U - gdn_in - g128 unless --weights), the n-gram table and the tokenizer, then serves on --host:--port (default
// 0.0.0.0:5300) until SIGINT / SIGTERM.
//
// Settings (serve/server_config.hpp): `--config PATH` reads a key = value file
// (deploy/strix-server.conf is the served one), `--name value` flags override it, `--print-config` prints the
// effective settings and exits, `--help` lists every option with its default and range.
// Default paths are under $STRIX_MODELS_DIR (else ~/models/strix-infer).

#include "common/check.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"
#include "serve/http_server.hpp"
#include "serve/json.hpp"
#include "serve/log.hpp"
#include "serve/memory_plan.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/server_config.hpp"
#include "serve/tokenizer.hpp"

#include <chrono>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>

using namespace strix;

namespace {

std::string models_dir() {
    const char *env = std::getenv("STRIX_MODELS_DIR");
    if (env && *env) return env;
    const char *home = std::getenv("HOME");
    STRIX_CHECK(home && *home, "strix_server: neither STRIX_MODELS_DIR nor HOME is set");
    return std::string(home) + "/models/strix-infer";
}

// generation_config.json's sampling defaults (temperature, top_k, top_p); every one must be there.
SamplingParams sampling_defaults(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "strix_server: can't open generation config '", path, "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    const json::Value g = json::Value::parse(ss.str());
    SamplingParams p;
    const auto need = [&](const char *k) -> const json::Value & {
        const json::Value *v = g.find(k);
        STRIX_CHECK(v != nullptr, "strix_server: '", path, "' has no '", k, "'");
        return *v;
    };
    p.temperature = need("temperature").as_double(path + ".temperature");
    p.top_k = need("top_k").as_int(path + ".top_k");
    p.top_p = need("top_p").as_double(path + ".top_p");
    return p;
}

HttpServer *g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->stop();
}

// Exit status after a GpuFatalError (the GPU context is lost): non-zero, so systemd's Restart=on-failure (or any
// supervisor) starts a fresh process - the only recovery (strixite issue #6).
constexpr int kExitGpuFatal = 3;
// Exit status when the startup memory check refuses (low-memory = fail): deploy/strix-server.service names it in
// RestartPreventExitStatus - a restart would load the weights again only to refuse again.
constexpr int kExitLowMemory = 4;

int run(int argc, char **argv) {
    ServerSettings settings(strix_server_options(models_dir()));
    settings.load(std::vector<std::string>(argv + 1, argv + argc));
    if (settings.help()) {
        std::fputs(settings.usage("strix_server").c_str(), stdout);
        return 0;
    }
    if (settings.print_config()) {
        std::fputs(settings.render().c_str(), stdout);
        return 0;
    }
    slog(LogLevel::Info, "startup: config file %s",
         settings.config_path().empty() ? "none (defaults and flags only)" : settings.config_path().c_str());
    for (const ConfigOption &o : settings.options()) {
        const ConfigSetting &s = settings.setting(o.name);
        slog(LogLevel::Info, "startup: config %s = %s (%s)", o.name.c_str(), s.value.c_str(), s.source.c_str());
    }
    const std::string weights = settings.text("weights"), ngram = settings.text("ngram");
    const std::string tokenizer = settings.text("tokenizer"), gen_config = settings.text("generation-config");
    ServerConfig cfg;
    cfg.host = settings.text("host");
    cfg.port = (int)settings.integer("port");
    cfg.model_id = settings.text("model-id");
    const int64_t capacity = settings.integer("capacity"), chunk = settings.integer("chunk");
    const float yarn_factor = (float)settings.number("rope-yarn-factor");
    const std::string cache_dir = settings.text("prompt-cache-dir");
    const int64_t cache_gib = settings.integer("prompt-cache-gib");
    PromptCache::Options cache_opts;
    cache_opts.ram_margin = (uint64_t)settings.integer("prompt-cache-ram-margin-gib") << 30;
    cache_opts.idle_seconds = settings.number("prompt-cache-idle-s");
    cache_opts.write_gib_per_hour = settings.number("prompt-cache-write-gib-per-hour");
    cache_opts.checkpoints_per_chain = (int)settings.integer("prompt-cache-checkpoints");
    cache_opts.delta_max_tokens = settings.integer("prompt-cache-delta-max-tokens");
    cache_opts.disk_free_margin = (uint64_t)settings.integer("prompt-cache-disk-free-gib") << 30;
    const bool use_mtp = settings.on("mtp");
    const int64_t mtp_draft = settings.integer("mtp-draft"), mtp_vocab = settings.integer("mtp-vocab");
    const float mtp_margin = (float)settings.number("mtp-margin");
    const int64_t ngram_cache_rows = settings.integer("ngram-cache-rows");
    const std::string capture_dir = settings.text("capture-dir");  // empty: no capture
    const bool think_nudge = settings.on("think-nudge");
    const bool think_end_guard = settings.on("think-end-guard");
    const bool literal_tags = settings.on("literal-tags");
    ThinkWatch::Policy nudge_policy;
    nudge_policy.rate = settings.number("think-nudge-rate");
    nudge_policy.min_tokens = settings.integer("think-nudge-min-tokens");
    const std::string nudge_wording = settings.text("think-nudge-wording");
    think_nudge1_text(nudge_wording);  // an unknown wording fails startup, naming the choices
    cfg.thinking_default = settings.on("thinking");
    cfg.defaults = sampling_defaults(gen_config);
    const Tokenizer tok(tokenizer);
    slog(LogLevel::Info, "startup: tokenizer %s (%d tokens)", tokenizer.c_str(), tok.size());
    Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, /*allow_truncated=*/false, ngram_cache_rows, yarn_factor);
    if (yarn_factor > 1.0f)
        slog(LogLevel::Info, "startup: YaRN factor %g: %lld positions (%lld trained), cos/sin factor %.4f",
             (double)yarn_factor, (long long)model.dims().max_positions(), (long long)model.dims().trained_positions,
             (double)model.rope_scale());
    slog(LogLevel::Info, "startup: %.1f GiB of weights loaded in %.1f s",
                 (double)model.weights().data_bytes() / (1ull << 30), model.weights().load_seconds());
    // MTP drafts score a Q4 copy of the draft vocabulary's LM head rows (Qwen4ExpModel::make_draft_head_q4; strixite
    // PR #2): half the bytes per draft call, the verify still decides every token.
    if (use_mtp && model.has_mtp()) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t rows = make_served_mtp_draft_head(model, mtp_vocab);
        slog(LogLevel::Info, "startup: MTP draft head: Q4 G64 copy of the first %lld LM head rows, %.1f MB, made in %.2f s",
             (long long)rows, (double)model.draft_head_q4_bytes() / 1e6,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    Qwen4ExpBackend backend(model, capacity, chunk, use_mtp, mtp_vocab);
    {
        // Where the GPU memory went (all of it is system RAM here - the prompt cache's RAM tier gets what's left).
        const auto tally = GpuMemoryTally::snapshot();
        int64_t total = 0;
        for (const auto &c : tally) total += c.second;
        std::string top;
        for (size_t i = 0; i < tally.size(); ++i) {
            if (tally[i].second < (64ll << 20)) break;  // largest first: the rest is small
            char item[160];
            std::snprintf(item, sizeof item, "%s%s %.2f", i ? ", " : "", tally[i].first.c_str(),
                          (double)tally[i].second / (1ull << 30));
            top += item;
        }
        slog(LogLevel::Info, "startup: GPU memory %.1f GiB in %zu buffer kinds; GiB per kind (>= 64 MiB): %s",
             (double)total / (1ull << 30), tally.size(), top.c_str());
    }
    std::string degraded;  // what the memory plan turned down (Engine::Options::degraded)
    {
        // The startup memory check (serve/memory_plan.hpp, strixite issue #6): every GPU buffer is allocated now.
        int64_t gpu_total = 0;
        for (const auto &c : GpuMemoryTally::snapshot()) gpu_total += c.second;
        const NgramTableRows *table =
            model.has_ngram_rows() ? dynamic_cast<const NgramTableRows *>(&model.ngram_rows()) : nullptr;
        MemoryPlanInput mi;
        mi.adapt = parse_low_memory(settings.text("low-memory"));
        mi.mem_available = proc_mem_available();
        mi.gpu_bytes = (uint64_t)gpu_total;
        mi.prompt_cache = cache_gib > 0;
        mi.ram_margin = cache_opts.ram_margin;
        mi.capacity = capacity;
        mi.state_bytes = [&backend](int64_t n) { return backend.state_bytes_at(n); };
        mi.row_cache_rows = table ? table->cache_rows() : 0;
        mi.row_cache_row_bytes = table ? table->cache_bytes_per_row() : 0;
        mi.row_cache_min_rows = std::min<int64_t>(mi.row_cache_rows, NgramTableRows::kDefaultCacheRows);
        const MemoryPlan plan = plan_memory(mi);
        for (const std::string &l : plan.info) slog(LogLevel::Info, "%s", l.c_str());
        if (plan.refuse) {
            slog(LogLevel::Error, "fatal: %s", plan.refusal.c_str());
            std::fflush(nullptr);
            return kExitLowMemory;
        }
        for (const std::string &l : plan.warnings) slog(LogLevel::Warning, "startup: %s", l.c_str());
        if (plan.spares_off) cache_opts.max_spares = 0, cache_opts.prefault = false;
        if (plan.ram_tier_off) cache_opts.ram_tier = false, cache_opts.ram_margin = kLowMemoryFloor;
        if (table && plan.row_cache_rows != table->cache_rows()) {
            table->resize_cache(plan.row_cache_rows);
            slog(LogLevel::Info, "startup: memory: MemAvailable %.1f GiB after the row cache shrink",
                 (double)proc_mem_available() / (1ull << 30));
        }
        cache_opts.status_note = plan.status;
        degraded = plan.status;
    }
    std::unique_ptr<PromptCache> cache;
    if (cache_gib > 0) {
        cache = std::make_unique<PromptCache>(cache_dir, (uint64_t)cache_gib << 30, backend.state_fingerprint(), cache_opts);
        const PromptCacheStats s = cache->stats();
        slog(LogLevel::Info, "startup: prompt cache %s: %lld entries, %.1f of %lld GiB (%lld invalid dropped)",
             cache_dir.c_str(), (long long)s.entries, (double)s.bytes / (1ull << 30), (long long)cache_gib, (long long)s.invalid);
        slog(LogLevel::Info, "startup: prompt cache RAM tier: margin %.0f GiB, idle write after %.0f s, budget %.0f GiB/h, "
             "disk free margin %.0f GiB", (double)cache_opts.ram_margin / (1ull << 30), cache_opts.idle_seconds,
             cache_opts.write_gib_per_hour, (double)cache_opts.disk_free_margin / (1ull << 30));
        // Disk space (logged only; whether low space should adapt too is open): the cap is reachable when the
        // filesystem's free space minus the free margin covers what the cache doesn't hold yet. Short of it, the cache
        // stays smaller (disk_room_locked's rule) - and with no room at all, nothing is written.
        const double gib = (double)(1ull << 30);
        const uint64_t free = cache->disk_free_now(), margin = cache_opts.disk_free_margin;
        const uint64_t cap = (uint64_t)cache_gib << 30, used = (uint64_t)s.bytes;
        const uint64_t usable = used + (free > margin ? free - margin : 0);  // what the cache can grow to
        if (usable >= cap)
            slog(LogLevel::Info, "startup: disk: %.0f GiB free on the prompt cache's filesystem; the %lld GiB cap fits "
                 "(%.1f GiB used, %.0f GiB kept free)", free / gib, (long long)cache_gib, used / gib, margin / gib);
        else
            slog(LogLevel::Warning, "startup: WARNING: disk: %.0f GiB free on the prompt cache's filesystem (%s): the "
                 "cache can grow to %.1f GiB, not its %lld GiB cap (%.1f GiB used, %.0f GiB kept free by "
                 "prompt-cache-disk-free-gib)%s - fewer conversations survive a switch or a restart. Free disk space, or "
                 "lower prompt-cache-gib / prompt-cache-disk-free-gib", free / gib, cache_dir.c_str(), usable / gib,
                 (long long)cache_gib, used / gib, margin / gib,
                 free <= margin ? "; below the free margin: no entry is written" : "");
    }
    Engine::Options eng_opts;
    eng_opts.cache = cache.get();
    eng_opts.mtp_margin = mtp_margin;
    eng_opts.mtp_draft = mtp_draft;
    eng_opts.think_nudge = think_nudge;
    eng_opts.think_nudge_policy = nudge_policy;
    eng_opts.think_nudge_wording = nudge_wording;
    eng_opts.think_end_guard = think_end_guard;
    eng_opts.literal_tags = literal_tags;
    eng_opts.degraded = degraded;
    slog(LogLevel::Info, "startup: think-block / tool-call guard %s, literal tags %s", think_end_guard ? "on" : "off",
         literal_tags ? "on" : "off");
    if (think_nudge)
        slog(LogLevel::Info, "startup: thinking nudge on: %.0f%% self-copy over %lld tokens past %lld, wording %s",
             100.0 * nudge_policy.rate, (long long)nudge_policy.window, (long long)nudge_policy.min_tokens,
             nudge_wording.c_str());
    else
        slog(LogLevel::Info, "startup: thinking nudge off");
    slog(LogLevel::Info, "startup: thinking %s for requests that don't say", cfg.thinking_default ? "on" : "off");
    if (!capture_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(capture_dir, ec);
        STRIX_CHECK(!ec, "--capture-dir ", capture_dir, ": ", ec.message());
        eng_opts.capture_dir = capture_dir;
        slog(LogLevel::Info, "startup: capturing every request's tokens and MTP steps to %s", capture_dir.c_str());
    }
    std::atomic<bool> gpu_fatal{false};
    HttpServer *server_ptr = nullptr;  // set before serving; on_fatal only runs once a request did
    eng_opts.on_fatal = [&](const std::string &) {
        gpu_fatal = true;
        if (server_ptr) server_ptr->stop();
    };
    Engine engine(backend, tok, eng_opts);
    HttpServer server(engine, tok, cfg);
    const int port = server.bind();
    server_ptr = &server;
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    slog(LogLevel::Info, "startup: %s", backend.describe().c_str());
    slog(LogLevel::Info, "startup: serving %s on %s:%d", cfg.model_id.c_str(), cfg.host.c_str(), port);
    server.serve();
    g_server = nullptr;
    if (gpu_fatal) {
        // No destructors past this point: the backend's would call HIP on the lost context (and may hang there). The
        // prompt cache is host memory only: its shutdown rule still writes the RAM entries before the exit.
        slog(LogLevel::Error, "fatal: the GPU context is lost (%s); exiting with status %d for a restart",
             engine.fatal_error().c_str(), kExitGpuFatal);
        cache.reset();
        std::fflush(nullptr);
        std::_Exit(kExitGpuFatal);
    }
    slog(LogLevel::Info, "stopped");
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        slog(LogLevel::Error, "fatal: %s", e.what());
        return 1;
    }
}
