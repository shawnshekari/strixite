#pragma once

// strix_server's settings: one table of options drives the command-line
// flags, the config file and the usage text, so the three can't drift apart.
//
// Config file: one `key = value` a line; `#` starts a comment anywhere on a line; blank lines are ignored. Keys are
// the flag names without `--`, values are in the flags' own syntax. Paths in the file must be absolute (a relative
// one would depend on the working directory of whoever starts the server).
//
// Precedence: built-in default < file (--config PATH; no file is read unless it's named) < command-line flag.
// Strict: an unknown key, a duplicate (in the file or among the flags), a line without `=`, an empty value, a value
// that doesn't parse or is out of range, an unreadable file - each throws, naming the file:line (or flag), the key,
// what was expected and what was found. Never a silently ignored setting.

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

struct ConfigOption {
    enum class Kind {
        Int,        // integer in [lo, hi]
        Number,     // floating point in [lo, hi]
        OnOff,      // "on" or "off"
        Path,       // a non-empty path (absolute in the file)
        PathOrOff,  // a path, or "off" for none
        Text,       // any non-empty text
    };
    std::string name;  // the flag without "--" and the file's key
    Kind kind = Kind::Text;
    double lo = 0, hi = 0;  // Int / Number only
    std::string default_value;  // in the file / flag syntax; validated like any other value
    std::string help;           // one line for the usage text
};

// One option's effective value and where it came from: "default", "file:LINE" or "flag".
struct ConfigSetting {
    std::string value;
    std::string source;
};

class ServerSettings {
public:
    explicit ServerSettings(std::vector<ConfigOption> options);

    // args: argv[1..]. `--config PATH` names the file, `--print-config` and `--help` are switches; every other
    // argument is a `--name value` pair from the table. Reads the file (if named), then applies the precedence.
    void load(const std::vector<std::string> &args);
    // The file front end on its own (load() calls it): the file's settings over the defaults. Exposed for tests.
    void load_file(const std::string &path);

    int64_t integer(const std::string &name) const;
    double number(const std::string &name) const;
    bool on(const std::string &name) const;
    const std::string &text(const std::string &name) const;  // Path / Text; PathOrOff: "" for off
    const ConfigSetting &setting(const std::string &name) const;

    const std::vector<ConfigOption> &options() const { return options_; }
    const std::string &config_path() const { return config_path_; }  // "" if no --config
    bool print_config() const { return print_config_; }
    bool help() const { return help_; }

    std::string usage(const std::string &program) const;
    // The effective settings in the file's syntax, each with its source as a trailing comment (loadable as a file).
    std::string render() const;

private:
    size_t index(const std::string &name, const char *caller) const;
    // Checks value against option i's kind and range; `where` names the origin for the error ("file F:L", "--x").
    void validate(size_t i, const std::string &value, const std::string &where) const;

    std::vector<ConfigOption> options_;
    std::vector<ConfigSetting> settings_;
    std::string config_path_;
    bool print_config_ = false;
    bool help_ = false;
};

// The served n-gram row cache (8 M rows, ~3 GiB; deploy/strix-server.conf has the reason) - the server's default. The
// library's own default (NgramTableRows::kDefaultCacheRows, 2^18) stays for tools and benches.
constexpr int64_t kServedNgramCacheRows = 8388608;

// strix_server's option table. models_dir: $STRIX_MODELS_DIR (else ~/models/strix-infer), under which the default
// paths live. Every default is the served value (2026-10-10: the defaults are what a dedicated headless
// Strix Halo serving a coding agent uses); tests/test_server_config.cpp checks deploy/strix-server.conf against them.
std::vector<ConfigOption> strix_server_options(const std::string &models_dir);

}  // namespace strix
