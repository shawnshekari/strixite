#include "serve/prompt_cache.hpp"
#include "serve/log.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"  // strix_hash64

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace strix {

namespace fs = std::filesystem;

namespace {

constexpr char kMagic[8] = {'S', 'T', 'R', 'X', 'P', 'C', 'C', 'H'};
constexpr uint32_t kVersion = 2;  // 2: delta entries (base_n); version 1 files (no base) are still read
constexpr uint64_t kHeaderBytes = 4096;
// Header (little endian, x86-64 memcpy): 0 magic[8]  8 u32 version  12 u32 kind  16 u64 tokens  24 u64 state bytes
//   32 u64 state offset  40 u64 token hash  48 u64 state hash  56 char fingerprint[192]
//   version 1: 248 u64 header hash (of 0..248)
//   version 2: 248 u64 base_n (0: a whole state; else a delta on the entry of the first base_n tokens)
//              256 u64 header hash (of 0..256)
constexpr size_t kFpAt = 56, kFpBytes = 192, kHeaderHashAtV1 = 248, kBaseAt = 248, kHeaderHashAt = 256;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// For log lines: what an entry is.
const char *kind_name(PromptCache::Kind k) {
    return k == PromptCache::Kind::System ? "system-prefix" : k == PromptCache::Kind::Checkpoint ? "checkpoint" : "turn";
}

// For log lines: a duration as "40 s" / "12 min" / "3.5 h".
std::string ago(double seconds) {
    char b[32];
    if (seconds < 120) std::snprintf(b, sizeof b, "%.0f s", seconds);
    else if (seconds < 7200) std::snprintf(b, sizeof b, "%.0f min", seconds / 60);
    else std::snprintf(b, sizeof b, "%.1f h", seconds / 3600);
    return b;
}

// Continuation lines of a "prompt cache: " line start under its text.
constexpr const char *kCont = "              ";
uint64_t align4k(uint64_t x) { return (x + 4095) & ~uint64_t{4095}; }
uint64_t file_total(uint64_t n_tokens, uint64_t state_bytes) { return align4k(kHeaderBytes + n_tokens * 4) + state_bytes; }

void write_all(int fd, const void *p, size_t n, uint64_t at, const std::string &where) {
    const auto *b = static_cast<const uint8_t *>(p);
    while (n > 0) {
        const ssize_t w = ::pwrite(fd, b, n, (off_t)at);
        if (w < 0 && errno == EINTR) continue;
        STRIX_CHECK(w > 0, where, ": write of ", n, " bytes at ", at, " failed: ", w < 0 ? std::strerror(errno) : "0 bytes");
        b += w, n -= (size_t)w, at += (uint64_t)w;
    }
}

void read_all(int fd, void *p, size_t n, uint64_t at, const std::string &where) {
    auto *b = static_cast<uint8_t *>(p);
    while (n > 0) {
        const ssize_t r = ::pread(fd, b, n, (off_t)at);
        if (r < 0 && errno == EINTR) continue;
        STRIX_CHECK(r > 0, where, ": read of ", n, " bytes at ", at, " failed: ", r < 0 ? std::strerror(errno) : "end of file");
        b += r, n -= (size_t)r, at += (uint64_t)r;
    }
}

struct Header {
    uint32_t kind = 0;
    uint64_t tokens = 0, state_bytes = 0, state_at = 0, token_hash = 0, state_hash = 0;
    uint64_t base_n = 0;  // > 0: a delta
    std::string fingerprint;
};

// Reads and checks the header of an open entry against the expected fingerprint and the file size; throws.
Header read_header(int fd, const std::string &path, const std::string &fingerprint, uint64_t file_bytes) {
    std::vector<uint8_t> h(kHeaderBytes);
    read_all(fd, h.data(), h.size(), 0, path);
    auto rd = [&](size_t off, auto v) {
        std::memcpy(&v, h.data() + off, sizeof(v));
        return v;
    };
    STRIX_CHECK(std::memcmp(h.data(), kMagic, 8) == 0, path, ": not a prompt cache entry (bad magic)");
    const uint32_t version = rd(8, uint32_t{});
    STRIX_CHECK(version == 1 || version == kVersion, path, ": version ", version, ", this build reads 1..", kVersion);
    const size_t hash_at = version == 1 ? kHeaderHashAtV1 : kHeaderHashAt;
    STRIX_CHECK(rd(hash_at, uint64_t{}) == strix_hash64(h.data(), hash_at), path, ": header hash mismatch");
    Header r;
    r.base_n = version == 1 ? 0 : rd(kBaseAt, uint64_t{});
    r.kind = rd(12, uint32_t{}), r.tokens = rd(16, uint64_t{}), r.state_bytes = rd(24, uint64_t{});
    r.state_at = rd(32, uint64_t{}), r.token_hash = rd(40, uint64_t{}), r.state_hash = rd(48, uint64_t{});
    r.fingerprint = std::string(reinterpret_cast<const char *>(h.data() + kFpAt), strnlen(reinterpret_cast<const char *>(h.data() + kFpAt), kFpBytes));
    STRIX_CHECK(r.fingerprint == fingerprint, path, ": saved for '", r.fingerprint, "', this server is '", fingerprint, "'");
    STRIX_CHECK(r.kind >= 1 && r.kind <= 3, path, ": kind ", r.kind, ", expected 1 (system), 2 (turn) or 3 (checkpoint)");
    STRIX_CHECK(r.tokens >= 1 && r.tokens < (1ull << 24), path, ": ", r.tokens, " tokens");
    STRIX_CHECK(r.base_n < r.tokens && (r.base_n == 0 || r.kind != 1), path, ": base of ", r.base_n, " tokens for a kind ",
                r.kind, " entry of ", r.tokens, " (a delta is a turn or a checkpoint, on a shorter base)");
    STRIX_CHECK(r.state_at == align4k(kHeaderBytes + r.tokens * 4) && r.state_at + r.state_bytes == file_bytes, path,
                ": layout says ", r.state_at, " + ", r.state_bytes, " bytes, the file has ", file_bytes);
    return r;
}


// The writer thread's I/O at idle priority (ioprio_set, IOPRIO_CLASS_IDLE on this thread): the n-gram table's reads
// on the same SSD go first. Only effective under an I/O scheduler that honours priorities (bfq); logged either way.
void set_idle_io_priority() {
    constexpr int kWhoProcess = 1, kClassIdle = 3, kClassShift = 13;
    const long r = ::syscall(SYS_ioprio_set, kWhoProcess, 0, kClassIdle << kClassShift);
    if (r != 0)
        slog(LogLevel::Warning, "prompt cache: the writer's idle I/O priority wasn't set: %s", std::strerror(errno));
}

}  // namespace

// /proc/meminfo's MemAvailable in bytes; throws if it can't be read.
uint64_t proc_mem_available() {
    std::ifstream f("/proc/meminfo");
    STRIX_CHECK(f.good(), "PromptCache: can't open /proc/meminfo");
    std::string key;
    uint64_t kib = 0;
    std::string unit;
    while (f >> key >> kib >> unit)
        if (key == "MemAvailable:") return kib << 10;
    STRIX_FAIL("PromptCache: /proc/meminfo has no MemAvailable line");
}

PromptCache::PromptCache(std::string dir, uint64_t max_bytes, std::string fingerprint)
    : PromptCache(std::move(dir), max_bytes, std::move(fingerprint), Options{}) {}

PromptCache::PromptCache(std::string dir, uint64_t max_bytes, std::string fingerprint, Options options)
    : dir_(std::move(dir)), fingerprint_(std::move(fingerprint)), max_bytes_(max_bytes), opt_(std::move(options)) {
    STRIX_CHECK(!dir_.empty(), "PromptCache: empty directory");
    STRIX_CHECK(max_bytes >= (1ull << 20), "PromptCache: max_bytes ", max_bytes, ", expected >= 1 MiB");
    STRIX_CHECK(!fingerprint_.empty() && fingerprint_.size() < kFpBytes, "PromptCache: fingerprint of ",
                fingerprint_.size(), " chars, expected 1..", kFpBytes - 1);
    STRIX_CHECK(std::isfinite(opt_.idle_seconds) && opt_.idle_seconds > 0, "PromptCache: idle_seconds ",
                opt_.idle_seconds, ", expected > 0");
    STRIX_CHECK(std::isfinite(opt_.ram_watch_seconds) && opt_.ram_watch_seconds > 0 && opt_.ram_watch_seconds <= 60,
                "PromptCache: ram_watch_seconds ", opt_.ram_watch_seconds, ", expected 0 < s <= 60");
    STRIX_CHECK(opt_.ram_watch_headroom < (1ull << 40), "PromptCache: ram_watch_headroom ", opt_.ram_watch_headroom,
                " bytes, expected < 1 TiB");
    STRIX_CHECK(std::isfinite(opt_.write_gib_per_hour) && opt_.write_gib_per_hour >= 0,
                "PromptCache: write_gib_per_hour ", opt_.write_gib_per_hour, ", expected >= 0 (0 = no limit)");
    STRIX_CHECK(std::isfinite(opt_.shutdown_seconds) && opt_.shutdown_seconds >= 0, "PromptCache: shutdown_seconds ",
                opt_.shutdown_seconds, ", expected >= 0");
    STRIX_CHECK(opt_.checkpoints_per_chain >= 1, "PromptCache: checkpoints_per_chain ", opt_.checkpoints_per_chain,
                ", expected >= 1");
    STRIX_CHECK(opt_.max_spares >= 0 && opt_.max_spares <= kMaxSpares, "PromptCache: max_spares ", opt_.max_spares,
                ", expected 0..", kMaxSpares);
    fs::create_directories(dir_);
    // Probe both once, so a broken probe fails here and not on the request path.
    (void)mem_available();
    (void)disk_free();
    budget_bytes_ = opt_.write_gib_per_hour * (double)(1ull << 30), budget_at_ = now_s();
    // The disk index: every entry's header and tokens. Anything that doesn't check out goes.
    std::vector<fs::path> files;
    for (const auto &de : fs::directory_iterator(dir_)) files.push_back(de.path());
    std::sort(files.begin(), files.end());
    const double t = now_s();
    std::vector<std::pair<EntryPtr, uint64_t>> deltas;  // (entry, base_n): linked to their bases below
    for (const fs::path &p : files) {
        const std::string path = p.string();
        if (p.extension() == ".partial") {  // a write that never finished
            fs::remove(p);
            continue;
        }
        if (p.extension() != ".pcache") continue;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        try {
            STRIX_CHECK(fd >= 0, path, ": open failed: ", std::strerror(errno));
            const uint64_t size = fs::file_size(p);
            const Header h = read_header(fd, path, fingerprint_, size);
            auto e = std::make_shared<Entry>();
            e->tokens.resize(h.tokens);
            read_all(fd, e->tokens.data(), e->tokens.size() * 4, kHeaderBytes, path);
            STRIX_CHECK(strix_hash64(e->tokens.data(), e->tokens.size() * 4) == h.token_hash, path, ": token hash mismatch");
            e->kind = (Kind)h.kind, e->last_used = ++clock_, e->touched_s = t, e->path = path, e->file_bytes = size;
            if (h.base_n > 0) deltas.emplace_back(e, h.base_n);
            entries_.push_back(std::move(e));
            disk_bytes_ += size;
        } catch (const std::exception &e) {
            slog(LogLevel::Warning, "prompt cache: dropping %s", e.what());
            fs::remove(p);
            ++stats_.invalid;
        }
        if (fd >= 0) ::close(fd);
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        // Deltas onto their bases: the whole entry of the delta's first base_n tokens. Without it a delta is useless.
        for (const auto &[d, base_n] : deltas) {
            const int i = find_locked(d->tokens.data(), (int64_t)base_n);
            if (i >= 0 && entries_[(size_t)i]->base == nullptr) {
                d->base = entries_[(size_t)i];
                ++d->base->dependents;
                continue;
            }
            slog(LogLevel::Warning, "prompt cache: dropping %s: a delta whose %s-token base isn't there", d->path.c_str(),
                 fmt_n((long long)base_n).c_str());
            remove_locked((size_t)find_locked(d.get()));
            ++stats_.invalid, ++stats_.deltas_dropped;
        }
        (void)disk_room_locked(0, UINT64_MAX);  // trims the directory to the cap (and the free-space margin)
    }
    thread_ = std::thread([this] { writer(); });
}

PromptCache::~PromptCache() {
    // The shutdown rule: RAM-only entries go to disk, newest first, no budget, until shutdown_seconds are up.
    const double t0 = now_s();
    std::unique_lock<std::mutex> lock(mu_);
    shutting_down_ = true;
    std::vector<EntryPtr> todo;
    for (const EntryPtr &e : entries_)
        if (!e->ram.empty() && e->path.empty() && !e->queued && !e->ram_only) todo.push_back(e);
    std::sort(todo.begin(), todo.end(), [](const EntryPtr &a, const EntryPtr &b) { return a->last_used > b->last_used; });
    for (const EntryPtr &e : todo)
        if (!e->queued) enqueue_locked(e, Why::Shutdown);  // (a delta's enqueue may have queued its base already)
    cv_.notify_all();
    const bool done = idle_cv_.wait_for(lock, std::chrono::duration<double>(opt_.shutdown_seconds),
                                        [&] { return queue_.empty() && !writing_; });
    if (!done)
        slog(LogLevel::Warning, "prompt cache: shutdown flush stopped after %s s, %s writes not done",
                     fmt_rate(now_s() - t0, 0).c_str(), fmt_n((long long)queue_.size()).c_str());
    for (Job &j : queue_) j.e->queued = false;
    queue_.clear(), pending_bytes_ = 0;
    stop_ = true;
    lock.unlock();
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

uint64_t PromptCache::mem_available() const {
    return opt_.mem_available ? opt_.mem_available() : proc_mem_available();
}

uint64_t PromptCache::disk_free() const {
    if (opt_.disk_free) return opt_.disk_free();
    struct statvfs s {};
    STRIX_CHECK(::statvfs(dir_.c_str(), &s) == 0, "PromptCache: statvfs('", dir_, "') failed: ", std::strerror(errno));
    return (uint64_t)s.f_bavail * (uint64_t)s.f_frsize;
}

int PromptCache::find_locked(const int32_t *tokens, int64_t n) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if ((int64_t)entries_[i]->tokens.size() == n && std::memcmp(entries_[i]->tokens.data(), tokens, (size_t)n * 4) == 0)
            return (int)i;
    return -1;
}

int PromptCache::find_locked(const Entry *e) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].get() == e) return (int)i;
    return -1;
}

int64_t PromptCache::best(const std::vector<int32_t> &prompt, int64_t max_n) const {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t best = 0;
    for (const EntryPtr &e : entries_) {
        const int64_t n = (int64_t)e->tokens.size();
        if (n <= best || n > max_n || n > (int64_t)prompt.size()) continue;
        if (std::memcmp(e->tokens.data(), prompt.data(), (size_t)n * 4) == 0) best = n;
    }
    return best;
}

PromptCache::Nearest PromptCache::nearest(const std::vector<int32_t> &prompt) const {
    std::lock_guard<std::mutex> lock(mu_);
    Nearest r;
    for (const EntryPtr &e : entries_) {
        const int64_t m = std::min((int64_t)e->tokens.size(), (int64_t)prompt.size());
        if (m <= r.common) continue;  // can't share more than the best so far
        int64_t k = 0;
        while (k < m && e->tokens[(size_t)k] == prompt[(size_t)k]) ++k;
        if (k > r.common) r = Nearest{k, (int64_t)e->tokens.size(), e->kind, !e->ram.empty()};
    }
    return r;
}

bool PromptCache::contains(const int32_t *tokens, int64_t n) const {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::contains: ", n, " tokens at ", (const void *)tokens);
    std::lock_guard<std::mutex> lock(mu_);
    return find_locked(tokens, n) >= 0;
}

bool PromptCache::in_ram(const int32_t *tokens, int64_t n) const {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::in_ram: ", n, " tokens at ", (const void *)tokens);
    std::lock_guard<std::mutex> lock(mu_);
    const int i = find_locked(tokens, n);
    return i >= 0 && !entries_[(size_t)i]->ram.empty();
}

bool PromptCache::on_disk(const int32_t *tokens, int64_t n) const {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::on_disk: ", n, " tokens at ", (const void *)tokens);
    std::lock_guard<std::mutex> lock(mu_);
    const int i = find_locked(tokens, n);
    return i >= 0 && !entries_[(size_t)i]->path.empty();
}

void PromptCache::release_locked(Entry &e) {
    STRIX_CHECK(e.readers > 0, "PromptCache::release_locked: an entry of ", e.tokens.size(), " tokens has no reader");
    if (--e.readers == 0 && e.dead) drop_ram_locked(e, true);
}

void PromptCache::load_file(const std::string &path, const std::vector<int32_t> &prompt, int64_t n,
                            HostBuffer &state) const {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, path, ": open failed: ", std::strerror(errno));
    try {
        const Header h = read_header(fd, path, fingerprint_, fs::file_size(path));
        STRIX_CHECK((int64_t)h.tokens == n, path, ": ", h.tokens, " tokens, the index said ", n);
        std::vector<int32_t> tokens(h.tokens);
        read_all(fd, tokens.data(), tokens.size() * 4, kHeaderBytes, path);
        STRIX_CHECK(std::memcmp(tokens.data(), prompt.data(), (size_t)n * 4) == 0, path, ": tokens changed on disk");
        state.resize(h.state_bytes);
        read_all(fd, state.data(), state.size(), h.state_at, path);
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // don't keep GBs of it in the page cache (n-gram table's room)
        STRIX_CHECK(strix_hash64(state.data(), state.size()) == h.state_hash, path, ": state checksum mismatch");
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
}

bool PromptCache::with_state(const std::vector<int32_t> &prompt, int64_t n,
                             const std::function<void(const HostBuffer &)> &use, bool *from_ram) {
    STRIX_CHECK((bool)use, "PromptCache::with_state: no function to call");
    return with_state(prompt, n, [&](const HostBuffer &state, int64_t from, int64_t) {
        STRIX_CHECK(from == 0, "PromptCache::with_state: the entry of ", n, " tokens is a delta (on ", from,
                    " tokens) - this caller takes whole states only");
        use(state);
    }, from_ram);
}

bool PromptCache::with_state(const std::vector<int32_t> &prompt, int64_t n,
                             const std::function<void(const HostBuffer &, int64_t, int64_t)> &use, bool *from_ram) {
    STRIX_CHECK(n >= 1 && n <= (int64_t)prompt.size(), "PromptCache::with_state: n = ", n, " of a ", prompt.size(),
                "-token prompt");
    STRIX_CHECK((bool)use, "PromptCache::with_state: no function to call");
    if (from_ram) *from_ram = false;
    // The parts to hand over, in import order: a delta's base, then the delta; else the entry alone. A part in RAM
    // is held by its reader count until the end; one on disk is read into a spare buffer first.
    struct Part {
        EntryPtr e;
        std::string path;  // empty: RAM
        HostBuffer loaded;
        uint64_t file_bytes = 0;  // the file's size (a disk part): what the load needs in RAM, about
    };
    std::vector<Part> parts;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const int i = find_locked(prompt.data(), n);
        if (i < 0) return false;  // gone since best()
        const EntryPtr e = entries_[(size_t)i];
        if (e->base) parts.push_back({e->base, {}, {}});
        parts.push_back({e, {}, {}});
        for (Part &p : parts) {
            p.e->last_used = ++clock_, p.e->touched_s = now_s(), p.e->no_room = false;
            STRIX_CHECK(!p.e->ram.empty() || !p.e->path.empty(), "PromptCache::with_state: the entry of ",
                        p.e->tokens.size(), " tokens is neither in RAM nor on disk");
            if (!p.e->ram.empty()) ++p.e->readers;
            else p.path = p.e->path, p.file_bytes = p.e->file_bytes;
        }
    }
    // Every RAM part's reader count goes back down however this ends; loaded buffers are promoted or kept as spares.
    struct Done {
        PromptCache &c;
        std::vector<Part> &parts;
        bool loaded_ok = false;
        ~Done() {
            std::vector<HostBuffer> trash;
            std::lock_guard<std::mutex> lock(c.mu_);
            for (Part &p : parts) {
                if (p.path.empty()) {
                    c.release_locked(*p.e);
                } else if (p.loaded.capacity() > 0) {
                    if (loaded_ok && !p.e->dead && p.e->ram.empty()) {  // promoted: the next hit is a RAM hit
                        c.ram_bytes_ += p.loaded.capacity();
                        p.e->ram = std::move(p.loaded);
                    } else {
                        c.keep_spare_locked(std::move(p.loaded));
                    }
                }
            }
            if (loaded_ok) c.make_room_locked();
            trash.swap(c.trash_);
        }
    } done{*this, parts};
    const double t0 = now_s();
    bool any_disk = false;
    for (Part &p : parts) {
        if (p.path.empty()) continue;
        any_disk = true;
        p.loaded = take_buffer();
        // Room first, the margin counted at the allocation (make_room_for): refused, the prompt prefills instead.
        if (p.file_bytes > p.loaded.capacity() && !make_room_for(p.file_bytes - p.loaded.capacity())) {
            slog(LogLevel::Warning, "prompt cache: not loading the %s-token entry from disk: no room in RAM (needs %s GB "
                 "more; MemAvailable %s GiB, margin %s GiB)", fmt_n((long long)p.e->tokens.size()).c_str(),
                 fmt_rate((p.file_bytes - p.loaded.capacity()) / 1e9, 1).c_str(),
                 fmt_rate(mem_available() / (double)(1ull << 30), 1).c_str(),
                 fmt_rate(opt_.ram_margin / (double)(1ull << 30), 0).c_str());
            return false;
        }
        try {
            load_file(p.path, prompt, (int64_t)p.e->tokens.size(), p.loaded);
        } catch (const std::exception &ex) {
            slog(LogLevel::Warning, "prompt cache: dropping a damaged entry: %s", ex.what());
            std::lock_guard<std::mutex> lock(mu_);
            if (const int k = find_locked(p.e.get()); k >= 0) remove_locked((size_t)k);  // a base takes its deltas
            ++stats_.load_failures, ++stats_.invalid;
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (any_disk) ++stats_.loads, stats_.load_seconds += now_s() - t0;
        else ++stats_.ram_hits;
    }
    done.loaded_ok = true;
    if (from_ram) *from_ram = !any_disk;
    for (size_t k = 0; k < parts.size(); ++k) {
        const Part &p = parts[k];
        const int64_t from = parts.size() == 2 && k == 1 ? (int64_t)parts[0].e->tokens.size() : 0;  // the delta
        use(p.path.empty() ? p.e->ram : p.loaded, from, (int64_t)p.e->tokens.size());
    }
    return true;
}

HostBuffer PromptCache::take_buffer() {
    std::lock_guard<std::mutex> lock(mu_);
    // Near the margin with no spare big enough for the next state: take the buffer of the least recently used RAM
    // entry that's already on disk (it stays there) instead of letting the export map a fresh one - a fresh
    // MADV_HUGEPAGE mapping under memory pressure compacts in the faulting thread (OCtest run 4: exports ~3 GB/s,
    // 75.7 s on the request path, once the RAM tier was full).
    const uint64_t want = prefault_bytes_;
    bool spare_fits = false;
    for (const HostBuffer &b : spares_) spare_fits |= b.capacity() >= want;
    if (want > 0 && !spare_fits && mem_available() < opt_.ram_margin + want) {
        Entry *lru = nullptr;
        for (const EntryPtr &e : entries_)
            if (e->ram.capacity() >= want && !e->path.empty() && e->readers == 0 && !e->queued &&
                (!lru || e->last_used < lru->last_used))
                lru = e.get();
        if (lru) {
            ++stats_.ram_evicted, ++stats_.buffers_reused;
            ram_bytes_ -= lru->ram.capacity();
            return std::move(lru->ram);
        }
    }
    if (spares_.empty()) return HostBuffer();
    size_t big = 0;
    for (size_t i = 1; i < spares_.size(); ++i)
        if (spares_[i].capacity() > spares_[big].capacity()) big = i;
    HostBuffer b = std::move(spares_[big]);
    spares_.erase(spares_.begin() + (int64_t)big);
    spare_bytes_ -= b.capacity();
    return b;
}

HostBuffer PromptCache::take_replaced_buffer(const int32_t *tokens, int64_t n, int64_t keep_n) {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::take_replaced_buffer: ", n, " tokens at ", (const void *)tokens);
    STRIX_CHECK(keep_n >= 0 && keep_n < n, "PromptCache::take_replaced_buffer: keep_n = ", keep_n, ", expected 0..", n - 1);
    std::vector<HostBuffer> trash;
    std::lock_guard<std::mutex> lock(mu_);
    int best = -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry &e = *entries_[i];
        if (e.kind != Kind::Turn || (int64_t)e.tokens.size() >= n || e.ram.empty() || e.readers > 0 || e.queued ||
            e.dependents > 0 || (int64_t)e.tokens.size() == keep_n)  // a base stays: its deltas need it
            continue;
        if (best >= 0 && e.tokens.size() <= entries_[(size_t)best]->tokens.size()) continue;
        if (std::memcmp(e.tokens.data(), tokens, e.tokens.size() * 4) == 0) best = (int)i;
    }
    if (best < 0) return HostBuffer();
    Entry &e = *entries_[(size_t)best];
    ram_bytes_ -= e.ram.capacity();
    HostBuffer b = std::move(e.ram);
    remove_locked((size_t)best);  // its RAM is already out: only the index and a disk file go
    ++stats_.replaced, ++stats_.turn_buffers_reused;
    trash.swap(trash_);
    return b;
}

void PromptCache::give_back(HostBuffer b) {
    std::vector<HostBuffer> trash;
    std::lock_guard<std::mutex> lock(mu_);
    keep_spare_locked(std::move(b));
    trash.swap(trash_);
}

void PromptCache::keep_spare_locked(HostBuffer b) {
    if (b.capacity() == 0) return;
    spare_bytes_ += b.capacity();
    spares_.push_back(std::move(b));
    while ((int)spares_.size() > opt_.max_spares) {  // the smallest goes
        size_t small = 0;
        for (size_t i = 1; i < spares_.size(); ++i)
            if (spares_[i].capacity() < spares_[small].capacity()) small = i;
        spare_bytes_ -= spares_[small].capacity();
        trash_.push_back(std::move(spares_[small]));
        spares_.erase(spares_.begin() + (int64_t)small);
    }
}

void PromptCache::drop_ram_locked(Entry &e, bool to_spare) {
    if (e.ram.capacity() == 0) return;
    ram_bytes_ -= e.ram.capacity();
    HostBuffer b = std::move(e.ram);
    if (to_spare) keep_spare_locked(std::move(b));
    else trash_.push_back(std::move(b));
}

void PromptCache::unlink_locked(Entry &e) {
    if (e.path.empty()) return;
    std::error_code ec;
    fs::remove(e.path, ec);
    disk_bytes_ -= e.file_bytes;
    e.path.clear(), e.file_bytes = 0;
}

void PromptCache::remove_locked(size_t i) {
    STRIX_CHECK(i < entries_.size(), "PromptCache::remove_locked: index ", i, " of ", entries_.size());
    EntryPtr e = entries_[i];
    entries_.erase(entries_.begin() + (int64_t)i);
    e->dead = true;
    unlink_locked(*e);
    if (e->readers == 0) drop_ram_locked(*e, true);  // else the last reader does
    if (e->base) {  // a delta lets go of its base
        --e->base->dependents;
        e->base.reset();
    }
    // A base takes its deltas with it: they're useless without it. (Deltas never have deltas, so removing one erases
    // only its own index: walking down from the end stays valid.)
    for (size_t k = entries_.size(); e->dependents > 0 && k-- > 0;)
        if (entries_[k]->base == e) {
            ++stats_.deltas_dropped;
            remove_locked(k);
        }
    STRIX_CHECK(e->dependents == 0, "PromptCache::remove_locked: a removed base still has ", e->dependents, " deltas");
}

int64_t PromptCache::delta_base(const int32_t *tokens, int64_t n, Kind kind) const {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::delta_base: ", n, " tokens at ", (const void *)tokens);
    if (kind == Kind::System || opt_.delta_max_tokens <= 0) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    int64_t best = 0;
    for (const EntryPtr &e : entries_) {
        const int64_t b = (int64_t)e->tokens.size();
        if (e->base || e->dead || b <= best || b >= n) continue;  // whole entries only; deltas don't chain
        // A Checkpoint's base is never a Turn: the next turn replaces a Turn, and the checkpoint would go with it.
        if (kind == Kind::Checkpoint && e->kind == Kind::Turn) continue;
        if (std::memcmp(e->tokens.data(), tokens, (size_t)b * 4) == 0) best = b;
    }
    // Worth it while the delta stays small against the base (a quarter) and in absolute terms; else save whole.
    if (best == 0 || n - best > opt_.delta_max_tokens || (n - best) * 4 > best) return 0;
    return best;
}

bool PromptCache::put(std::vector<int32_t> tokens, Kind kind, HostBuffer state, bool ram_only, int64_t base_n,
                      int64_t saved_by) {
    STRIX_CHECK(saved_by >= 0, "PromptCache::put: saved_by request id ", saved_by, ", expected >= 0");
    STRIX_CHECK(!tokens.empty() && !state.empty(), "PromptCache::put: ", tokens.size(), " tokens, ", state.size(), " state bytes");
    STRIX_CHECK(kind == Kind::System || kind == Kind::Turn || kind == Kind::Checkpoint, "PromptCache::put: kind ",
                (uint32_t)kind, ", expected 1..3");
    STRIX_CHECK(base_n >= 0 && base_n < (int64_t)tokens.size() && (base_n == 0 || kind != Kind::System),
                "PromptCache::put: base of ", base_n, " tokens for a kind ", (uint32_t)kind, " entry of ", tokens.size(),
                " (a delta is a Turn or a Checkpoint on a shorter base)");
    std::vector<HostBuffer> trash;
    std::lock_guard<std::mutex> lock(mu_);
    EntryPtr base;
    if (base_n > 0) {
        const int b = find_locked(tokens.data(), base_n);
        if (b < 0 || entries_[(size_t)b]->base) {  // gone (or replaced) since delta_base: the delta is useless
            slog(LogLevel::Warning, "prompt cache: dropped a %s-token delta: its %s-token base is gone",
                 fmt_n((long long)tokens.size()).c_str(), fmt_n(base_n).c_str());
            keep_spare_locked(std::move(state));
            trash.swap(trash_);
            return false;
        }
        base = entries_[(size_t)b];
        base->last_used = ++clock_, base->touched_s = now_s();
    }
    if (const int i = find_locked(tokens.data(), (int64_t)tokens.size()); i >= 0) {
        Entry &e = *entries_[(size_t)i];
        e.last_used = ++clock_, e.touched_s = now_s(), e.no_room = false;  // already there
        if (!ram_only) e.ram_only = false;  // a request that may continue it wants it on disk too
        if (e.ram.empty() && !e.dead) {  // on disk only: this copy saves the next hit a read
            ram_bytes_ += state.capacity();
            e.ram = std::move(state);
            make_room_locked();
        } else {
            keep_spare_locked(std::move(state));
        }
        trash.swap(trash_);
        return true;
    }
    // A new turn or checkpoint replaces the turns it extends (the same conversation, one step further); checkpoints
    // stay. In RAM this is where most of the old per-turn disk writes disappear.
    const size_t n = tokens.size();
    // (Collected first: removing a base also removes its deltas, which shifts indices.)
    if (kind == Kind::Turn || kind == Kind::Checkpoint) {
        std::vector<Entry *> replaced;
        for (const EntryPtr &e : entries_)
            if (e->kind == Kind::Turn && e != base && e->tokens.size() < n &&
                std::memcmp(e->tokens.data(), tokens.data(), e->tokens.size() * 4) == 0)
                replaced.push_back(e.get());
        for (Entry *r : replaced)
            if (const int k = find_locked(r); k >= 0) {  // (a delta may have gone with its base already)
                remove_locked((size_t)k);
                ++stats_.replaced;
            }
    }
    // Checkpoints per conversation: the older ones this one extends, beyond the newest checkpoints_per_chain - 1, go.
    if (kind == Kind::Checkpoint) {
        std::vector<std::pair<size_t, Entry *>> chain;  // (length, entry)
        for (const EntryPtr &e : entries_)
            if (e->kind == Kind::Checkpoint && e->tokens.size() < n &&
                std::memcmp(e->tokens.data(), tokens.data(), e->tokens.size() * 4) == 0)
                chain.emplace_back(e->tokens.size(), e.get());
        std::sort(chain.begin(), chain.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        // Longest first, so a delta goes before its base. A base stays while a delta needs it - this put's own base
        // and the base of a checkpoint delta kept above (Checkpoint deltas, 2026-10-08); it goes on a later trim.
        for (size_t k = (size_t)opt_.checkpoints_per_chain - 1; k < chain.size(); ++k) {
            if (chain[k].second == base.get() || chain[k].second->dependents > 0) continue;
            const int i = find_locked(chain[k].second);
            if (i < 0) continue;  // (not expected: only a base takes others with it, and bases are skipped)
            remove_locked((size_t)i);
            ++stats_.checkpoints_dropped;
        }
    }
    // The spares to keep ready follow whole states (a delta's few hundred MB would shrink them below the next rebase).
    if (!base) prefault_bytes_ = state.size() + opt_.prefault_headroom;
    auto e = std::make_shared<Entry>();
    e->tokens = std::move(tokens), e->kind = kind, e->last_used = ++clock_, e->touched_s = now_s();
    e->ram_only = ram_only, e->saved_by = saved_by;
    if (base) {
        e->base = base, ++base->dependents;
        ++stats_.delta_puts, stats_.delta_bytes += (int64_t)state.size();
    }
    ram_bytes_ += state.capacity();
    e->ram = std::move(state);
    entries_.push_back(std::move(e));
    make_room_locked();
    trash.swap(trash_);
    if (opt_.prefault) cv_.notify_one();  // the writer tops the spares up
    return true;
}

double PromptCache::budget_now_locked() const {
    const double cap = opt_.write_gib_per_hour * (double)(1ull << 30);
    return std::min(cap, budget_bytes_ + (now_s() - budget_at_) * cap / 3600.0);
}

bool PromptCache::spend_budget_locked(uint64_t bytes) {
    if (opt_.write_gib_per_hour <= 0) return true;
    budget_bytes_ = budget_now_locked(), budget_at_ = now_s();
    if (budget_bytes_ < (double)bytes) return false;
    budget_bytes_ -= (double)bytes;
    return true;
}

bool PromptCache::disk_room_locked(uint64_t need, uint64_t last_used) {
    const uint64_t free = disk_free();
    // The cap: max_bytes, and never less than disk_free_margin left on the filesystem.
    const uint64_t fs_cap = disk_bytes_ + (free > opt_.disk_free_margin ? free - opt_.disk_free_margin : 0);
    const uint64_t cap = std::min(max_bytes_, fs_cap);
    if (need > cap) return false;
    // Enough to evict? Only entries used less recently than the one being written count.
    uint64_t evictable = 0;
    for (const EntryPtr &e : entries_)
        if (!e->path.empty() && e->last_used < last_used) evictable += e->file_bytes;
    if (disk_bytes_ + need > cap + evictable) return false;
    while (disk_bytes_ + need > cap) {
        Entry *lru = nullptr;
        for (const EntryPtr &e : entries_)
            if (!e->path.empty() && e->last_used < last_used && (!lru || e->last_used < lru->last_used)) lru = e.get();
        STRIX_CHECK(lru != nullptr, "PromptCache: ", disk_bytes_, " bytes on disk, ", need, " more wanted under a cap of ",
                    cap, ", and no entry left to evict (the evictable count said ", evictable, ")");
        ++stats_.evicted;
        if (!lru->ram.empty() || lru->queued) unlink_locked(*lru), lru->no_room = true;  // RAM keeps it
        else remove_locked((size_t)find_locked(lru));
    }
    return true;
}

uint64_t PromptCache::write_bytes_locked(const Entry &e) const {
    uint64_t b = file_total(e.tokens.size(), e.ram.size());
    if (e.base && e.base->path.empty() && !e.base->queued) b += file_total(e.base->tokens.size(), e.base->ram.size());
    return b;
}

void PromptCache::enqueue_locked(const EntryPtr &e, Why why) {
    // A delta reaches the disk only with its base: the base's write goes first (the queue is FIFO).
    if (e->base && e->base->path.empty() && !e->base->queued) {
        STRIX_CHECK(!e->base->ram.empty(), "PromptCache: a delta's base of ", e->base->tokens.size(),
                    " tokens is neither on disk nor in RAM");
        enqueue_locked(e->base, why);
    }
    e->queued = true;
    const uint64_t bytes = e->ram.size();
    pending_bytes_ += bytes;
    queue_.push_back({e, why, bytes});
    cv_.notify_one();
}

bool PromptCache::make_room_for(uint64_t bytes) {
    STRIX_CHECK(bytes < (1ull << 40), "PromptCache::make_room_for: ", bytes, " bytes, expected < 1 TiB");
    if (bytes == 0) return true;
    {
        std::vector<HostBuffer> trash;
        std::lock_guard<std::mutex> lock(mu_);
        make_room_locked(bytes);
        trash.swap(trash_);
    }  // the evicted buffers are unmapped here, before MemAvailable is read again
    const uint64_t avail = mem_available();
    if (avail >= opt_.ram_margin + bytes) return true;
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.room_refused;
    return false;
}

void PromptCache::make_room_locked(uint64_t extra) {
    const uint64_t avail = mem_available(), target = opt_.ram_margin + extra;
    if (avail >= target && opt_.ram_tier) return;
    // No RAM tier: every entry that can leave does (need never reaches 0 below).
    uint64_t need = opt_.ram_tier ? target - avail : UINT64_MAX;
    // Spares go first: nothing is lost.
    while (need > 0 && !spares_.empty()) {
        need -= std::min(need, (uint64_t)spares_.back().capacity());
        spare_bytes_ -= spares_.back().capacity();
        trash_.push_back(std::move(spares_.back()));
        spares_.pop_back();
    }
    while (need > 0) {
        // Candidates: every RAM entry not being imported and not already leaving. One queued for a write (idle, or
        // being written) counts too - it leaves RAM once written - so the least recently used goes first, never the
        // newest just because the older ones sat in the write queue (the margin switch test, 5848183: the entry just
        // saved was dropped, "write queue full", and its conversation re-prefilled 132k tokens).
        Entry *lru = nullptr;
        for (const EntryPtr &e : entries_)
            if (!e->ram.empty() && !e->drop_ram && (e->readers == 0 || e->queued) && (!lru || e->last_used < lru->last_used))
                lru = e.get();
        if (!lru) break;  // the rest is in use or on its way out; the next put tries again
        if (opt_.ram_tier) need -= std::min(need, (uint64_t)lru->ram.capacity());
        ++stats_.ram_evicted;
        if (lru->queued) {  // written soon: out of RAM after that
            lru->drop_ram = true;
            continue;
        }
        if (!lru->path.empty()) {  // on disk already
            drop_ram_locked(*lru, false);
            continue;
        }
        if (lru->ram_only) {  // never written: gone
            ++stats_.ram_only_dropped;
            remove_locked((size_t)find_locked(lru));
            continue;
        }
        // The eviction rule: write it, unless the queue, the budget or the disk says no - then it's gone.
        const uint64_t total = write_bytes_locked(*lru);
        const uint64_t free = disk_free();
        const uint64_t fs_cap = disk_bytes_ + (free > opt_.disk_free_margin ? free - opt_.disk_free_margin : 0);
        const uint64_t queue_cap = opt_.max_pending_bytes ? opt_.max_pending_bytes
                                                            : std::max(kMaxPendingBytes, 4 * prefault_bytes_);
        const char *no = pending_bytes_ + lru->ram.size() > queue_cap ? "queue"
                         : total > std::min(max_bytes_, fs_cap)               ? "space"
                         : !spend_budget_locked(total)                        ? "budget"
                                                                              : nullptr;
        if (no == nullptr) {
            lru->drop_ram = true;
            for (const EntryPtr &e : entries_)
                if (e.get() == lru) enqueue_locked(e, Why::Evict);
            continue;
        }
        (no[0] == 'q' ? stats_.rejected_queue : no[0] == 's' ? stats_.rejected_space : stats_.rejected_budget)++;
        // What was lost, why it had to leave RAM, and why the disk said no. Long unused: a conversation that's likely
        // over (an info line); recent: one that may come back to a full prefill (a warning).
        const double idle = now_s() - lru->touched_s;
        const LogLevel level = idle >= opt_.idle_seconds ? LogLevel::Info : LogLevel::Warning;
        const std::string by = lru->saved_by > 0 ? "saved by Request " + std::to_string(lru->saved_by) + ", " : "";
        const std::string deltas = lru->dependents > 0 ? " and its " + fmt_n((long long)lru->dependents) + " deltas" : "";
        slog(level, "prompt cache: dropped a %s entry of %s tokens%s, %s GB (%slast used %s ago)", kind_name(lru->kind),
             fmt_n((long long)lru->tokens.size()).c_str(), deltas.c_str(), fmt_rate(lru->ram.size() / 1e9, 1).c_str(),
             by.c_str(), ago(idle).c_str());
        if (opt_.ram_tier)
            slog(level, "%sit had to leave RAM (MemAvailable %s GiB, below the %s GiB margin) and can't go to disk:", kCont,
                 fmt_rate(avail / (double)(1ull << 30), 1).c_str(), fmt_rate(opt_.ram_margin / (double)(1ull << 30), 0).c_str());
        else
            slog(level, "%sit had to leave RAM (the RAM tier is off: low-memory adapt) and can't go to disk:", kCont);
        if (no[0] == 'q')
            slog(level, "%sthe write queue is full (%s of %s GB queued)", kCont, fmt_rate(pending_bytes_ / 1e9, 1).c_str(),
                 fmt_rate(queue_cap / 1e9, 1).c_str());
        else if (no[0] == 's')
            slog(level, "%sno disk room (needs %s GB, the cap leaves %s GB)", kCont, fmt_rate(total / 1e9, 1).c_str(),
                 fmt_rate(std::min(max_bytes_, fs_cap) / 1e9, 1).c_str());
        else {
            const double rate = opt_.write_gib_per_hour * (double)(1ull << 30) / 3600.0;  // bytes per second
            slog(level, "%sthe write budget is spent (%s of %s GiB left this hour, needs %s GB: enough in %s)", kCont,
                 fmt_rate(budget_bytes_ / (double)(1ull << 30), 1).c_str(), fmt_rate(opt_.write_gib_per_hour, 0).c_str(),
                 fmt_rate(total / 1e9, 1).c_str(), ago(((double)total - budget_bytes_) / rate).c_str());
        }
        remove_locked((size_t)find_locked(lru));
    }
}

void PromptCache::summary_locked(double now) {
    const uint64_t sig = clock_ + (uint64_t)stats_.writes + (uint64_t)stats_.ram_evicted;
    if (now - summary_at_ < kSummarySeconds || sig == summary_sig_) return;
    summary_at_ = now, summary_sig_ = sig;
    int64_t ram_n = 0, disk_n = 0;
    for (const EntryPtr &e : entries_) ram_n += !e->ram.empty(), disk_n += !e->path.empty();
    const double gib = (double)(1ull << 30);
    slog(LogLevel::Info, "prompt cache: RAM %s entries, %s GB (+ %s GB spare buffers); MemAvailable %s GiB (margin %s)%s%s",
         fmt_n(ram_n).c_str(), fmt_rate(ram_bytes_ / 1e9, 1).c_str(), fmt_rate(spare_bytes_ / 1e9, 1).c_str(),
         fmt_rate(mem_available() / gib, 1).c_str(), fmt_rate(opt_.ram_margin / gib, 0).c_str(),
         opt_.status_note.empty() ? "" : "; ", opt_.status_note.c_str());
    char budget[64] = "no write budget";
    if (opt_.write_gib_per_hour > 0)
        std::snprintf(budget, sizeof budget, "write budget %s of %s GiB left this hour",
                      fmt_rate(budget_now_locked() / gib, 1).c_str(), fmt_rate(opt_.write_gib_per_hour, 0).c_str());
    slog(LogLevel::Info, "%sdisk %s entries, %s of %s GiB; %s", kCont, fmt_n(disk_n).c_str(), fmt_rate(disk_bytes_ / gib, 1).c_str(),
         fmt_rate(max_bytes_ / gib, 0).c_str(), budget);
    slog(LogLevel::Info, "%ssince the start: %s written, %s dropped leaving RAM, %s write failures", kCont,
         fmt_n(stats_.writes).c_str(),
         fmt_n(stats_.rejected_queue + stats_.rejected_space + stats_.rejected_budget).c_str(),
         fmt_n(stats_.write_failures).c_str());
}

void PromptCache::write_idle_locked(double now) {
    if (shutting_down_) return;
    for (const EntryPtr &e : entries_) {
        if (e->ram.empty() || !e->path.empty() || e->queued || e->no_room || e->ram_only ||
            now - e->touched_s < opt_.idle_seconds)
            continue;
        const uint64_t total = write_bytes_locked(*e);
        const uint64_t free = disk_free();
        const uint64_t fs_cap = disk_bytes_ + (free > opt_.disk_free_margin ? free - opt_.disk_free_margin : 0);
        if (total > std::min(max_bytes_, fs_cap) || !spend_budget_locked(total)) continue;  // later, maybe
        enqueue_locked(e, Why::Idle);
    }
}

uint64_t PromptCache::prefault_want_locked() const {
    if (!opt_.prefault || prefault_bytes_ == 0 || shutting_down_ || stop_) return 0;
    int ready = 0;
    for (const HostBuffer &b : spares_) ready += b.capacity() >= prefault_bytes_;
    if (ready >= opt_.max_spares) return 0;
    if (mem_available() < opt_.ram_margin + prefault_bytes_) return 0;  // never into memory pressure
    return prefault_bytes_;
}

void PromptCache::write_idle() {
    std::lock_guard<std::mutex> lock(mu_);
    write_idle_locked(now_s());
}

void PromptCache::flush() {
    std::unique_lock<std::mutex> lock(mu_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
}

PromptCacheStats PromptCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PromptCacheStats s = stats_;
    s.entries = (int64_t)entries_.size();
    for (const EntryPtr &e : entries_) s.ram_entries += !e->ram.empty(), s.disk_entries += !e->path.empty();
    s.ram_bytes = (int64_t)ram_bytes_, s.spare_bytes = (int64_t)spare_bytes_;
    s.bytes = (int64_t)disk_bytes_, s.max_bytes = (int64_t)max_bytes_;
    s.pending = (int64_t)queue_.size();
    return s;
}

void PromptCache::writer() {
    set_idle_io_priority();
    // The idle rule's scan: a quarter of idle_seconds, between 0.05 and 30 s. The margin watch runs more often.
    const double scan_s = std::clamp(opt_.idle_seconds / 4, 0.05, 30.0);
    const double wake_s = std::min(scan_s, opt_.ram_watch_seconds);
    double scan_at = now_s() + scan_s, watch_at = now_s() + opt_.ram_watch_seconds;
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            // Between writes too: a queue of multi-GB writes takes seconds each.
            if (now_s() >= watch_at) watch_margin(lock), watch_at = now_s() + opt_.ram_watch_seconds;
            while (!stop_ && queue_.empty()) {
                if (const uint64_t want = prefault_want_locked()) {  // populate a spare off the request path
                    lock.unlock();
                    HostBuffer b;
                    try {
                        b.resize(want);
                    } catch (const std::exception &e) {
                        slog(LogLevel::Warning, "prompt cache: prefaulting a spare failed: %s", e.what());
                    }
                    lock.lock();
                    if (b.capacity() == 0) {  // failed: don't spin on it
                        prefault_bytes_ = 0;
                        continue;
                    }
                    keep_spare_locked(std::move(b));
                    ++stats_.prefaulted;
                    std::vector<HostBuffer> trash;
                    trash.swap(trash_);
                    lock.unlock();
                    trash.clear();
                    lock.lock();
                    continue;
                }
                if (cv_.wait_for(lock, std::chrono::duration<double>(wake_s)) == std::cv_status::timeout) {
                    const double now = now_s();
                    if (now >= watch_at) {
                        watch_margin(lock), watch_at = now_s() + opt_.ram_watch_seconds;
                        if (stop_ || !queue_.empty()) break;
                    }
                    if (now >= scan_at) {
                        write_idle_locked(now);
                        summary_locked(now);
                        scan_at = now + scan_s;
                    }
                }
            }
            if (queue_.empty()) return;  // stop_ with nothing left
            job = std::move(queue_.front());
            queue_.pop_front();
            pending_bytes_ -= job.bytes;
            writing_ = true;
        }
        try {
            write_entry(job);
        } catch (const std::exception &e) {
            slog(LogLevel::Error, "prompt cache: write failed: %s", e.what());
            std::vector<HostBuffer> trash;
            std::lock_guard<std::mutex> lock(mu_);
            ++stats_.write_failures;
            // An eviction's entry can't stay in RAM (that's why it left) and has no file: it's gone.
            if (job.e->drop_ram && !job.e->dead)
                if (const int i = find_locked(job.e.get()); i >= 0) remove_locked((size_t)i);
            trash.swap(trash_);
        }
        {
            std::vector<HostBuffer> trash;
            std::lock_guard<std::mutex> lock(mu_);
            writing_ = false;
            job.e.reset();
            trash.swap(trash_);
        }
        idle_cv_.notify_all();
    }
}

void PromptCache::watch_margin(std::unique_lock<std::mutex> &lock) {
    STRIX_CHECK(lock.owns_lock(), "PromptCache::watch_margin: called without the cache's lock");
    // Off: no RAM tier keeps nothing to evict; shutting down, the shutdown rule writes RAM entries itself.
    if (!opt_.ram_tier || shutting_down_ || stop_) return;
    const uint64_t avail = mem_available();
    if (avail >= opt_.ram_margin) return;
    const int64_t before = stats_.ram_evicted;
    const uint64_t ram_before = ram_bytes_ + spare_bytes_;
    make_room_locked(opt_.ram_watch_headroom);
    const int64_t n = stats_.ram_evicted - before;
    stats_.ram_evicted_watch += n;
    // A line only when something left: a machine sitting below the margin with nothing left to evict stays quiet.
    const uint64_t freed = ram_before - std::min(ram_before, ram_bytes_ + spare_bytes_);
    const double gib = (double)(1ull << 30);
    if (n > 0 || freed > 0)
        slog(LogLevel::Info, "prompt cache: MemAvailable %s GiB, below the %s GiB margin between saves (another "
             "process grew): %s entries leave RAM, %s GB freed now (queued writes free theirs once written)",
             fmt_rate(avail / gib, 1).c_str(), fmt_rate(opt_.ram_margin / gib, 0).c_str(), fmt_n((long long)n).c_str(),
             fmt_rate(freed / 1e9, 1).c_str());
    std::vector<HostBuffer> trash;
    trash.swap(trash_);
    lock.unlock();
    trash.clear();  // unmapped outside the lock
    lock.lock();
}

void PromptCache::write_entry(Job &job) {
    Entry &e = *job.e;
    const double t0 = now_s();
    uint64_t n = 0, total = 0, state_at = 0, state_bytes = 0, base_n = 0;
    {
        std::vector<HostBuffer> trash;
        std::lock_guard<std::mutex> lock(mu_);
        e.queued = false;
        if (e.dead || !e.path.empty() || e.ram.empty()) return;  // replaced / written / gone meanwhile
        if (e.base && e.base->path.empty()) {  // its base's write failed or was refused: a delta file alone is useless
            ++stats_.rejected_space;
            e.no_room = true;
            if (e.drop_ram) remove_locked((size_t)find_locked(&e));
            trash.swap(trash_);
            return;
        }
        base_n = e.base ? e.base->tokens.size() : 0;
        n = e.tokens.size(), state_bytes = e.ram.size();
        state_at = align4k(kHeaderBytes + n * 4), total = state_at + state_bytes;
        if (!disk_room_locked(total, e.last_used)) {
            ++stats_.rejected_space;
            e.no_room = true;
            if (e.drop_ram) remove_locked((size_t)find_locked(&e));
            trash.swap(trash_);
            return;
        }
        ++e.readers;  // the RAM copy stays put while the file is written
        e.queued = true;
    }
    const uint64_t token_hash = strix_hash64(e.tokens.data(), n * 4);
    char name[64];
    std::snprintf(name, sizeof name, "%016llx-%llu.pcache", (unsigned long long)token_hash, (unsigned long long)n);
    const std::string path = dir_ + "/" + name, partial = path + ".partial";
    bool ok = false;
    try {
        // O_DIRECT: the state never becomes dirty page cache. Buffered, a multi-GB write under memory pressure (free
        // memory ~0 with the n-gram table cached) made the request path's allocations wait in reclaim for writeback -
        // exports 7-12 s in the warm switch test (0bc9472, idle writes on) vs 1.5-2.4 s with only eviction writes -
        // and pushed the n-gram table's pages out. Aligned pieces: [header | tokens | pad] from `head` (page-aligned),
        // the state's whole 4 KiB blocks straight from its HostBuffer, the last partial block via `head`'s spare
        // page; then the file is cut to its exact size.
        const uint64_t body = state_bytes & ~uint64_t{4095}, tail = state_bytes - body;
        HostBuffer head(state_at + 4096);
        std::memset(head.data(), 0, head.size());
        uint8_t *h = head.data();
        auto put = [&](size_t off, auto v) { std::memcpy(h + off, &v, sizeof(v)); };
        std::memcpy(h, kMagic, 8);
        put(8, kVersion), put(12, (uint32_t)e.kind), put(16, n), put(24, state_bytes), put(32, state_at);
        put(40, token_hash), put(48, strix_hash64(e.ram.data(), state_bytes));
        std::memcpy(h + kFpAt, fingerprint_.data(), fingerprint_.size());
        put(kBaseAt, base_n);
        put(kHeaderHashAt, strix_hash64(h, kHeaderHashAt));
        std::memcpy(h + kHeaderBytes, e.tokens.data(), n * 4);
        if (tail) std::memcpy(h + state_at, e.ram.data() + body, tail);
        const int fd = ::open(partial.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0644);
        STRIX_CHECK(fd >= 0, partial, ": create failed: ", std::strerror(errno));
        try {
            int flags = 0;  // compression off while empty (KV doesn't compress; a compressed extent costs reads)
            if (::ioctl(fd, FS_IOC_GETFLAGS, &flags) == 0) {
                flags |= FS_NOCOMP_FL;
                (void)::ioctl(fd, FS_IOC_SETFLAGS, &flags);
            }
            write_all(fd, h, state_at, 0, partial);
            write_all(fd, e.ram.data(), body, state_at, partial);
            if (tail) write_all(fd, h + state_at, 4096, state_at + body, partial);
            STRIX_CHECK(::ftruncate(fd, (off_t)total) == 0, partial, ": truncate to ", total, " bytes failed: ",
                        std::strerror(errno));
            STRIX_CHECK(::fsync(fd) == 0, partial, ": fsync failed: ", std::strerror(errno));
        } catch (...) {
            ::close(fd);
            fs::remove(partial);
            throw;
        }
        ::close(fd);
        fs::rename(partial, path);
        ok = true;
    } catch (...) {
        std::lock_guard<std::mutex> lock(mu_);
        e.queued = false;
        release_locked(e);
        throw;
    }
    std::vector<HostBuffer> trash;
    std::lock_guard<std::mutex> lock(mu_);
    e.queued = false;
    if (e.dead) {  // replaced while it was written: the file goes too
        std::error_code ec;
        fs::remove(path, ec);
    } else if (ok) {
        e.path = path, e.file_bytes = total;
        disk_bytes_ += total;
        const std::string by = e.saved_by > 0 ? " (saved by Request " + std::to_string(e.saved_by) + ")" : "";
        slog(LogLevel::Info, "prompt cache: wrote a %s entry of %s tokens%s to disk, %s MB in %s s", kind_name(e.kind),
             fmt_n((long long)n).c_str(), by.c_str(), fmt_rate(total / 1e6, 0).c_str(), fmt_rate(now_s() - t0, 2).c_str());
        const std::string delta = base_n > 0 ? "; a delta on the " + fmt_n((long long)base_n) + "-token entry" : "";
        slog(LogLevel::Info, "%swhy: %s%s", kCont,
             job.why == Why::Idle    ? ("unused for " + ago(now_s() - e.touched_s)).c_str()
             : job.why == Why::Evict ? "leaving RAM (MemAvailable below the margin)"
                                     : "server shutdown",
             delta.c_str());
        if (e.drop_ram) drop_ram_locked(e, true), e.drop_ram = false;
    }
    release_locked(e);
    ++stats_.writes, stats_.bytes_written += (int64_t)total, stats_.write_seconds += now_s() - t0;
    (job.why == Why::Evict ? stats_.writes_evict : job.why == Why::Idle ? stats_.writes_idle : stats_.writes_shutdown)++;
    trash.swap(trash_);
}

}  // namespace strix
