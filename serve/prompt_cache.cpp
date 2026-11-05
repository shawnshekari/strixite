#include "serve/prompt_cache.hpp"
#include "serve/log.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"  // strix_hash64

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
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
constexpr size_t kReleaseChunkBytes = 64ull << 20;  // write_all_releasing: give back a whole chunk of pages at a time
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

// write_all for a source that is being thrown away: as each chunk of `src` reaches the file, its pages go back to
// the kernel (MADV_DONTNEED, whole pages, only after every byte of theirs is written). pwrite with O_DIRECT returns
// once the bytes are at the device, and a buffered pwrite has copied them into the page cache - either way the
// source's pages are dead afterwards. A queued eviction write used to hold all its RAM until the write ended
// (~0.7 s per 1.6 GB) - the dip the margin watch chased under a fast-growing memory hog - so the memory it
// frees is freed as it is written. Returns the
// bytes given back; a madvise failure stops the releasing, not the write.
uint64_t write_all_releasing(int fd, HostBuffer &src, size_t n, uint64_t at, const std::string &where,
                             const std::function<void(uint64_t, uint64_t)> &on_progress) {
    STRIX_CHECK(fd >= 0, where, ": write_all_releasing with fd ", fd);
    STRIX_CHECK(n % 4096 == 0, where, ": write_all_releasing of ", n, " bytes, expected a whole number of pages");
    STRIX_CHECK(src.capacity() >= n, where, ": write_all_releasing of ", n, " bytes from a buffer of ", src.capacity());
    STRIX_CHECK(src.data() != nullptr || n == 0, where, ": write_all_releasing of ", n, " bytes from a null buffer");
    size_t done = 0, released = 0;
    bool releasing = true;
    while (done < n) {
        const size_t want = std::min(n - done, kReleaseChunkBytes);
        const ssize_t w = ::pwrite(fd, src.data() + done, want, (off_t)(at + done));
        if (w < 0 && errno == EINTR) continue;
        STRIX_CHECK(w > 0, where, ": write of ", want, " bytes at ", at + done, " failed: ",
                    w < 0 ? std::strerror(errno) : "0 bytes");
        done += (size_t)w;
        const size_t landed = done & ~(size_t)4095;  // pages whose every byte is in the file (a short write may end mid-page)
        if (releasing && landed > released) {
            if (::madvise(src.data() + released, landed - released, MADV_DONTNEED) != 0) releasing = false;
            else {
                released = landed;
                if (on_progress) on_progress((uint64_t)done, (uint64_t)released);
            }
        }
    }
    return (uint64_t)released;
}

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
    : dir_(std::move(dir)), fingerprint_(std::move(fingerprint)), max_bytes_(max_bytes), opt_(std::move(options)),
      gov_(opt_.ram_margin, [this] { return mem_available(); }) {
    STRIX_CHECK(!dir_.empty(), "PromptCache: empty directory");
    STRIX_CHECK(max_bytes >= (1ull << 20), "PromptCache: max_bytes ", max_bytes, ", expected >= 1 MiB");
    STRIX_CHECK(!fingerprint_.empty() && fingerprint_.size() < kFpBytes, "PromptCache: fingerprint of ",
                fingerprint_.size(), " chars, expected 1..", kFpBytes - 1);
    STRIX_CHECK(std::isfinite(opt_.idle_seconds) && opt_.idle_seconds > 0, "PromptCache: idle_seconds ",
                opt_.idle_seconds, ", expected > 0");
    STRIX_CHECK(std::isfinite(opt_.ram_watch_seconds) && opt_.ram_watch_seconds > 0 && opt_.ram_watch_seconds <= 60,
                "PromptCache: ram_watch_seconds ", opt_.ram_watch_seconds, ", expected 0 < s <= 60");
    STRIX_CHECK(std::isfinite(opt_.room_wait_seconds) && opt_.room_wait_seconds >= 0 && opt_.room_wait_seconds <= 600,
                "PromptCache: room_wait_seconds ", opt_.room_wait_seconds, ", expected 0 <= s <= 600");
    STRIX_CHECK(opt_.ram_watch_headroom < (1ull << 40), "PromptCache: ram_watch_headroom ", opt_.ram_watch_headroom,
                " bytes, expected < 1 TiB");
    STRIX_CHECK(std::isfinite(opt_.write_gib_per_hour) && opt_.write_gib_per_hour >= 0,
                "PromptCache: write_gib_per_hour ", opt_.write_gib_per_hour, ", expected >= 0 (0 = no limit)");
    STRIX_CHECK(std::isfinite(opt_.shutdown_seconds) && opt_.shutdown_seconds >= 0, "PromptCache: shutdown_seconds ",
                opt_.shutdown_seconds, ", expected >= 0");
    STRIX_CHECK(opt_.checkpoints_per_chain >= 1, "PromptCache: checkpoints_per_chain ", opt_.checkpoints_per_chain,
                ", expected >= 1");
    STRIX_CHECK(opt_.max_points >= 0, "PromptCache: max_points ", opt_.max_points, ", expected >= 0 (0 = no points)");
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
        if (e->in_ram() && e->path.empty() && !e->queued && !e->ram_only) todo.push_back(e);
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
    // The shutdown flush retires records of its own writes, and nothing drains this list after the cache is
    // gone: the engine's last drain ran earlier, so those parked pages go with the process. The backend that
    // owns them outlives this destructor (declaration order), so nothing dangles - but a shutdown that lost
    // more than the flush's own writes means the serving drain was not running, and that deserves a line.
    const std::vector<int64_t> left = take_retired();
    if (!left.empty())
        slog(LogLevel::Warning, "prompt cache: %s parked records were retired by the shutdown flush - their "
             "pages stay mapped until the process ends", fmt_n((long long)left.size()).c_str());
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
        if (k > r.common) r = Nearest{k, (int64_t)e->tokens.size(), e->kind, e->in_ram()};
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
    return i >= 0 && entries_[(size_t)i]->in_ram();
}

bool PromptCache::on_disk(const int32_t *tokens, int64_t n) const {
    STRIX_CHECK(tokens != nullptr && n >= 1, "PromptCache::on_disk: ", n, " tokens at ", (const void *)tokens);
    std::lock_guard<std::mutex> lock(mu_);
    const int i = find_locked(tokens, n);
    return i >= 0 && !entries_[(size_t)i]->path.empty();
}

void PromptCache::release_locked(Entry &e) {
    STRIX_CHECK(e.readers > 0, "PromptCache::release_locked: an entry of ", e.tokens.size(), " tokens has no reader");
    // The last user also finishes a pending leave-RAM: a removal (dead) or an eviction the writer found still read
    // (drop_ram, deferring as in remove_locked). An import_state whose base was dropped over it used to get a
    // moved-from buffer and fail the request with "null input".
    if (--e.readers == 0 && (e.dead || e.drop_ram)) {
        drop_ram_locked(e, true);
        if (!e.dead) e.drop_ram = false;  // a kept entry can come back to RAM (a disk read promotes it) and leave again
    }
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
        std::unique_lock<std::mutex> lock(mu_);
        // A part whose eviction write is giving its RAM back reads as zero pages: wait for the file (the writer
        // notifies idle_cv_ after every write), then it is a disk part. The entry may be gone after the wait.
        // The same wait covers a paged entry mid-write: its parked pages leave RAM only as the gather finishes
        // (write_entry drops both halves the moment the file is there), and until then neither half is a
        // state this path can hand - so a resume that came from with_paged's miss on a queued entry waits
        // for the write and reads the file, exactly as it does for a releasing whole entry.
        const auto waits = [](const Entry &f) { return f.releasing || (!f.paged.empty() && f.queued); };
        bool logged = false;
        const double deadline = now_s() + 120;
        for (;;) {
            const int i = find_locked(prompt.data(), n);
            if (i < 0) return false;  // gone since best()
            const EntryPtr e = entries_[(size_t)i];
            const Entry *busy = waits(*e) ? e.get() : e->base && waits(*e->base) ? e->base.get() : nullptr;
            if (busy == nullptr) {
                if (e->base) parts.push_back({e->base, {}, {}});
                parts.push_back({e, {}, {}});
                break;
            }
            if (!logged) {
                logged = true;
                slog(LogLevel::Info, "prompt cache: the %s-token entry is being written as it leaves RAM - waiting for "
                     "the file before loading it", fmt_n((long long)busy->tokens.size()).c_str());
            }
            if (now_s() > deadline) {
                slog(LogLevel::Warning, "prompt cache: gave up waiting 120 s for the %s-token entry's write",
                     fmt_n((long long)busy->tokens.size()).c_str());
                return false;
            }
            idle_cv_.wait_for(lock, std::chrono::milliseconds(50));
        }
        for (Part &p : parts) {
            p.e->last_used = ++clock_, p.e->touched_s = now_s(), p.e->no_room = false;
            STRIX_CHECK(p.e->in_ram() || !p.e->path.empty(), "PromptCache::with_state: the entry of ",
                        p.e->tokens.size(), " tokens is neither in RAM nor on disk");
            // A paged entry's small buffer without its parked pages is not a state - importing it would restore
            // a corrupt conversation - so it is refused by name, never quietly served: read it with with_paged
            // while it is in RAM. A file means the gather already ran, and the file has the whole state: that
            // is what this path reads (the write itself was waited for above; a paged entry keeps no RAM past it).
            STRIX_CHECK(p.e->paged.empty() || !p.e->path.empty(),
                        "PromptCache::with_state: the ", fmt_n((long long)p.e->tokens.size()),
                        "-token entry is paged: its state's big ranges are parked pages that only with_paged can "
                        "hand (its file, if it has one, has the whole state)");
            if (p.e->in_ram() && p.e->paged.empty()) ++p.e->readers;
            else p.path = p.e->path, p.file_bytes = p.e->file_bytes;
        }
    }
    // Every RAM part's reader count goes back down however this ends; loaded buffers are promoted or kept as spares.
    struct Done {
        PromptCache &c;
        std::vector<Part> &parts;
        bool loaded_ok = false;
        ~Done() {
            Handback hb;
            std::lock_guard<std::mutex> lock(c.mu_);
            for (Part &p : parts) {
                if (p.path.empty()) {
                    c.release_locked(*p.e);
                } else if (p.loaded.capacity() > 0) {
                    if (loaded_ok && !p.e->dead && !p.e->in_ram()) {  // promoted: the next hit is a RAM hit
                        c.ram_bytes_ += p.loaded.capacity();
                        p.e->ram = std::move(p.loaded);
                    } else {
                        c.keep_spare_locked(std::move(p.loaded));
                    }
                }
            }
            if (loaded_ok) c.make_room_locked();
            hb.take(&c);
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
                 fmt_rate(gov_.floor() / (double)(1ull << 30), 0).c_str());
            if (parts.size() == 2 && &p == &parts[0])
                slog(LogLevel::Warning, "%sit is the base of the %s-token entry the prompt resumes from", kCont,
                     fmt_n((long long)parts[1].e->tokens.size()).c_str());
            slog(LogLevel::Warning, "%s%s", kCont, ram_summary().c_str());
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
    // MADV_HUGEPAGE mapping under memory pressure compacts in the faulting thread (exports ~3 GB/s,
    // 75.7 s on the request path, once the RAM tier was full).
    const uint64_t want = prefault_bytes_;
    bool spare_fits = false;
    for (const HostBuffer &b : spares_) spare_fits |= b.capacity() >= want;
    if (want > 0 && !spare_fits && !gov_.room_now(want)) {
        Entry *lru = nullptr;
        for (const EntryPtr &e : entries_)
            if (e->ram.capacity() >= want && e->paged.empty() && !e->path.empty() && e->readers == 0 && !e->queued &&
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
    Handback hb;
    std::lock_guard<std::mutex> lock(mu_);
    int best = -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry &e = *entries_[i];
        if (e.kind != Kind::Turn || (int64_t)e.tokens.size() >= n || !e.in_ram() || !e.paged.empty() ||
            e.readers > 0 || e.queued ||
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
    hb.take(this);
    return b;
}

void PromptCache::give_back(HostBuffer b) {
    Handback hb;
    std::lock_guard<std::mutex> lock(mu_);
    keep_spare_locked(std::move(b));
    hb.take(this);
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
    if (e.ram_cost() == 0) return;
    ram_bytes_ -= e.ram_cost();
    if (!e.paged.empty())  // its parked pages wait in retired_ for their owner (take_retired; never HIP here)
        retired_.push_back(e.paged.parked_id), e.paged = PagedState();
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

void PromptCache::remove_locked(size_t i, bool to_spare) {
    STRIX_CHECK(i < entries_.size(), "PromptCache::remove_locked: index ", i, " of ", entries_.size());
    EntryPtr e = entries_[i];
    entries_.erase(entries_.begin() + (int64_t)i);
    e->dead = true;
    unlink_locked(*e);
    if (e->readers == 0) drop_ram_locked(*e, to_spare);  // else the last reader does
    if (e->base) {  // a delta lets go of its base
        --e->base->dependents;
        e->base.reset();
    }
    // A base takes its deltas with it: they're useless without it. (Deltas never have deltas, so removing one erases
    // only its own index: walking down from the end stays valid.)
    for (size_t k = entries_.size(); e->dependents > 0 && k-- > 0;)
        if (entries_[k]->base == e) {
            ++stats_.deltas_dropped;
            remove_locked(k, to_spare);
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
        // A paged entry never bases a delta: with_state would refuse it by name (its state's big ranges are
        // parked pages), and a resume from the delta must be able to import its base - so the base has to be
        // a state this cache can hand over as bytes. A turn that would land on a parked conversation saves
        // whole (and may become the next parked record itself).
        if (!e->paged.empty()) continue;
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
    return put_impl(std::move(tokens), kind, std::move(state), PagedState(), ram_only, base_n, saved_by);
}

bool PromptCache::put_paged(std::vector<int32_t> tokens, Kind kind, HostBuffer small, PagedState paged,
                            bool ram_only, int64_t base_n, int64_t saved_by) {
    STRIX_CHECK(!paged.empty(), "PromptCache::put_paged: ", tokens.size(),
                " tokens with no parked record - a whole state goes through put()");
    STRIX_CHECK(!small.empty(), "PromptCache::put_paged: a paged entry of ", tokens.size(),
                " tokens with no small-state buffer");
    // The two halves must be the whole state exactly: the writer gathers from them and cannot notice a byte that
    // is in neither. (paged_state checked the tiling; this checks what the CALLER brought.)
    size_t ranges = 0;
    for (const PagedChunk &c : paged.chunks) {
        STRIX_CHECK(c.bytes > 0 && c.host != nullptr, "PromptCache::put_paged: a range at state byte ", c.state_off,
                    " with ", c.bytes, " bytes at host ", static_cast<const void *>(c.host),
                    " (", tokens.size(), " tokens; expected bytes > 0 and an address)");
        ranges += c.bytes;
    }
    STRIX_CHECK(ranges + small.size() == paged.state_bytes, "PromptCache::put_paged: ", ranges, " parked + ",
                small.size(), " small bytes, the state is ", paged.state_bytes, " (", tokens.size(), " tokens)");
    return put_impl(std::move(tokens), kind, std::move(small), std::move(paged), ram_only, base_n, saved_by);
}

bool PromptCache::put_impl(std::vector<int32_t> tokens, Kind kind, HostBuffer state, PagedState paged, bool ram_only,
                           int64_t base_n, int64_t saved_by) {
    const uint64_t page_bytes = paged.bytes();  // RAM this entry holds besides its buffer, until it is given back
    STRIX_CHECK(saved_by >= 0, "PromptCache::put: saved_by request id ", saved_by, ", expected >= 0");
    STRIX_CHECK(!tokens.empty() && !(state.empty() && paged.empty()), "PromptCache::put: ", tokens.size(),
                " tokens, ", state.size(), " buffer + ", page_bytes, " parked state bytes");
    STRIX_CHECK(kind == Kind::System || kind == Kind::Turn || kind == Kind::Checkpoint, "PromptCache::put: kind ",
                (uint32_t)kind, ", expected 1..3");
    STRIX_CHECK(base_n >= 0 && base_n < (int64_t)tokens.size() && (base_n == 0 || kind != Kind::System),
                "PromptCache::put: base of ", base_n, " tokens for a kind ", (uint32_t)kind, " entry of ", tokens.size(),
                " (a delta is a Turn or a Checkpoint on a shorter base)");
    Handback hb;
    std::lock_guard<std::mutex> lock(mu_);
    EntryPtr base;
    if (base_n > 0) {
        const int b = find_locked(tokens.data(), base_n);
        if (b < 0 || entries_[(size_t)b]->base) {  // gone (or replaced) since delta_base: the delta is useless
            slog(LogLevel::Warning, "prompt cache: dropped a %s-token delta: its %s-token base is gone",
                 fmt_n((long long)tokens.size()).c_str(), fmt_n(base_n).c_str());
            keep_spare_locked(std::move(state));
            if (!paged.empty()) retired_.push_back(paged.parked_id), paged = PagedState();  // never stored
            hb.take(this);
            return false;
        }
        base = entries_[(size_t)b];
        base->last_used = ++clock_, base->touched_s = now_s();
    }
    if (const int i = find_locked(tokens.data(), (int64_t)tokens.size()); i >= 0) {
        Entry &e = *entries_[(size_t)i];
        e.last_used = ++clock_, e.touched_s = now_s(), e.no_room = false;  // already there
        if (!ram_only) e.ram_only = false;  // a request that may continue it wants it on disk too
        if (!e.in_ram() && !e.dead) {  // on disk only: this copy saves the next hit a read
            ram_bytes_ += state.capacity() + page_bytes;
            e.ram = std::move(state), e.paged = std::move(paged);
            make_room_locked();
        } else {  // it has a state already: this copy is surplus, and its pages go back
            keep_spare_locked(std::move(state));
            if (!paged.empty()) retired_.push_back(paged.parked_id), paged = PagedState();
        }
        hb.take(this);
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
        // and the base of a checkpoint delta kept above (Checkpoint deltas); it goes on a later trim.
        for (size_t k = (size_t)opt_.checkpoints_per_chain - 1; k < chain.size(); ++k) {
            if (chain[k].second == base.get() || chain[k].second->dependents > 0) continue;
            const int i = find_locked(chain[k].second);
            if (i < 0) continue;  // (not expected: only a base takes others with it, and bases are skipped)
            remove_locked((size_t)i);
            ++stats_.checkpoints_dropped;
        }
    }
    // The spares to keep ready follow whole states (a delta's few hundred MB would shrink them below the next rebase).
    if (!base) prefault_bytes_ = state.size() + page_bytes + opt_.prefault_headroom;
    auto e = std::make_shared<Entry>();
    e->tokens = std::move(tokens), e->kind = kind, e->last_used = ++clock_, e->touched_s = now_s();
    e->ram_only = ram_only, e->saved_by = saved_by;
    if (base) {
        e->base = base, ++base->dependents;
        ++stats_.delta_puts, stats_.delta_bytes += (int64_t)(state.size() + page_bytes);
    }
    ram_bytes_ += state.capacity() + page_bytes;
    e->ram = std::move(state), e->paged = std::move(paged);
    entries_.push_back(std::move(e));
    make_room_locked();
    hb.take(this);
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
        if (lru->in_ram() || lru->queued) unlink_locked(*lru), lru->no_room = true;  // RAM keeps it
        else remove_locked((size_t)find_locked(lru));
    }
    return true;
}

uint64_t PromptCache::write_bytes_locked(const Entry &e) const {
    uint64_t b = file_total(e.tokens.size(), e.state_size());
    if (e.base && e.base->path.empty() && !e.base->queued) b += file_total(e.base->tokens.size(), e.base->state_size());
    return b;
}

void PromptCache::enqueue_locked(const EntryPtr &e, Why why) {
    // A delta reaches the disk only with its base: the base's write goes first (the queue is FIFO).
    if (e->base && e->base->path.empty() && !e->base->queued) {
        STRIX_CHECK(e->base->in_ram(), "PromptCache: a delta's base of ", e->base->tokens.size(),
                    " tokens is neither on disk nor in RAM");
        enqueue_locked(e->base, why);
    }
    e->queued = true;
    const uint64_t bytes = e->state_size();
    pending_bytes_ += bytes;
    queue_.push_back({e, why, bytes});
    cv_.notify_one();
}

PromptCache::Spill PromptCache::spill_parked(double max_seconds) {
    STRIX_CHECK(max_seconds > 0 && max_seconds <= 3600, "PromptCache::spill_parked: max_seconds ", max_seconds,
                ", expected (0, 3600]");
    const double t0 = now_s();
    Spill out;
    Handback hb;
    std::unique_lock<std::mutex> lock(mu_);
    STRIX_CHECK(!shutting_down_ && !stop_, "PromptCache::spill_parked: the cache is shutting down");
    std::vector<EntryPtr> todo;
    for (const EntryPtr &e : entries_)
        if (!e->paged.empty()) todo.push_back(e);
    std::vector<EntryPtr> gone;  // one-shot entries: never written by design - they leave
    std::vector<std::pair<EntryPtr, uint64_t>> sent;  // the ones queued, and their parked page bytes
    for (const EntryPtr &e : todo) {
        if (e->ram_only) {
            gone.push_back(e);
            continue;
        }
        sent.emplace_back(e, e->paged.bytes());
        e->drop_ram = true;  // a refused write (no disk room) removes it rather than keep the pages
        if (!e->queued && e->path.empty()) enqueue_locked(e, Why::Renew);
    }
    for (const EntryPtr &e : gone)
        if (const int i = find_locked(e.get()); i >= 0) {
            STRIX_CHECK(e->readers == 0, "PromptCache::spill_parked: a one-shot parked entry of ", e->tokens.size(),
                        " tokens is being read - nothing should run during a K/V renewal");
            remove_locked((size_t)i);
            ++out.dropped;
        }
    cv_.notify_all();
    const auto holds_pages = [&] {
        for (const EntryPtr &e : entries_)
            if (!e->paged.empty()) return true;
        return false;
    };
    idle_cv_.wait_for(lock, std::chrono::duration<double>(max_seconds),
                      [&] { return (queue_.empty() && !writing_) || !holds_pages(); });
    // What is still paged now could not be written (refused for room, failed) or was still queued at the
    // deadline: it leaves (a queued job finds it dead and skips it). One under a reader - its write running at
    // the deadline - cannot: its pages are in use, and it is reported as left.
    for (size_t i = 0; i < entries_.size();) {
        const EntryPtr e = entries_[i];
        if (e->paged.empty()) {
            ++i;
            continue;
        }
        if (e->readers > 0) {
            ++out.left, ++i;
            continue;
        }
        remove_locked(i);
    }
    for (const auto &[e, b] : sent) {
        if (!e->dead && !e->path.empty()) ++out.written, out.bytes += b;
        if (e->dead) ++out.dropped;  // refused for room, failed, or removed just above
    }
    out.seconds = now_s() - t0;
    hb.take(this);
    return out;
}

void PromptCache::set_retired_hook(std::function<void()> hook) {
    std::lock_guard<std::mutex> lock(hook_mu_);
    retired_hook_ = std::move(hook);
}

void PromptCache::poke_retired() {
    bool any = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        any = !retired_.empty();
    }
    if (!any) return;
    std::lock_guard<std::mutex> lock(hook_mu_);
    if (!retired_hook_) return;
    try {
        retired_hook_();
    } catch (const std::exception &e) {
        slog(LogLevel::Error, "prompt cache: the retired-records hook threw: %s", e.what());
    }
}

bool PromptCache::make_room_for(uint64_t bytes, const RetiredRelease &release_retired) {
    STRIX_CHECK(bytes < (1ull << 40), "PromptCache::make_room_for: ", bytes, " bytes, expected < 1 TiB");
    if (bytes == 0) return true;
    // Parked records already retired go back first (their pages are RAM the reading below should see), then
    // after every eviction pass. Off-lock: the releaser does HIP work.
    const auto release_now = [&] {
        if (!release_retired) return;
        const std::vector<int64_t> ids = take_retired();
        if (!ids.empty()) release_retired(ids);
    };
    // Entries the RAM rule sends to disk give their RAM back only as the writer writes them (seconds), so the first
    // check after the rule often comes up short although the room is on its way - and a refused load re-prefills its
    // whole prompt. So while evicted entries are still being written, wait for them, up to room_wait_seconds; refuse
    // only when nothing more is leaving or the wait runs out.
    // The rule decides on the reading that found the shortfall: on separate reads the rule could see "enough" (and
    // evict nothing) where the check, a few MiB of MemAvailable movement later, saw "short" - a refusal with the RAM
    // tier full of entries free to leave.
    const double deadline = now_s() + opt_.room_wait_seconds;
    for (;;) {
        release_now();
        const uint64_t avail = gov_.available();  // the one reading the rule and the shortfall both see
        if (gov_.room(avail, bytes)) return true;
        {
            Handback hb;
            std::lock_guard<std::mutex> lock(mu_);
            make_room_locked(bytes, avail);
            hb.take(this);
        }  // the evicted buffers are unmapped here, before MemAvailable is read again
        release_now();  // and the evicted parked records' pages given back
        if (gov_.room_now(bytes)) return true;  // the evicted buffers were unmapped before this reading
        std::unique_lock<std::mutex> lock(mu_);
        const double left = deadline - now_s();
        if (leaving_bytes_locked() == 0 || left <= 0) {
            ++stats_.room_refused;
            return false;
        }
        ++stats_.room_waits;
        idle_cv_.wait_for(lock, std::chrono::duration<double>(std::min(left, 0.05)));  // a write done, or poll
    }
}

std::string PromptCache::ram_summary() const {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t n = 0, in_use = 0, queued = 0, free_n = 0;
    uint64_t bytes = 0, in_use_b = 0, queued_b = 0, free_b = 0;
    for (const EntryPtr &e : entries_) {
        if (!e->in_ram()) continue;
        const uint64_t b = e->ram_cost();
        ++n, bytes += b;
        if (e->queued) ++queued, queued_b += b;
        else if (e->readers > 0) ++in_use, in_use_b += b;
        else ++free_n, free_b += b;
    }
    const auto gb = [](uint64_t b) { return fmt_rate(b / 1e9, 1); };
    return "RAM tier " + fmt_n(n) + " entries, " + gb(bytes) + " GB: " + fmt_n(in_use) + " in use (" + gb(in_use_b) +
           " GB), " + fmt_n(queued) + " queued for a write (" + gb(queued_b) + " GB), " + fmt_n(free_n) +
           " free to leave (" + gb(free_b) + " GB); spares " + fmt_n((long long)spares_.size()) + " (" + gb(spare_bytes_) +
           " GB); leaving " + gb(leaving_bytes_locked()) + " GB";
}

uint64_t PromptCache::leaving_bytes_locked() const {
    uint64_t b = 0;
    for (const EntryPtr &e : entries_)
        if (e->drop_ram && e->in_ram()) b += e->ram_cost();
    return b;
}

void PromptCache::make_room_locked(uint64_t extra, uint64_t avail_in) {
    const uint64_t avail = avail_in != UINT64_MAX ? avail_in : gov_.available();
    const uint64_t target = gov_.floor() + extra;  // the RAM rule's other half: the governor owns the floor
    if (avail >= target && opt_.ram_tier) return;
    // No RAM tier: every entry that can leave does (need never reaches 0 below).
    uint64_t need = opt_.ram_tier ? target - avail : UINT64_MAX;
    // RAM already on its way out (evicted entries still being written) counts as freed, as it does for the entries
    // this pass sends: a second pass while those writes run (make_room_for's wait, the margin watch) must not evict
    // more for the same shortfall.
    if (opt_.ram_tier) need -= std::min(need, leaving_bytes_locked());
    // Spares go first: nothing is lost.
    while (need > 0 && !spares_.empty()) {
        need -= std::min(need, (uint64_t)spares_.back().capacity());
        spare_bytes_ -= spares_.back().capacity();
        trash_.push_back(std::move(spares_.back()));
        spares_.pop_back();
    }
    // Then resume points: the least lost (a point only helps a client that edits old history), and small.
    while (need > 0 && !points_.empty()) {
        size_t lru = 0;
        for (size_t i = 1; i < points_.size(); ++i)
            if (points_[i].last_used < points_[lru].last_used) lru = i;
        need -= std::min(need, (uint64_t)points_[lru].small->capacity());
        drop_point_locked();
        ++stats_.points_evicted;
    }
    while (need > 0) {
        // Candidates: every RAM entry not being imported and not already leaving. One queued for a write (idle, or
        // being written) counts too - it leaves RAM once written, its last read releasing it if one holds it - so the
        // least recently used goes first, never the newest just because the older ones sat in the write queue
        // (otherwise the entry just saved was dropped, "write queue full", and its conversation re-prefilled 132k
        // tokens).
        Entry *lru = nullptr;
        for (const EntryPtr &e : entries_)
            if (e->in_ram() && !e->drop_ram && (e->readers == 0 || e->queued) && (!lru || e->last_used < lru->last_used))
                lru = e.get();
        if (!lru) break;  // the rest is in use or on its way out; the next put tries again
        if (opt_.ram_tier) need -= std::min(need, lru->ram_cost());
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
            remove_locked((size_t)find_locked(lru), false);
            continue;
        }
        // The eviction rule: write it, unless the queue, the budget or the disk says no - then it's gone.
        const uint64_t total = write_bytes_locked(*lru);
        const uint64_t free = disk_free();
        const uint64_t fs_cap = disk_bytes_ + (free > opt_.disk_free_margin ? free - opt_.disk_free_margin : 0);
        const uint64_t queue_cap = opt_.max_pending_bytes ? opt_.max_pending_bytes
                                                            : std::max(kMaxPendingBytes, 4 * prefault_bytes_);
        const char *no = pending_bytes_ + lru->state_size() > queue_cap ? "queue"
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
             fmt_n((long long)lru->tokens.size()).c_str(), deltas.c_str(), fmt_rate(lru->state_size() / 1e9, 1).c_str(),
             by.c_str(), ago(idle).c_str());
        if (opt_.ram_tier)
            slog(level, "%sit had to leave RAM (MemAvailable %s GiB, below the %s GiB margin) and can't go to disk:", kCont,
                 fmt_rate(avail / (double)(1ull << 30), 1).c_str(), fmt_rate(gov_.floor() / (double)(1ull << 30), 0).c_str());
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
        remove_locked((size_t)find_locked(lru), false);
    }
}

void PromptCache::summary_locked(double now) {
    const uint64_t sig = clock_ + (uint64_t)stats_.writes + (uint64_t)stats_.ram_evicted;
    if (now - summary_at_ < kSummarySeconds || sig == summary_sig_) return;
    summary_at_ = now, summary_sig_ = sig;
    int64_t ram_n = 0, disk_n = 0;
    for (const EntryPtr &e : entries_) ram_n += e->in_ram(), disk_n += !e->path.empty();
    const double gib = (double)(1ull << 30);
    slog(LogLevel::Debug, "prompt cache: RAM %s entries, %s GB (+ %s GB spare buffers); MemAvailable %s GiB (margin %s)%s%s",
         fmt_n(ram_n).c_str(), fmt_rate(ram_bytes_ / 1e9, 1).c_str(), fmt_rate(spare_bytes_ / 1e9, 1).c_str(),
         fmt_rate(gov_.available() / gib, 1).c_str(), fmt_rate(gov_.floor() / gib, 0).c_str(),
         opt_.status_note.empty() ? "" : "; ", opt_.status_note.c_str());
    char budget[64] = "no write budget";
    if (opt_.write_gib_per_hour > 0)
        std::snprintf(budget, sizeof budget, "write budget %s of %s GiB left this hour",
                      fmt_rate(budget_now_locked() / gib, 1).c_str(), fmt_rate(opt_.write_gib_per_hour, 0).c_str());
    slog(LogLevel::Debug, "%sdisk %s entries, %s of %s GiB; %s", kCont, fmt_n(disk_n).c_str(), fmt_rate(disk_bytes_ / gib, 1).c_str(),
         fmt_rate(max_bytes_ / gib, 0).c_str(), budget);
    slog(LogLevel::Debug, "%ssince the start: %s written, %s dropped leaving RAM, %s write failures", kCont,
         fmt_n(stats_.writes).c_str(),
         fmt_n(stats_.rejected_queue + stats_.rejected_space + stats_.rejected_budget).c_str(),
         fmt_n(stats_.write_failures).c_str());
}

void PromptCache::write_idle_locked(double now) {
    if (shutting_down_) return;
    for (const EntryPtr &e : entries_) {
        if (!e->in_ram() || !e->path.empty() || e->queued || e->no_room || e->ram_only ||
            now - e->touched_s < opt_.idle_seconds)
            continue;
        // A paged entry under a read is being RESUMED: the hand maps its parked pages into a slot and takes
        // the parked mapping down, so a gather started now would read pages that are leaving the address
        // space. (The RAM half of an ordinary pinned entry is immutable - a paged one is not.) Next scan.
        if (!e->paged.empty() && e->readers > 0) continue;
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
    if (!gov_.room_now(prefault_bytes_)) return 0;  // never into memory pressure
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

bool PromptCache::drop_point_locked() {
    if (points_.empty()) return false;
    size_t lru = 0;
    for (size_t i = 1; i < points_.size(); ++i)
        if (points_[i].last_used < points_[lru].last_used) lru = i;
    Point &p = points_[lru];
    ram_bytes_ -= std::min<uint64_t>(ram_bytes_, p.small->capacity());
    points_[lru] = std::move(points_.back());
    points_.pop_back();
    return true;
}

bool PromptCache::put_point(std::vector<int32_t> tokens, HostBuffer small, int64_t saved_by) {
    STRIX_CHECK(!tokens.empty() && !small.empty(), "PromptCache::put_point: ", tokens.size(), " tokens, ", small.size(),
                " small-state bytes (expected both > 0)");
    STRIX_CHECK(saved_by >= 0, "PromptCache::put_point: saved_by request id ", saved_by, ", expected >= 0");
    // No RAM tier (low-memory adapt): nothing stays in RAM, so a point would only be dropped again.
    if (opt_.max_points <= 0 || !opt_.ram_tier) return false;
    Handback hb;
    std::lock_guard<std::mutex> lock(mu_);
    for (Point &p : points_)
        if (p.tokens.size() == tokens.size() && std::memcmp(p.tokens.data(), tokens.data(), tokens.size() * 4) == 0) {
            p.last_used = ++clock_;  // held already: this copy is surplus
            trash_.push_back(std::move(small));
            hb.take(this);
            return true;
        }
    ram_bytes_ += small.capacity();
    Point p;
    p.tokens = std::move(tokens), p.last_used = ++clock_, p.saved_by = saved_by;
    p.small = std::make_shared<const HostBuffer>(std::move(small));
    points_.push_back(std::move(p));
    ++stats_.point_puts;
    while ((int64_t)points_.size() > opt_.max_points && drop_point_locked()) ++stats_.points_dropped;
    make_room_locked();
    hb.take(this);
    return true;
}

int64_t PromptCache::best_point(const std::vector<int32_t> &prompt, int64_t max_n) const {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t best = 0;
    for (const Point &p : points_) {
        const int64_t n = (int64_t)p.tokens.size();
        if (n <= best || n > max_n || n > (int64_t)prompt.size()) continue;
        if (std::memcmp(p.tokens.data(), prompt.data(), (size_t)n * 4) == 0) best = n;
    }
    return best;
}

bool PromptCache::with_point(const std::vector<int32_t> &prompt, int64_t n,
                             const std::function<void(const HostBuffer &)> &use) {
    STRIX_CHECK(n >= 1 && n <= (int64_t)prompt.size(), "PromptCache::with_point: n = ", n, ", the prompt has ",
                prompt.size(), " tokens");
    std::shared_ptr<const HostBuffer> small;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (Point &p : points_)
            if ((int64_t)p.tokens.size() == n && std::memcmp(p.tokens.data(), prompt.data(), (size_t)n * 4) == 0) {
                p.last_used = ++clock_;
                small = p.small;
                ++stats_.point_hits;
                break;
            }
    }
    if (!small) return false;
    use(*small);
    return true;
}

std::vector<int32_t> PromptCache::donor(const std::vector<int32_t> &prompt, int64_t n) const {
    STRIX_CHECK(n >= 1 && n <= (int64_t)prompt.size(), "PromptCache::donor: n = ", n, ", the prompt has ",
                prompt.size(), " tokens");
    std::lock_guard<std::mutex> lock(mu_);
    const Entry *best = nullptr;
    int best_rank = 3;
    for (const EntryPtr &e : entries_) {
        if (e->dead || e->releasing || (int64_t)e->tokens.size() < n) continue;
        if (std::memcmp(e->tokens.data(), prompt.data(), (size_t)n * 4) != 0) continue;
        const int rank = !e->paged.empty() ? 0 : e->in_ram() ? 1 : !e->path.empty() ? 2 : 3;
        if (rank == 3) continue;  // neither in RAM nor on disk (being made): nothing to load
        if (rank < best_rank || (rank == best_rank && e->tokens.size() < best->tokens.size())) best = e.get(), best_rank = rank;
    }
    return best ? best->tokens : std::vector<int32_t>();
}

PromptCacheStats PromptCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PromptCacheStats s = stats_;
    s.entries = (int64_t)entries_.size();
    for (const EntryPtr &e : entries_) s.ram_entries += e->in_ram(), s.disk_entries += !e->path.empty();
    s.ram_bytes = (int64_t)ram_bytes_, s.spare_bytes = (int64_t)spare_bytes_;
    s.bytes = (int64_t)disk_bytes_, s.max_bytes = (int64_t)max_bytes_;
    s.pending = (int64_t)queue_.size();
    s.points = (int64_t)points_.size();
    for (const Point &p : points_) s.point_bytes += (int64_t)p.small->capacity();
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
                    Handback hb;
                    hb.take(this);
                    lock.unlock();
                    hb.trash.clear();
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
            Handback hb;
            std::lock_guard<std::mutex> lock(mu_);
            ++stats_.write_failures;
            // An eviction's entry can't stay in RAM (that's why it left) and has no file: it's gone.
            if (job.e->drop_ram && !job.e->dead)
                if (const int i = find_locked(job.e.get()); i >= 0) remove_locked((size_t)i);
            hb.take(this);
        }
        {
            Handback hb;
            std::lock_guard<std::mutex> lock(mu_);
            writing_ = false;
            job.e.reset();
            hb.take(this);
        }
        idle_cv_.notify_all();
        poke_retired();  // an eviction write's record retired once its file was complete
    }
}

void PromptCache::watch_margin(std::unique_lock<std::mutex> &lock) {
    STRIX_CHECK(lock.owns_lock(), "PromptCache::watch_margin: called without the cache's lock");
    // Off: no RAM tier keeps nothing to evict; shutting down, the shutdown rule writes RAM entries itself.
    if (!opt_.ram_tier || shutting_down_ || stop_) return;
    const uint64_t avail = gov_.available();
    if (gov_.room(avail, 0)) return;
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
             fmt_rate(avail / gib, 1).c_str(), fmt_rate(gov_.floor() / gib, 0).c_str(), fmt_n((long long)n).c_str(),
             fmt_rate(freed / 1e9, 1).c_str());
    Handback hb;
    hb.take(this);
    lock.unlock();
    hb.trash.clear();  // unmapped outside the lock
    poke_retired();    // parked records this pass retired: an idle owner drains them now, not at its next request
    lock.lock();
}

bool PromptCache::with_paged(const std::vector<int32_t> &prompt, int64_t n,
                             const std::function<void(const HostBuffer &, const PagedState &)> &use, bool consumed) {
    STRIX_CHECK(n >= 1 && static_cast<size_t>(n) <= prompt.size(), "PromptCache::with_paged: ", n,
                " tokens of a ", prompt.size(), "-token prompt");
    STRIX_CHECK(static_cast<bool>(use), "PromptCache::with_paged: no consumer");
    EntryPtr e;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (const int i = find_locked(prompt.data(), n); i >= 0) {
            Entry &f = *entries_[(size_t)i];
            // Pinned for the call: an eviction or a finishing write must not take the pages back from under a
            // resume that is mapping them (the same reason a RAM read holds its readers count - strixite #14).
            if (!f.paged.empty() && f.readers == 0 && !f.queued) {
                f.last_used = ++clock_, f.touched_s = now_s(), f.no_room = false;
                ++f.readers, e = entries_[(size_t)i];
            }
        }
    }
    if (!e) return false;  // no entry, on disk (its file has the whole state), or a write has it
    try {
        use(e->ram, e->paged);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mu_);
        release_locked(*e);
        throw;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (consumed) {
        // The record is spent: `use` mapped its pages into a slot as that conversation's live K/V. The entry
        // can hold neither half - the pages are not the cache's RAM any more and the small bytes alone are not
        // a state - so what survives of it is its file, or nothing. No retired id: the record was spent by
        // the resume, not left over for its owner to release (a second with_paged on the spent id would have
        // failed the next request in resume_paged - this is what makes a parked entry good for one resume).
        const uint64_t held = e->ram_cost();  // both halves leave the RAM tally right now (read before clearing)
        e->paged = PagedState();
        ram_bytes_ -= held;
        if (e->ram.capacity() > 0) trash_.push_back(std::move(e->ram));  // unmapped off-lock
        if (e->path.empty() && !e->dead && !e->queued) {
            // No file yet: the state this entry described now lives only in the slot's pages - it goes (a
            // miss later, never a half state; the next turn's save replaces it anyway).
            if (const int i = find_locked(e.get()); i >= 0) remove_locked(static_cast<size_t>(i), false);
        }
    }
    release_locked(*e);
    return true;
}

std::vector<int64_t> PromptCache::take_retired() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<int64_t> out;
    out.swap(retired_);
    return out;
}

uint64_t PromptCache::write_paged_state(int fd, const Entry &e, uint64_t state_at, const std::string &path) const {
    const PagedState &pg = e.paged;
    STRIX_CHECK(!pg.chunks.empty() && pg.state_bytes >= 1, path, ": a paged entry with no ranges");
    StrixHasher hash(pg.state_bytes);
    constexpr size_t kPiece = 64ull << 10;  // a few syscalls per GB, a buffer that is not a second state
    HostBuffer scratch(kPiece);
    // The pack in order: whatever the next chunk does not cover comes from the small buffer. Both cursors only
    // move forward, and every byte written is a byte of one of the two - so the sums at the end are the check
    // that the two halves really tiled the state and nothing was read from a hole.
    // A range may straddle a piece: inside one (state_off <= at < its end) the read continues at its offset.
    for (size_t k = 0; k < pg.chunks.size(); ++k)
        STRIX_CHECK(pg.chunks[k].bytes > 0 && pg.chunks[k].host != nullptr, path, ": paged range ", k, " of ",
                    pg.chunks.size(), " at state byte ", pg.chunks[k].state_off, " has ", pg.chunks[k].bytes,
                    " bytes", pg.chunks[k].host ? "" : " and no host address", " (expected bytes > 0 and an address)");
    size_t at = 0, small_at = 0, ci = 0;
    auto fill = [&](uint8_t *dst, size_t want) {
        for (size_t got = 0; got < want;) {
            const bool from_chunk = ci < pg.chunks.size() && at >= pg.chunks[ci].state_off;
            const size_t upto = from_chunk ? pg.chunks[ci].state_off + pg.chunks[ci].bytes
                                           : (ci < pg.chunks.size() ? pg.chunks[ci].state_off : pg.state_bytes);
            STRIX_CHECK(at < upto, path, ": nothing covers state byte ", at, " of the paged entry");
            const size_t take = std::min({upto - at, want - got, pg.state_bytes - at});
            STRIX_CHECK(from_chunk || small_at + take <= e.ram.size(), path, ": state byte ", at, " wants small bytes [",
                        small_at, ", +", take, "), the entry has ", e.ram.size());
            const uint8_t *src = from_chunk ? pg.chunks[ci].host + (at - pg.chunks[ci].state_off) : e.ram.data() + small_at;
            std::memcpy(dst + got, src, take);
            got += take, at += take;
            if (from_chunk) {
                if (at == pg.chunks[ci].state_off + pg.chunks[ci].bytes) ++ci;  // that range is spent
            } else small_at += take;
        }
    };
    for (uint64_t off = 0; off < pg.state_bytes; off += scratch.capacity()) {
        const size_t real = static_cast<size_t>(std::min<uint64_t>(scratch.capacity(), pg.state_bytes - off));
        std::memset(scratch.data(), 0, scratch.capacity());  // past `real` is padding the truncate cuts away
        fill(scratch.data(), real);
        hash.update(scratch.data(), real);  // the padding is not part of the state, and is not hashed
        write_all(fd, scratch.data(), align4k(real), state_at + off, path);
    }
    STRIX_CHECK(at == pg.state_bytes && ci == pg.chunks.size(), path, ": the paged state came to ", at,
                " bytes over ", ci, " of ", pg.chunks.size(), " ranges");
    STRIX_CHECK(small_at == e.ram.size(), path, ": the paged state used ", small_at, " small bytes, the entry has ",
                e.ram.size());
    return hash.final();
}

void PromptCache::write_entry(Job &job) {
    Entry &e = *job.e;
    const double t0 = now_s();
    uint64_t n = 0, total = 0, state_at = 0, state_bytes = 0, base_n = 0;
    bool releasing = false;  // set under the lock below: whether this write's source is being thrown away
    {
        Handback hb;
        std::lock_guard<std::mutex> lock(mu_);
        e.queued = false;
        if (e.dead || !e.path.empty() || !e.in_ram()) return;  // replaced / written / gone meanwhile
        if (e.base && e.base->path.empty()) {  // its base's write failed or was refused: a delta file alone is useless
            ++stats_.rejected_space;
            e.no_room = true;
            if (e.drop_ram) remove_locked((size_t)find_locked(&e));
            hb.take(this);
            return;
        }
        base_n = e.base ? e.base->tokens.size() : 0;
        n = e.tokens.size(), state_bytes = e.state_size();
        state_at = align4k(kHeaderBytes + n * 4), total = state_at + state_bytes;
        if (!disk_room_locked(total, e.last_used)) {
            ++stats_.rejected_space;
            e.no_room = true;
            if (e.drop_ram) remove_locked((size_t)find_locked(&e));
            hb.take(this);
            return;
        }
        // An eviction: its RAM goes back as its bytes are written (not all at the end) - unless a reader holds it, and
        // no reader takes it meanwhile (with_state waits for the file while `releasing`).
        releasing = e.drop_ram && e.readers == 0;
        e.releasing = releasing;
        ++e.readers;  // the RAM copy stays put while the file is written
        e.queued = true;
    }
    const uint64_t parked_bytes = e.paged.bytes();  // for the log line: this write's whole point
    const uint64_t token_hash = strix_hash64(e.tokens.data(), n * 4);
    char name[64];
    std::snprintf(name, sizeof name, "%016llx-%llu.pcache", (unsigned long long)token_hash, (unsigned long long)n);
    const std::string path = dir_ + "/" + name, partial = path + ".partial";
    bool ok = false;
    uint64_t released = 0;
    try {
        // O_DIRECT: the state never becomes dirty page cache. Buffered, a multi-GB write under memory pressure (free
        // memory ~0 with the n-gram table cached) made the request path's allocations wait in reclaim for writeback -
        // exports 7-12 s while switching conversations with idle writes on vs 1.5-2.4 s with only eviction writes -
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
        // The state's hash: the whole-buffer path has it before it writes. A paged entry's bytes are gathered as
        // they go out, so its header - the one thing that must carry that hash - is written AFTER its state.
        put(40, token_hash);
        const bool paged = !e.paged.empty();
        if (!paged) put(48, strix_hash64(e.ram.data(), state_bytes));
        std::memcpy(h + kFpAt, fingerprint_.data(), fingerprint_.size());
        put(kBaseAt, base_n);
        put(kHeaderHashAt, strix_hash64(h, kHeaderHashAt));
        std::memcpy(h + kHeaderBytes, e.tokens.data(), n * 4);
        if (tail && !paged) std::memcpy(h + state_at, e.ram.data() + body, tail);
        const int fd = ::open(partial.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0644);
        STRIX_CHECK(fd >= 0, partial, ": create failed: ", std::strerror(errno));
        try {
            int flags = 0;  // compression off while empty (KV doesn't compress; a compressed extent costs reads)
            if (::ioctl(fd, FS_IOC_GETFLAGS, &flags) == 0) {
                flags |= FS_NOCOMP_FL;
                (void)::ioctl(fd, FS_IOC_SETFLAGS, &flags);
            }
            if (!paged) {
                write_all(fd, h, state_at, 0, partial);
                if (releasing) released = write_all_releasing(fd, e.ram, body, state_at, partial);
                else write_all(fd, e.ram.data(), body, state_at, partial);
                if (tail) write_all(fd, h + state_at, 4096, state_at + body, partial);
            } else {
                put(48, write_paged_state(fd, e, state_at, partial));
                put(kHeaderHashAt, strix_hash64(h, kHeaderHashAt));
                write_all(fd, h, state_at, 0, partial);  // 4 KiB at 0: aligned, it lands in front of the state
            }
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
        e.queued = false, e.releasing = false;
        release_locked(e);
        throw;
    }
    Handback hb;
    std::lock_guard<std::mutex> lock(mu_);
    e.queued = false, e.releasing = false;
    if (e.dead) {  // replaced while it was written: the file goes too
        std::error_code ec;
        fs::remove(path, ec);
    } else if (ok) {
        if (!e.paged.empty()) {
            // Its file has the state now, so holding the copy is pure cost: the pages go back to their owner and
            // the small bytes with them (half a state is not a state - with_state would refuse it anyway). If a
            // read took it meanwhile the drop defers to its last release, as a whole entry's eviction does
            // (strixite #14) - the pages must not go while someone is mapping them into a slot.
            e.drop_ram = true;
            if (e.readers == 1) drop_ram_locked(e, true);
        }
        e.path = path, e.file_bytes = total;
        disk_bytes_ += total;
        const std::string by = e.saved_by > 0 ? " (saved by Request " + std::to_string(e.saved_by) + ")" : "";
        const std::string freeing =
            releasing && released > 0
                ? ", " + fmt_rate(released / 1e9, 1) + " GB gave back RAM as it wrote"
                : parked_bytes > 0
                      ? ", " + fmt_rate(parked_bytes / 1e9, 1) + " GB of parked pages given back (that parked "
                        "generation's addresses are spent)"
                      : "";
        slog(LogLevel::Debug, "prompt cache: wrote a %s entry of %s tokens%s to disk, %s MB in %s s%s", kind_name(e.kind),
             fmt_n((long long)n).c_str(), by.c_str(), fmt_rate(total / 1e6, 0).c_str(), fmt_rate(now_s() - t0, 2).c_str(),
             freeing.c_str());
        const std::string delta = base_n > 0 ? "; a delta on the " + fmt_n((long long)base_n) + "-token entry" : "";
        slog(LogLevel::Debug, "%swhy: %s%s", kCont,
             job.why == Why::Idle    ? ("unused for " + ago(now_s() - e.touched_s)).c_str()
             : job.why == Why::Evict ? "leaving RAM (MemAvailable below the margin)"
             : job.why == Why::Renew ? "the K/V budget renews (the parked pages go with the old session)"
                                     : "server shutdown",
             delta.c_str());
        if (e.drop_ram && e.readers > 1)
            slog(LogLevel::Info, "prompt cache: the %s-token entry's eviction write finished while a read still holds "
                 "its RAM - the RAM leaves when the read does", fmt_n((long long)n).c_str());  // strixite #14
    }
    release_locked(e);  // the last reader's release also finishes a deferred drop_ram (see release_locked)
    ++stats_.writes, stats_.bytes_written += (int64_t)total, stats_.write_seconds += now_s() - t0;
    (job.why == Why::Evict    ? stats_.writes_evict
     : job.why == Why::Idle   ? stats_.writes_idle
     : job.why == Why::Renew  ? stats_.writes_renew
                              : stats_.writes_shutdown)++;
    hb.take(this);
}

}  // namespace strix
