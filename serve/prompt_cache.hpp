#pragma once

// The prompt cache: saved session states
// keyed by the token ids they cover, so a prompt that starts with a saved prefix resumes there instead of prefilling
// it - across conversations and server restarts. Two tiers: host RAM first, the disk lazily.
//
// RAM tier: put() keeps the state's HostBuffer in RAM (no disk write); a lookup that hits RAM hands that buffer to the
// import directly (no copy, no disk). Its size follows free memory: after every put, least recently used entries
// leave RAM while MemAvailable is below ram_margin. An entry leaving RAM is written to disk if it isn't there yet
// and the write rules allow it, else it is gone (a miss later, a re-prefill - never a wrong state). A Turn replaced by
// the next turn of its conversation is dropped while still in RAM, so a tool loop's turns never reach the disk.
//
// Write rules: an entry is written (1) when RAM evicts it, (2) when it has sat idle in RAM for
// idle_seconds (in the background; it stays in RAM), (3) at clean shutdown (the destructor, newest first, within
// shutdown_seconds). (1) and (2) spend a write budget (write_gib_per_hour, a token bucket); every write needs the
// disk to keep disk_free_margin free after it. A rejected eviction write drops the entry and counts why.
// The writer runs at idle I/O priority, so the n-gram table's reads on the same SSD go first.
//
// An entry on disk is one file: a 4096-byte header (magic, version, kind, token count, state size, the backend's state
// fingerprint, hashes of the tokens and the state, a hash of the header), the token ids, then the state bytes
// (LmBackend::export_snapshot) at a 4096-aligned offset. Written as ".partial", then renamed; btrfs compression off
// for the file. The disk index is rebuilt from the directory at startup (RAM starts empty); an entry with another
// fingerprint, a bad header or a bad checksum is deleted and counted as invalid. A disk hit is promoted to RAM (the
// buffer it was read into becomes the entry's RAM copy). Disk eviction: least recently used first, until the
// entries fit min(max_bytes, what's on disk + free space - disk_free_margin) - and a write only evicts entries used
// less recently than its own (else it is refused for space), so RAM and disk never churn each other's entries.
//
// Kinds: System = the prefix up to the first message after the system prompt (shared by every conversation with
// that system prompt); Turn = a conversation up to its last generation prompt; Checkpoint = a conversation up to
// the start of its last user message (saved for prompts ending in the user's message, not tool results). A new
// Turn or Checkpoint replaces the Turn entries it extends (RAM and disk), never a Checkpoint: an agent's tool loop
// keeps one entry while the conversation's user messages each keep theirs, so a client that later rewrites earlier
// history (e.g. strips a reminder it appended to the previous user message) still resumes from the checkpoint.
//
// RAM-only entries (put(..., ram_only = true)): the engine marks those of one-shot requests - one message, no tools,
// nothing a later turn extends (a batch client's classification prompts). They serve repeats from RAM but never reach
// the disk - no eviction, idle or shutdown write; leaving RAM drops them (counted as ram_only_dropped). On 2026-09-30
// such prompts were 251 of the disk's 279 entries (50 GB) and most of its ~230 GB/day of writes. A put of the same
// tokens without the flag clears it.
//
// Checkpoints per conversation: a new Checkpoint keeps at most checkpoints_per_chain - 1 of the older Checkpoints it
// extends (the newest), so a long session doesn't fill RAM and disk with its early user messages.
//
// Delta entries: a Turn or a Checkpoint may be saved as a delta on
// a full entry it extends - its base - holding only what changed since the base's position (the K / V rows past it and
// the per-step state; LmBackend::export_snapshot with from > 0). A deep conversation then saves ~0.2 GB a turn instead
// of its whole state (11.5 GB at 410k). delta_base() picks the base: the longest full entry the Turn extends, if the
// delta stays within delta_max_tokens and a quarter of the base (else the Turn is saved whole and becomes the next
// base - a rebase). Deltas never chain: a resume is base + one delta (with_state hands both, base first). Rules:
// - a delta lives only while its base does: removing a base removes its deltas (RAM and disk);
// - a new Turn replaces the Turns it extends except its own base, and a base's buffer is never handed out for reuse;
// - a delta reaches the disk only with its base: writing one writes the base first if it isn't there (the budget
//   counts both); at startup a delta file whose base file is missing is dropped (counted invalid);
// - using or saving a delta refreshes its base's place in the LRU, so the two leave RAM together;
// - a Checkpoint's base is a Checkpoint or a System entry, never a Turn (the next turn would replace it, and the
//   checkpoint would go with it); the checkpoints_per_chain trim skips a base while a delta needs it (2026-10-08:
//   terminal-bench sends every terminal output as a user message - a whole ~1.2 GB checkpoint a request - and the
//   margin test's 264k conversations a whole 7.4 GB one).
// On disk a delta is a version-2 file whose header names its base by length (the base's tokens are the delta's own
// first base_n tokens).
//
// Buffers: a state leaving RAM (replaced, evicted, written after eviction) becomes a spare - up to kMaxSpares,
// the largest kept - which take_buffer() hands to the next export or disk load: at 100-200k context a fresh buffer
// costs ~0.2 s of page population, a reused one nothing (switch test, dac725e: 3.7 GB export 205-230 ms fresh vs
// ~48 ms reused). Since RAM entries keep their buffers, the writer thread prefaults spares while idle: up to
// kMaxSpares of the last saved state's size + prefault_headroom, only while MemAvailable stays above ram_margin plus
// that size. Spares count as RAM and go first when memory is short.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/host_buffer.hpp"

namespace strix {

// /proc/meminfo's MemAvailable in bytes; throws if it can't be read (the RAM tier's probe and the startup memory plan).
uint64_t proc_mem_available();

struct PromptCacheStats {
    int64_t entries = 0;                           // RAM or disk or both
    int64_t ram_entries = 0, ram_bytes = 0;        // RAM: entries' buffers (capacity)
    int64_t spare_bytes = 0;                       // RAM: spare buffers
    int64_t disk_entries = 0, bytes = 0, max_bytes = 0;  // disk
    int64_t ram_hits = 0, loads = 0, load_failures = 0;  // hits from RAM / loads from disk
    int64_t writes = 0, write_failures = 0, bytes_written = 0;
    int64_t writes_evict = 0, writes_idle = 0, writes_shutdown = 0;  // writes by rule
    int64_t ram_evicted = 0;  // entries that left RAM for lack of memory (written, already on disk, or dropped)
    int64_t ram_only_dropped = 0;  // RAM-only entries that left RAM (never written, by design)
    int64_t rejected_space = 0, rejected_budget = 0, rejected_queue = 0;  // eviction writes refused, by reason
    int64_t room_refused = 0;  // make_room_for found no room: a save skipped or a disk load refused
    int64_t evicted = 0, replaced = 0, invalid = 0;  // disk LRU / replaced turns / damaged or foreign
    int64_t checkpoints_dropped = 0;                 // older Checkpoints over checkpoints_per_chain
    int64_t prefaulted = 0;                          // spares the writer populated ahead of an export
    int64_t buffers_reused = 0;  // take_buffer handed out an on-disk RAM entry's buffer (memory near the margin)
    int64_t turn_buffers_reused = 0;  // take_replaced_buffer handed out the replaced turn's buffer
    int64_t delta_puts = 0, delta_bytes = 0;  // Turns saved as deltas, and their state bytes
    int64_t deltas_dropped = 0;               // deltas removed with their base (or whose base file was missing)
    double load_seconds = 0, write_seconds = 0;
    int64_t pending = 0;  // queued writes
};

class PromptCache {
public:
    enum class Kind : uint32_t { System = 1, Turn = 2, Checkpoint = 3 };

    struct Options {
        // Keep MemAvailable above this (RAM entries leave first; make_room_for holds it at every allocation). 24 GiB:
        // at 16 the switch test hit direct compaction at 23-27 GB available (MemAvailable is mostly page cache, and
        // large allocations compact it) - exports 4-5 s, a RAM resume 9.6 s; at 24, none. A single-purpose machine
        // measured fine at 4; the right default for one that also runs a desktop is still being measured.
        uint64_t ram_margin = 24ull << 30;
        double idle_seconds = 600;          // an entry idle in RAM this long is written (and kept in RAM)
        // Budget for eviction and idle writes; 0 = no limit. 32 GiB/h: the served value since 2026-10-03 (reasons in
        // deploy/strix-server.conf); the old 64 predates delta entries.
        double write_gib_per_hour = 32;
        uint64_t disk_free_margin = 32ull << 30;  // a write never leaves less free space on the cache's filesystem
        double shutdown_seconds = 40;       // the destructor's flush stops after this long
        int checkpoints_per_chain = 2;      // Checkpoints kept per conversation (>= 1)
        // Delta entries: a Turn is a delta on its base while it adds at most this many tokens (and at most a quarter of
        // the base's); 0 = every entry whole. 65536 (was 32768, 2026-10-08): below a 262k base the quarter rule
        // decides - a delta of <= ~1.4 GB (27 KiB a token + ~0.12 GB) instead of a whole rebase every 32k tokens.
        int64_t delta_max_tokens = 65536;
        bool prefault = true;               // the writer keeps populated spares ready for the next export
        int max_spares = 2;                 // spare buffers kept, 0..kMaxSpares (0: low-memory adapt, serve/memory_plan)
        // false (low-memory adapt, serve/memory_plan.hpp): no RAM tier - every entry leaves RAM right after its put or
        // load (written to disk under the usual rules, else dropped), so only the export or load in flight holds RAM.
        bool ram_tier = true;
        std::string status_note;            // appended to the periodic status line ("" = none): what adapt turned down
        uint64_t prefault_headroom = 256ull << 20;  // on top of the last saved state (a conversation grows)
        uint64_t max_pending_bytes = 0;  // the eviction write queue's cap; 0 = max(kMaxPendingBytes, 4 x last state)
        // Memory and disk probes: /proc/meminfo MemAvailable and statvfs(dir) by default; tests fake them.
        std::function<uint64_t()> mem_available;
        std::function<uint64_t()> disk_free;
    };

    // dir is created if absent. max_bytes: the cap on the directory's entries (>= 1 MiB).
    PromptCache(std::string dir, uint64_t max_bytes, std::string fingerprint);
    PromptCache(std::string dir, uint64_t max_bytes, std::string fingerprint, Options options);
    ~PromptCache();  // writes RAM-only entries to disk (the shutdown rule), then stops the writer
    PromptCache(const PromptCache &) = delete;
    PromptCache &operator=(const PromptCache &) = delete;

    // The entry covering the longest prefix of prompt with at most max_n tokens; 0 if none.
    int64_t best(const std::vector<int32_t> &prompt, int64_t max_n) const;
    // The entry sharing the longest prefix with prompt, whatever its length (a longer entry, or one that differs part
    // way): what a resume that fell short could have had (serve/resume_loss.hpp). O(entries x shared prefix) under the
    // lock - for requests that prefill a lot anyway, not every turn.
    struct Nearest {
        int64_t common = 0;  // tokens the entry and the prompt share from token 0; 0 = none (or no entries)
        int64_t tokens = 0;  // the entry's length
        Kind kind = Kind::Turn;
        bool in_ram = false;
    };
    Nearest nearest(const std::vector<int32_t> &prompt) const;
    // True if an entry covers exactly these tokens (in RAM or on disk).
    bool contains(const int32_t *tokens, int64_t n) const;
    // True if that entry has a RAM copy / a disk file.
    bool in_ram(const int32_t *tokens, int64_t n) const;
    bool on_disk(const int32_t *tokens, int64_t n) const;
    // Calls use(state, from, to) with the state of the entry covering prompt[0, n) (as best() found it), for positions
    // [from, to): for a delta, first its base's (0, base length), then the delta's (base length, n); else once (0, n).
    // Each state is
    // its RAM copy or the file read into a spare buffer, which then becomes the entry's RAM copy; valid only during
    // the call. *from_ram: every part came from RAM. False if the entry is gone or damaged (a damaged one is deleted
    // and logged) - before any call of use. An exception from use propagates.
    bool with_state(const std::vector<int32_t> &prompt, int64_t n,
                    const std::function<void(const HostBuffer &, int64_t from, int64_t to)> &use,
                    bool *from_ram = nullptr);
    // The same for callers that only ever see whole entries (no deltas): use(state). Throws on a delta entry.
    bool with_state(const std::vector<int32_t> &prompt, int64_t n, const std::function<void(const HostBuffer &)> &use,
                    bool *from_ram = nullptr);
    // The base a Turn of tokens[0, n) should be saved as a delta on: its length (export the delta from there and put
    // with that base_n), or 0 = save it whole. See "Delta entries" above.
    int64_t delta_base(const int32_t *tokens, int64_t n, Kind kind) const;
    // Adds an entry in RAM (a duplicate only refreshes the existing one; its buffer becomes a spare), replacing the
    // turns it extends, then makes room in RAM. ram_only: never written to disk (see "RAM-only entries" above).
    // base_n > 0: state is a delta on the full entry of tokens[0, base_n) (delta_base()); if that base is gone since,
    // the state is dropped (logged) and put returns false. saved_by: the request that saved it, for the log lines
    // about the entry later (0: none).
    bool put(std::vector<int32_t> tokens, Kind kind, HostBuffer state, bool ram_only = false, int64_t base_n = 0,
             int64_t saved_by = 0);
    // A spare buffer (the largest; empty if none) for an export or a load, resized by whoever fills it - and back.
    // Near the margin with no spare of the last saved state's size, the least recently used RAM entry already on
    // disk gives up its buffer instead (no fresh mapping under memory pressure).
    HostBuffer take_buffer();
    // The RAM buffer of the Turn entry a put of tokens[0, n) will replace (the longest Turn it extends - the same
    // conversation one step back), taken now: that entry leaves the index (RAM and disk) as the put would do anyway.
    // The export then grows it by the turn's few MB instead of populating a fresh multi-GB buffer - near the RAM
    // margin spares are the first thing dropped, so there usually is none (2026-09-30: 2-6 GB exports 278-546 ms
    // median). Empty if there is no such entry in RAM, or it is being read or written: use take_buffer(). If the
    // export then fails, the old turn is lost - a miss later, never a wrong state. Never a base: neither one with
    // deltas nor the entry of tokens[0, keep_n) - the base the coming put will name (delta_base()), whose old delta
    // this call may just have taken.
    HostBuffer take_replaced_buffer(const int32_t *tokens, int64_t n, int64_t keep_n = 0);
    // Before a buffer grows by `bytes` (an export into a taken buffer, a disk load): spares and least recently used
    // entries leave RAM until MemAvailable >= ram_margin + bytes, measured after their memory is unmapped. False if it
    // still isn't (an eviction write still queued holds its RAM until written): the caller then doesn't grow the
    // buffer - the save is skipped, the load refused - so the margin holds at every allocation, not only after a put
    // (2026-10-08: a whole 7.4 GB export a request, checked only afterwards, took the machine to 0.5 GB and hung it).
    bool make_room_for(uint64_t bytes);
    uint64_t mem_available_now() const { return mem_available(); }  // MemAvailable (or the test's probe), bytes
    uint64_t disk_free_now() const { return disk_free(); }          // free bytes on the directory's filesystem
    void give_back(HostBuffer b);
    // Queues the idle rule's writes now (the writer also does this on its own every few seconds).
    void write_idle();
    void flush();  // waits until no write is queued or running (tests)
    PromptCacheStats stats() const;
    const std::string &dir() const { return dir_; }
    const Options &options() const { return opt_; }

    // Queued eviction writes beyond max(this, 4 x the last saved state) are refused (Options::max_pending_bytes): 8 GiB
    // alone held two ~3.7 GB states, and the margin switch tests dropped an entry a run on "write queue full".
    static constexpr uint64_t kMaxPendingBytes = 8ull << 30;
    static constexpr int kMaxSpares = 2;                        // a checkpoint and a turn exported per request

private:
    enum class Why { Evict, Idle, Shutdown };
    struct Entry {
        std::vector<int32_t> tokens;
        Kind kind;
        uint64_t last_used = 0;  // LRU clock
        double touched_s = 0;    // put or last hit (the idle rule)
        HostBuffer ram;          // empty: not in RAM
        std::string path;        // empty: not on disk
        uint64_t file_bytes = 0;
        int readers = 0;           // with_state / the writer using ram (it must stay put)
        bool queued = false;       // a write is queued or running
        bool drop_ram = false;     // leave RAM once written (an eviction)
        bool dead = false;         // removed from the index while in use; the last user cleans up
        bool no_room = false;      // the disk had no room for it (or evicted it): no idle write until it's used again
        bool ram_only = false;     // never written to disk (a one-shot request's); leaving RAM drops it
        std::shared_ptr<Entry> base;  // a delta's base (a full entry, tokens[0, base->tokens.size())); null: whole
        int dependents = 0;           // deltas on this entry
        int64_t saved_by = 0;         // the request that put it (log lines; 0: none)
    };
    using EntryPtr = std::shared_ptr<Entry>;
    struct Job {
        EntryPtr e;
        Why why;
        uint64_t bytes;  // the state's size, counted in pending_bytes_
    };
    void writer();
    void write_entry(Job &job);
    // A state from disk: header, tokens (must equal prompt[0, n)) and checksum checked; throws.
    void load_file(const std::string &path, const std::vector<int32_t> &prompt, int64_t n, HostBuffer &state) const;
    // file_total of e, plus its base's if a write of e would have to write the base first.
    uint64_t write_bytes_locked(const Entry &e) const;
    int find_locked(const int32_t *tokens, int64_t n) const;
    int find_locked(const Entry *e) const;
    void remove_locked(size_t i);                      // from the index: RAM and disk
    void drop_ram_locked(Entry &e, bool to_spare);     // its buffer becomes a spare, or trash
    void release_locked(Entry &e);                     // a reader / the writer is done with e
    void unlink_locked(Entry &e);                      // its file
    void keep_spare_locked(HostBuffer b);
    void make_room_locked(uint64_t extra = 0);         // the RAM rule: MemAvailable >= ram_margin + extra
    void evict_disk_locked(uint64_t need);
    // Makes room for `need` bytes on disk by evicting entries used less recently than `last_used`; false if it can't.
    bool disk_room_locked(uint64_t need, uint64_t last_used);
    bool spend_budget_locked(uint64_t bytes);
    double budget_now_locked() const;  // write budget bytes left now (refilled since budget_at_); no spending
    // The writer's state line, every kSummarySeconds while something changed (puts, hits, writes, drops).
    void summary_locked(double now);
    static constexpr double kSummarySeconds = 600;
    void enqueue_locked(const EntryPtr &e, Why why);
    void write_idle_locked(double now);
    uint64_t prefault_want_locked() const;  // bytes of the spare to populate now; 0 = none
    uint64_t mem_available() const;
    uint64_t disk_free() const;

    std::string dir_, fingerprint_;
    uint64_t max_bytes_;
    Options opt_;
    mutable std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::vector<EntryPtr> entries_;
    std::deque<Job> queue_;
    std::vector<HostBuffer> spares_;
    std::vector<HostBuffer> trash_;  // unmapped after the lock is released (~20 ms for 5 GiB of huge pages)
    uint64_t pending_bytes_ = 0, clock_ = 0;
    uint64_t ram_bytes_ = 0, spare_bytes_ = 0, disk_bytes_ = 0;
    uint64_t prefault_bytes_ = 0;  // the spare size to keep ready: the last saved state + prefault_headroom
    double budget_bytes_ = 0, budget_at_ = 0;  // write budget tokens and when they were last topped up
    double summary_at_ = 0;                    // the last summary line (steady clock, s)
    uint64_t summary_sig_ = 0;                 // clock_ + writes + RAM evictions at that line: unchanged = no line
    bool stop_ = false, writing_ = false, shutting_down_ = false;
    PromptCacheStats stats_;
    std::thread thread_;
};

}  // namespace strix
