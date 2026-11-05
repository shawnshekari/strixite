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
// A RAM entry may also be held in parked form (put_paged): the state's big ranges are physical K/V pages the GPU
// session handed over instead of a host copy, and the request path never touches those bytes - a conversation
// switch copies ~10 MB, not gigabytes. The file it writes is the same file (write_paged_state gathers the halves
// back into order). The cache is host-only and never makes the page release a HIP call: retired records wait in
// a list the owner drains (take_retired), through the GPU arbiter.
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
// the disk - no eviction, idle or shutdown write; leaving RAM drops them (counted as ram_only_dropped). On one day
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
//   checkpoint would go with it); the checkpoints_per_chain trim skips a base while a delta needs it (an
//   agent harness that sends every terminal output as a user message made a whole ~1.2 GB checkpoint a request, and
//   264k conversations a whole 7.4 GB one).
// On disk a delta is a version-2 file whose header names its base by length (the base's tokens are the delta's own
// first base_n tokens).
//
// Buffers: a state leaving RAM (replaced, evicted, written after eviction) becomes a spare - up to kMaxSpares,
// the largest kept - which take_buffer() hands to the next export or disk load: at 100-200k context a fresh buffer
// costs ~0.2 s of page population, a reused one nothing (3.7 GB export 205-230 ms fresh vs
// ~48 ms reused).
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

#include "runtime/paged_state.hpp"  // PagedState: the host-only view of a parked conversation
#include "serve/host_buffer.hpp"
#include "serve/memory_governor.hpp"

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
    int64_t writes_evict = 0, writes_idle = 0, writes_shutdown = 0, writes_renew = 0;  // writes by rule
    int64_t ram_evicted = 0;  // entries that left RAM for lack of memory (written, already on disk, or dropped)
    int64_t ram_evicted_watch = 0;  // of those, by the writer's margin watch (another process grew between saves)
    int64_t ram_only_dropped = 0;  // RAM-only entries that left RAM (never written, by design)
    int64_t rejected_space = 0, rejected_budget = 0, rejected_queue = 0;  // eviction writes refused, by reason
    int64_t room_refused = 0;  // make_room_for found no room: a save skipped or a disk load refused
    int64_t room_waits = 0;    // make_room_for waited for evicted entries' writes to give their RAM back (polls)
    int64_t evicted = 0, replaced = 0, invalid = 0;  // disk LRU / replaced turns / damaged or foreign
    int64_t checkpoints_dropped = 0;                 // older Checkpoints over checkpoints_per_chain
    int64_t prefaulted = 0;                          // spares the writer populated ahead of an export
    int64_t buffers_reused = 0;  // take_buffer handed out an on-disk RAM entry's buffer (memory near the margin)
    int64_t turn_buffers_reused = 0;  // take_replaced_buffer handed out the replaced turn's buffer
    int64_t delta_puts = 0, delta_bytes = 0;  // Turns saved as deltas, and their state bytes
    int64_t deltas_dropped = 0;               // deltas removed with their base (or whose base file was missing)
    double load_seconds = 0, write_seconds = 0;
    int64_t pending = 0;  // queued writes
    // Resume points (PromptCache "Resume points"): held now and their bytes; saved, resumed from, dropped over the
    // cap (Options::max_points), and left RAM for the margin (they go before any entry).
    int64_t points = 0, point_bytes = 0, point_puts = 0, point_hits = 0, points_dropped = 0, points_evicted = 0;
};

class PromptCache {
public:
    enum class Kind : uint32_t { System = 1, Turn = 2, Checkpoint = 3 };

    struct Options {
        // Keep MemAvailable above this (RAM entries leave first; make_room_for holds it at every allocation). 5 GiB
        // (was 4, and 24 before that - 24 was set against direct compaction from MADV_HUGEPAGE buffers and buffered
        // writes, both gone now): at 512k capacity / chunk 16384, two 264k conversations switching, n-gram table warm,
        // MemAvailable bottomed at 5.5 GiB with 0 compaction stalls, RAM resumes 1.2-1.4 s.
        uint64_t ram_margin = 5ull << 30;
        double idle_seconds = 3600;  // an entry idle in RAM this long is written (and kept in RAM)
        // The margin watch: the writer applies the RAM rule this often, so memory another process takes between saves
        // (a build or test an agent runs, a sidecar) is answered within seconds, not at the next save (it was held
        // only at allocations before). One /proc/meminfo read a check. Below the margin it frees down to
        // margin + ram_watch_headroom: freeing only the shortfall let a hog growing 0.5 GiB/s outrun a 2 s watch to
        // 2.8 GiB.
        double ram_watch_seconds = 1;
        uint64_t ram_watch_headroom = 1ull << 30;
        // Budget for eviction and idle writes; 0 = no limit. 32 GiB/h: the served value (reasons in
        // deploy/strix-server.conf); the old 64 predates delta entries.
        double write_gib_per_hour = 32;
        uint64_t disk_free_margin = 32ull << 30;  // a write never leaves less free space on the cache's filesystem
        double shutdown_seconds = 40;       // the destructor's flush stops after this long
        int checkpoints_per_chain = 2;      // Checkpoints kept per conversation (>= 1)
        // Resume points kept (put_point; 0 = none saved): ~0.11 GB each. 16 is ~1.8 GiB at most, and the RAM rule
        // drops them before any entry. See "Resume points" below.
        int max_points = 16;
        // Delta entries: a Turn is a delta on its base while it adds at most this many tokens (and at most a quarter of
        // the base's); 0 = every entry whole. 65536 (was 32768): below a 262k base the quarter rule
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
        // make_room_for waits up to this long for evicted entries still being written to give their RAM back before
        // it refuses (a refused disk load re-prefills its whole prompt: 154 s at 239k; writes run at ~1 GB/s).
        double room_wait_seconds = 20;
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
    // The paged form: the state's big ranges are the parked pages `paged` describes
    // and `small` is the rest of it - together exactly the bytes put() would have been handed as one buffer, so
    // the entry's file is byte for byte the same (write_paged_state gathers them back into order). The point is
    // the request path: those bytes never pass through a host buffer, so switching conversations copies ~10 MB
    // per save instead of multiple GB. Same rules as put (a duplicate refreshes, turns are replaced, RAM is made
    // room for); the record's bytes count in the RAM tier from this call until it is given back - through
    // take_retired(), the only way a parked record ever leaves this cache: the cache is host-only and never
    // touches the pages, its owner releases the ids it drains. Resume: with_paged, or the file once written.
    bool put_paged(std::vector<int32_t> tokens, Kind kind, HostBuffer small, PagedState paged,
                   bool ram_only = false, int64_t base_n = 0, int64_t saved_by = 0);
    // The ids of parked records no entry holds anymore (written, evicted, dropped, or surplus in a duplicate
    // put). Releasing one is HIP work on the session that carved the pages, so the cache never does it - the
    // engine drains this list while it holds the GPU, through the arbiter like any other device work. The
    // pages come back a moment late instead of never; a parked id must never outlive that session (the
    // server's declaration order: backend before cache, the cache's writer flushed before the backend goes).
    std::vector<int64_t> take_retired();
    // Hands a paged entry that is IN RAM to `use` (its small bytes + the record's byte map), synchronously, on
    // the caller's thread: the resume maps the parked pages into a slot's next generation and imports the
    // small half. The entry is pinned (readers) for the call, so nothing evicts its pages mid-read. False =
    // no entry, or it is on disk (its file holds the whole state - with_state reads that), or a write has it.
    // `consumed` (default: yes): use mapped the record's pages into a slot - a parked record is handed back
    // exactly once, so the entry gives up its paged half inside the same pin. What can remain of it is the
    // file (which has the whole state); with no file yet, the entry is gone (a miss later, never a half
    // state). A caller that only reads the byte map passes false.
    bool with_paged(const std::vector<int32_t> &prompt, int64_t n,
                    const std::function<void(const HostBuffer &, const PagedState &)> &use, bool consumed = true);
    // Resume points (strixite #32 / #33). A client that edits a message it sent long ago (an agent pruning old tool
    // output, a front-end dropping old screenshots, an edited message) leaves no entry that matches past the edit
    // but the system prompt's: the recurrent state can only resume where it was saved. A point is a saved state's
    // small half alone (LmBackend::small_states - no K/V rows, ~0.11 GB at any depth) at a message boundary of a
    // conversation; the K/V below it is the same in every state that starts with those tokens, so the engine lays
    // a point over K/V that holds them - the slot's live state, or a longer entry of the same conversation it
    // loads first (donor()). The resume then starts at the last point before the edit.
    // Points live in RAM only: never written, not entries (best(), contains(), nearest() and delta_base() don't
    // see them), at most Options::max_points, least recently used dropped first; under the RAM rule they leave
    // before any entry (they are the cheapest thing to lose). A duplicate refreshes the one held. False (nothing
    // kept): max_points 0, or no RAM tier.
    bool put_point(std::vector<int32_t> tokens, HostBuffer small, int64_t saved_by = 0);
    // The point covering the longest prefix of prompt with at most max_n tokens; 0 if none.
    int64_t best_point(const std::vector<int32_t> &prompt, int64_t max_n) const;
    // use(small) with the point of prompt[0, n), off the cache's lock (the point stays alive for the call even if
    // it is dropped meanwhile); false if there is none. Counts a hit.
    bool with_point(const std::vector<int32_t> &prompt, int64_t n, const std::function<void(const HostBuffer &)> &use);
    // The K/V a point at prompt[0, n) can lie on: the tokens of an entry at least n long that starts with
    // prompt[0, n) - parked in RAM first (its pages map back), then a RAM copy, then a disk file; the shortest of
    // a kind (the least to load). Empty: none. The caller loads it with with_paged / with_state, passing these
    // tokens as the prompt.
    std::vector<int32_t> donor(const std::vector<int32_t> &prompt, int64_t n) const;
    // A spare buffer (the largest; empty if none) for an export or a load, resized by whoever fills it - and back.
    // Near the margin with no spare of the last saved state's size, the least recently used RAM entry already on
    // disk gives up its buffer instead (no fresh mapping under memory pressure).
    HostBuffer take_buffer();
    // The RAM buffer of the Turn entry a put of tokens[0, n) will replace (the longest Turn it extends - the same
    // conversation one step back), taken now: that entry leaves the index (RAM and disk) as the put would do anyway.
    // The export then grows it by the turn's few MB instead of populating a fresh multi-GB buffer - near the RAM
    // margin spares are the first thing dropped, so there usually is none (2-6 GB exports took 278-546 ms
    // median). Empty if there is no such entry in RAM, or it is being read or written: use take_buffer(). If the
    // export then fails, the old turn is lost - a miss later, never a wrong state. Never a base: neither one with
    // deltas nor the entry of tokens[0, keep_n) - the base the coming put will name (delta_base()), whose old delta
    // this call may just have taken.
    HostBuffer take_replaced_buffer(const int32_t *tokens, int64_t n, int64_t keep_n = 0);
    // Before a buffer grows by `bytes` (an export into a taken buffer, a disk load): spares and least recently used
    // entries leave RAM until MemAvailable >= ram_margin + bytes, measured after their memory is unmapped. False if it
    // still isn't (an eviction write still queued holds its RAM until written): the caller then doesn't grow the
    // buffer - the save is skipped, the load refused - so the margin holds at every allocation, not only after a put
    // (a whole 7.4 GB export a request, checked only afterwards, once took the machine to 0.5 GB and hung it).
    // release_retired: a caller that can give parked records back (it holds the GPU, or hands work to the arbiter)
    // passes it, and the ids this call's evictions retire go back through it BEFORE MemAvailable is read again.
    // Without it a parked entry's eviction frees nothing until the engine's next drain - the rule would count
    // its pages as gone, read MemAvailable unchanged, and refuse (or evict the whole tier for nothing).
    using RetiredRelease = std::function<void(const std::vector<int64_t> &)>;
    bool make_room_for(uint64_t bytes, const RetiredRelease &release_retired = {});
    // Called on the writer thread, off the cache's lock, when parked ids wait in the retired list after the
    // writer's own work (a margin-watch eviction, a finished eviction write): the owner's cue to drain them while
    // it is idle. The hook must not call back into this cache. Set {} to clear it - that waits for a call in
    // progress, so the owner may go away right after.
    void set_retired_hook(std::function<void()> hook);
    // What holds the RAM tier, for the line after a refusal: entries (in use by a reader or the writer / queued for a
    // write / free to leave), spares, bytes still leaving. Free-to-leave entries left after a refusal are a bug.
    std::string ram_summary() const;
    uint64_t mem_available_now() const { return mem_available(); }  // MemAvailable (or the test's probe), bytes
    // The RAM budget owner (floor, the rule, one reading at a time): the governor every consumer of this memory
    // answers to - K/V growth and the RAM tier decide on the same number through it, and parking will ask it too.
    const MemoryGovernor &governor() const { return gov_; }
    uint64_t disk_free_now() const { return disk_free(); }          // free bytes on the directory's filesystem
    void give_back(HostBuffer b);
    // Queues the idle rule's writes now (the writer also does this on its own every few seconds).
    void write_idle();
    // The soft K/V reset's first half (Engine::renew_kv): every entry whose state is parked pages is written to
    // disk now - no budget, like the shutdown rule - and its record retired (take_retired), or, when it cannot be
    // (a one-shot entry, no disk room, a failed write), it leaves the cache. Waits up to max_seconds for the
    // writes. After a call with left == 0 no entry holds a parked record, which is what lets the session that
    // owns the pages be destroyed; left > 0 (a write still running at the deadline) = do not destroy it.
    struct Spill {
        int64_t written = 0, dropped = 0, left = 0;
        uint64_t bytes = 0;  // the parked page bytes of the entries now on disk
        double seconds = 0;
    };
    Spill spill_parked(double max_seconds);
    void flush();  // waits until no write is queued or running (tests)
    PromptCacheStats stats() const;
    const std::string &dir() const { return dir_; }
    const Options &options() const { return opt_; }

    // Queued eviction writes beyond max(this, 4 x the last saved state) are refused (Options::max_pending_bytes): 8 GiB
    // alone held two ~3.7 GB states, and switching two long conversations dropped an entry a run on "write queue full".
    static constexpr uint64_t kMaxPendingBytes = 8ull << 30;
    static constexpr int kMaxSpares = 2;                        // a checkpoint and a turn exported per request

private:
    enum class Why { Evict, Idle, Shutdown, Renew };
    struct Entry {
        std::vector<int32_t> tokens;
        Kind kind;
        uint64_t last_used = 0;  // LRU clock
        double touched_s = 0;    // put or last hit (the idle rule)
        HostBuffer ram;          // empty: not in RAM (for a paged entry: the state's small bytes only)
        PagedState paged;        // non-empty: the state's big ranges are parked pages, held until written
        std::string path;        // empty: not on disk
        uint64_t file_bytes = 0;
        int readers = 0;           // with_state / the writer using ram (it must stay put)
        bool queued = false;       // a write is queued or running
        bool drop_ram = false;     // leave RAM once written and unread (an eviction; the last release_locked drops it)
        bool releasing = false;    // its RAM is given back as the writer writes it: unreadable (with_state waits)
        bool dead = false;         // removed from the index while in use; the last user cleans up
        bool no_room = false;      // the disk had no room for it (or evicted it): no idle write until it's used again
        bool ram_only = false;     // never written to disk (a one-shot request's); leaving RAM drops it
        std::shared_ptr<Entry> base;  // a delta's base (a full entry, tokens[0, base->tokens.size())); null: whole
        int dependents = 0;           // deltas on this entry
        int64_t saved_by = 0;         // the request that put it (log lines; 0: none)
        // What the rest of the cache asks about an entry's state. A paged entry's state is in two places -
        // `ram` the small bytes, `paged` the parked pages - and NEITHER half alone is a state, so a paged entry
        // whose record was given back is a DISK entry: both halves go empty together (write_entry, drop_ram).
        size_t state_size() const { return paged.empty() ? ram.size() : paged.state_bytes; }  // 0: no RAM copy
        uint64_t ram_cost() const { return ram.capacity() + paged.bytes(); }  // what ram_bytes_ counts
        bool in_ram() const { return ram_cost() > 0; }
    };
    using EntryPtr = std::shared_ptr<Entry>;
    struct Point {  // a resume point (put_point): tokens [0, n) and the state's small half at n
        std::vector<int32_t> tokens;
        std::shared_ptr<const HostBuffer> small;
        uint64_t last_used = 0;
        int64_t saved_by = 0;
    };
    // Drops the least recently used point (its bytes off ram_bytes_; the buffer is unmapped when its last holder
    // lets go - here, under the lock (~0.11 GB, a millisecond or two), unless a with_point call still reads it).
    // False if there is none.
    bool drop_point_locked();
    struct Job {
        EntryPtr e;
        Why why;
        uint64_t bytes;  // the state's size, counted in pending_bytes_
    };
    void writer();
    void write_entry(Job &job);
    bool put_impl(std::vector<int32_t> tokens, Kind kind, HostBuffer state, PagedState paged, bool ram_only,
                  int64_t base_n, int64_t saved_by);  // put() and put_paged() both land here
    // Gathers a paged entry's state to the file in whole 4 KiB blocks (O_DIRECT's requirement): the parked
    // ranges and, in the gaps between them, the small buffer, in pack order - the bytes put() would have
    // written from one buffer. Hashed on the way, and the value comes back for the header, which for a paged
    // entry is therefore written AFTER its state (the whole-buffer path has its hash before it writes).
    uint64_t write_paged_state(int fd, const Entry &e, uint64_t state_at, const std::string &path) const;
    // A state from disk: header, tokens (must equal prompt[0, n)) and checksum checked; throws.
    void load_file(const std::string &path, const std::vector<int32_t> &prompt, int64_t n, HostBuffer &state) const;
    // file_total of e, plus its base's if a write of e would have to write the base first.
    uint64_t write_bytes_locked(const Entry &e) const;
    int find_locked(const int32_t *tokens, int64_t n) const;
    int find_locked(const Entry *e) const;
    // From the index: RAM and disk. to_spare: its RAM (and its deltas') becomes a spare buffer, else trash - the RAM
    // rule passes false: its evictions must unmap (a spare keeps the memory the eviction was meant to free).
    void remove_locked(size_t i, bool to_spare = true);
    void drop_ram_locked(Entry &e, bool to_spare);     // its buffer becomes a spare, or trash
    void release_locked(Entry &e);                     // a reader / the writer is done with e
    void unlink_locked(Entry &e);                      // its file
    void keep_spare_locked(HostBuffer b);
    // The RAM rule: MemAvailable >= ram_margin + extra. avail: the caller's reading of MemAvailable (UINT64_MAX: read it
    // here) - make_room_for passes the one its check used, so both decide on the same number.
    void make_room_locked(uint64_t extra = 0, uint64_t avail = UINT64_MAX);
    uint64_t leaving_bytes_locked() const;             // RAM of evicted entries not yet written (on its way out)
    // The writer's margin watch (Options::ram_watch_seconds): the RAM rule, its buffers unmapped outside the lock.
    void watch_margin(std::unique_lock<std::mutex> &lock);
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
    // The RAM budget: floor + one-reading rule (serve/memory_governor.hpp), built from the options, the probe the
    // options' (or /proc/meminfo). Every margin decision of the cache answers through this one object.
    MemoryGovernor gov_;
    mutable std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::vector<EntryPtr> entries_;
    std::vector<Point> points_;  // resume points, unordered (LRU by last_used)
    std::deque<Job> queue_;
    std::vector<HostBuffer> spares_;
    std::vector<HostBuffer> trash_;  // unmapped after the lock is released (~20 ms for 5 GiB of huge pages)
    // Parked ids whose record no entry holds anymore, waiting for the engine's take_retired(): releasing is
    // HIP work the cache must not do (it runs on the writer thread), so the ids wait here - added under mu_
    // (a vector swap is not the slow part; the hip calls are), taken whole off-lock.
    std::vector<int64_t> retired_;
    // The owner's drain cue (set_retired_hook) and its own lock: held across the call, so clearing the hook
    // waits for a running call - never taken while mu_ is held.
    std::mutex hook_mu_;
    std::function<void()> retired_hook_;
    void poke_retired();  // calls the hook if ids wait (takes mu_ briefly, then hook_mu_; call without mu_)
    // What a method hands back when it lets RAM go: buffers the destructor unmaps outside the lock. (Parked
    // records stay in retired_ - their owner takes them with take_retired(), never under mu_.)
    // Declared before the lock_guard, as the trash idiom does, so it goes out of scope after the lock is gone.
    struct Handback {
        std::vector<HostBuffer> trash;
        void take(PromptCache *c) { trash.swap(c->trash_); }
        ~Handback() { trash.clear(); }
    };
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

// write_all for a source that is being thrown away: a pwrite loop that hands back the buffer's already-written
// whole pages (MADV_DONTNEED) as the write progresses, so a queued eviction returns its RAM with its bytes instead of
// all at the end (the margin watch's dip). `where` names the file in errors. Returns the bytes
// given back; a madvise failure stops the releasing, not the write.
// on_progress (optional, for tests) sees (bytes written, bytes given back) after every chunk that released.
uint64_t write_all_releasing(int fd, HostBuffer &src, size_t n, uint64_t at, const std::string &where,
                             const std::function<void(uint64_t, uint64_t)> &on_progress = {});

}  // namespace strix
