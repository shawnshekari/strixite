#include "serve/memory_plan.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cstdio>

namespace strix {

namespace {

std::string gib(uint64_t bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f GiB", (double)bytes / (double)(1ull << 30));
    return b;
}

std::string tokens(int64_t n) {
    std::string s = std::to_string(n);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, ",");
    return s;
}

// The largest n (multiple of 1024, <= capacity) whose state fits in `room` bytes; 0 if none does.
int64_t largest_fitting(const MemoryPlanInput &in, uint64_t room) {
    int64_t lo = 0, hi = in.capacity / 1024;  // in units of 1024 tokens
    while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if (in.state_bytes(mid * 1024) <= room) lo = mid;
        else hi = mid - 1;
    }
    return lo * 1024;
}

// "capacity = 262144 (a state 6.8 GiB, ~6.8 GiB less GPU memory)" for the halved and quartered capacities.
std::string capacity_options(const MemoryPlanInput &in) {
    std::string s;
    const uint64_t full = in.state_bytes(in.capacity);
    for (int64_t div : {2, 4}) {
        const int64_t c = in.capacity / div;
        if (c < 4096) break;
        const uint64_t st = in.state_bytes(c);
        // The GPU's KV cache and block keys scale with capacity exactly as the state's do: at least this much less.
        s += std::string(s.empty() ? "" : ", ") + "capacity = " + std::to_string(c) + " (state " + gib(st) + ", at least " +
             gib(full - st) + " less GPU memory)";
    }
    return s;
}

}  // namespace

bool parse_low_memory(const std::string &value) {
    if (value == "fail") return false;
    if (value == "adapt") return true;
    STRIX_FAIL("low-memory = '", value, "': expected fail (refuse to start when memory is short) or adapt (turn the "
               "RAM caches down and start)");
}

MemoryPlan plan_memory(const MemoryPlanInput &in) {
    STRIX_CHECK(in.mem_available < (1ull << 50), "plan_memory: MemAvailable ", in.mem_available, " bytes, expected < 1 PiB");
    STRIX_CHECK(in.capacity >= 1, "plan_memory: capacity ", in.capacity, ", expected >= 1");
    STRIX_CHECK((bool)in.state_bytes, "plan_memory: no state_bytes function (the capacity advice needs it)");
    STRIX_CHECK(in.ram_margin < (1ull << 50), "plan_memory: RAM margin ", in.ram_margin, " bytes, expected < 1 PiB");
    STRIX_CHECK(in.row_cache_rows >= 0 && in.row_cache_min_rows >= 0, "plan_memory: row cache ", in.row_cache_rows,
                " rows, minimum ", in.row_cache_min_rows, " - expected >= 0");
    STRIX_CHECK(in.row_cache_rows == 0 || in.row_cache_row_bytes > 0, "plan_memory: ", in.row_cache_rows,
                " cached rows of 0 bytes each");

    MemoryPlan p;
    p.row_cache_rows = in.row_cache_rows;
    const uint64_t A = in.mem_available, F = kLowMemoryFloor;
    const uint64_t S = in.prompt_cache ? in.state_bytes(in.capacity) : 0;
    const uint64_t M = std::max(in.ram_margin, F);
    const uint64_t row_cache = (uint64_t)in.row_cache_rows * in.row_cache_row_bytes;
    p.need = in.prompt_cache ? M + S : F;
    const std::string measured = "MemAvailable " + gib(A) + " after loading (GPU buffers " + gib(in.gpu_bytes) +
                                 ", n-gram row cache " + gib(row_cache) + ")";
    const std::string rule = in.prompt_cache
                                 ? "the prompt cache's RAM margin " + gib(M) + " + one state at full capacity (" +
                                       tokens(in.capacity) + " tokens) " + gib(S) + " = " + gib(p.need)
                                 : "the floor " + gib(F) + " (no prompt cache)";
    p.info.push_back("startup: memory: " + measured + "; needs " + rule +
                     (in.adapt ? " (low-memory adapt: short of it, the RAM caches are turned down)"
                               : " (low-memory fail: short of it, startup fails; adapt would start with the RAM "
                                 "caches turned down)"));
    if (A >= p.need) {
        if (in.prompt_cache && A < M + 2 * S)
            p.info.push_back("startup: memory: room for one full-capacity state above the margin, not two: spare "
                             "buffers stay only while MemAvailable allows");
        return p;
    }

    if (!in.adapt) {
        std::string fixes = "free memory (another process?), or change: " + capacity_options(in);
        if (in.row_cache_rows > in.row_cache_min_rows)
            fixes += ", ngram-cache-rows = " + std::to_string(in.row_cache_min_rows) + " (frees " +
                     gib((uint64_t)(in.row_cache_rows - in.row_cache_min_rows) * in.row_cache_row_bytes) + ")";
        if (in.prompt_cache && in.ram_margin > F)
            fixes += ", prompt-cache-ram-margin-gib = " + std::to_string(F >> 30) + " (the lowest that counts)";
        if (in.prompt_cache) fixes += ", prompt-cache-gib = 0 (no prompt cache)";
        p.refuse = true;
        p.refusal = "not enough memory for these settings: " + measured + ", short of " + rule + " by " +
                    gib(p.need - A) + ". To fit: " + fixes +
                    ". Or set low-memory = adapt to start anyway with the RAM caches turned down (each step is logged "
                    "with its cost). low-memory = fail is the default because a server short of memory fails later and "
                    "worse (strixite issue #6: driver faults, then a crash)";
        return p;
    }

    uint64_t avail = A;
    std::vector<std::string> status;
    if (in.prompt_cache) {
        // 1. Spares + prefault: a second full state's worth of buffers; their loss is ~0.15 s per export.
        p.spares_off = true;
        status.push_back("spares off");
        p.warnings.push_back("WARNING: low memory (" + measured + "): prompt cache spare buffers and prefault off - each "
                             "export maps a fresh buffer (~0.2 s at 100-200k context instead of ~0.05). Back on with "
                             "more free memory or a lower capacity; low-memory = fail refuses to start instead");
        if (avail < M + S && in.ram_margin > F) {
            // 2. The RAM tier: entries go to disk and leave RAM; an allocation keeps only the floor. (With the margin at
            // the floor it would lower nothing - make_room_for already evicts before every export - and only lose the
            // RAM resumes: skipped when the margin already sits at that floor - make_room_for then evicts down to it
            // at every allocation anyway, so turning the tier off would change nothing but the log.)
            p.ram_tier_off = true;
            status.push_back("RAM tier off");
            p.warnings.push_back("WARNING: low memory: " + gib(avail) + " is short of " + rule +
                                 ": the prompt cache's RAM tier is off - every save goes to disk (within the write "
                                 "budget) and leaves RAM, other conversations resume from disk (the live one still "
                                 "from the GPU snapshot); allocations keep " + gib(F) + " free instead of the " +
                                 gib(M) + " margin. Back on above " + gib(M + S) + " MemAvailable");
        }
    }
    const uint64_t want = in.prompt_cache ? F + S : F;  // what the plan now needs
    if (avail < want && in.row_cache_rows > in.row_cache_min_rows) {
        // 3. The n-gram row cache, down to the default: rows come from the table file instead (SSD reads).
        const uint64_t short_by = want - avail;
        const int64_t drop = std::min<int64_t>(in.row_cache_rows - in.row_cache_min_rows,
                                               (int64_t)((short_by + in.row_cache_row_bytes - 1) / in.row_cache_row_bytes));
        p.row_cache_rows = in.row_cache_rows - drop;
        const uint64_t freed = (uint64_t)drop * in.row_cache_row_bytes;
        avail += freed;
        status.push_back("row cache " + std::to_string(p.row_cache_rows) + " rows");
        p.warnings.push_back("WARNING: low memory: the n-gram row cache shrunk from " + std::to_string(in.row_cache_rows) +
                             " to " + std::to_string(p.row_cache_rows) + " rows (frees " + gib(freed) +
                             ") - more PLE rows read from the table file during prefill. Back with ngram-cache-rows "
                             "and more free memory");
    }
    if (in.prompt_cache && avail < F + S) {
        // 4. Saves past what fits above the floor are skipped (make_room_for refuses them).
        p.max_saved_tokens = avail > F ? largest_fitting(in, avail - F) : 0;
        status.push_back("saves up to " + tokens(p.max_saved_tokens) + " tokens");
        p.warnings.push_back("WARNING: low memory: " + gib(avail) + " leaves room above the " + gib(F) +
                             " floor for a state of " + gib(avail > F ? avail - F : 0) +
                             ": conversations past ~" + tokens(p.max_saved_tokens) +
                             " tokens are not saved (they re-prefill after a switch). To fix: " + capacity_options(in));
    }
    if (avail < F)
        p.warnings.push_back("WARNING: low memory: " + gib(avail) + " is below the " + gib(F) +
                             " floor even with the RAM caches down - the desktop and the server's pinned buffers may run "
                             "out (driver faults, a hang). Lower capacity: " + capacity_options(in));
    for (size_t i = 0; i < status.size(); ++i) p.status += (i ? ", " : "low-memory adapt: ") + status[i];
    return p;
}

}  // namespace strix
