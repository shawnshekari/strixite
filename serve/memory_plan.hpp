#pragma once

// The startup memory check (strixite issue #6: a 27B sidecar next to the server left MemAvailable at 0.4 GiB, and the
// GPU allocating from the same RAM turned that into driver faults and a segfault instead of a clean error). Run once
// after the weights and every GPU buffer are allocated - nothing on the GPU grows after that - with MemAvailable
// measured then. What can still grow is host RAM: the prompt cache (its RAM tier, spare buffers, the export buffer of
// every save) and nothing else of note (the n-gram row cache is allocated whole at startup).
//
// The rule: the prompt cache must be able to save the largest conversation the capacity allows while keeping its RAM
// margin - MemAvailable >= max(margin, floor) + one state at full capacity. Without a prompt cache, MemAvailable >=
// floor. The floor (kLowMemoryFloor) is what the desktop, the page cache and the server's own pinned staging buffers
// need; a test machine hard-hung at 0.5 GB MemAvailable (2026-10-08).
//
// low-memory = fail (the default everywhere, 2026-10-10): short of the rule, startup fails with what was
// measured, what is needed, and the settings that would fit - including low-memory = adapt.
// low-memory = adapt: the RAM caches are turned down, cheapest loss first, re-checked after each step, each step one
// WARNING (measured, changed, impact, how to get it back):
//   1. below margin + 2 states: spare buffers and prefault off (the next export maps a fresh buffer: ~0.2 s at
//      100-200k context instead of ~0.05);
//   2. below margin + 1 state, with the margin above the floor: the RAM tier off - every save goes to disk (within the write budget) and leaves RAM,
//      other conversations resume from disk; the margin for an allocation becomes the floor;
//   3. below floor + 1 state: the n-gram row cache shrunk to fit, down to 2^18 rows;
//   4. still short: conversations past the largest state that fits above the floor aren't saved; lower capacity.
//   Below the floor even after that: a WARNING naming capacity - it starts anyway, as asked.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strix {

constexpr uint64_t kLowMemoryFloor = 4ull << 30;

struct MemoryPlanInput {
    bool adapt = false;            // low-memory = adapt (false: fail)
    uint64_t mem_available = 0;    // bytes, measured after the backend allocated
    uint64_t gpu_bytes = 0;        // what the backend allocated (for the message)
    bool prompt_cache = true;      // prompt-cache-gib > 0
    uint64_t ram_margin = 0;       // prompt-cache-ram-margin-gib, bytes
    int64_t capacity = 0;          // positions
    std::function<uint64_t(int64_t)> state_bytes;  // a saved state's size at n tokens (n <= capacity); required
    int64_t row_cache_rows = 0;    // ngram-cache-rows as allocated (0: no row cache, e.g. an HF checkpoint source)
    uint64_t row_cache_row_bytes = 0;  // RAM per cached row (data + index)
    int64_t row_cache_min_rows = 0;    // the shrink stops here (NgramTableRows::kDefaultCacheRows)
};

struct MemoryPlan {
    bool refuse = false;           // low-memory = fail and short: refusal holds the message
    std::string refusal;
    bool spares_off = false;       // step 1: PromptCache::Options max_spares = 0, prefault = false
    bool ram_tier_off = false;     // step 2: PromptCache::Options ram_tier = false, ram_margin = floor
    int64_t row_cache_rows = 0;    // step 3: the rows to keep (== the input's when unchanged)
    int64_t max_saved_tokens = -1; // step 4: saves past this many tokens can't fit (-1: no limit)
    uint64_t need = 0;             // the rule's bytes for the settings as given
    std::vector<std::string> info;      // startup lines, INFO
    std::vector<std::string> warnings;  // startup lines, WARNING (one per step taken)
    std::string status;            // a short note for the prompt cache's status line ("" = as configured)
};

// Validates the input (sizes, a state_bytes function when a prompt cache is on) and plans; never touches memory.
MemoryPlan plan_memory(const MemoryPlanInput &in);

// "fail" / "adapt" -> adapt; anything else throws, naming the choices.
bool parse_low_memory(const std::string &value);

}  // namespace strix
