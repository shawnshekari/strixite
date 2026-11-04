#pragma once

// The host-only half of parking: what a parked conversation looks like to code that cannot name a HIP handle.
//
// A parked conversation's K/V is physical device pages, and the pages belong to the session's one VMM
// reservation: the record that names them holds HIP allocation handles, so it lives in the runtime layer
// (runtime/qwen4exp.hpp, `ParkedKv`). The prompt cache is host-only code (`strix_serve`: JSON, tokenizer,
// matching - built with no HIP on the compile line), and it is the thing that has to *hold* a parked
// conversation while it waits to be resumed or evicted. So the cache holds this instead: everything about a
// parked conversation that is not a handle.
//
// The id is the parked generation the pages were carved into - unique for the process's life (parking never
// refunds one; that is the budget's whole design), so an id names exactly one conversation's pages and can
// never point at a recycled slot. Everything the cache does with it - count its bytes, write its file, give
// it back - goes through the session that owns the record (`paged_state`, `release_parked`, `resume_parked`).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix {

// One range of a paged entry's state pack, sitting in the parked generation: its offset in the pack, its
// length, and its host-readable address (the parked pages are mapped with the host access flag - probe check
// 2's measured "CPU read needs host" - so the disk writer reads them over the host bus). Valid while the
// record lives; the cache must not touch them after it gives the id back.
struct PagedChunk {
    size_t state_off = 0;
    size_t bytes = 0;
    const uint8_t *host = nullptr;
};

// A whole entry's parked form, as the cache sees it: which parked conversation (`parked_id`), how big the
// state is, how much RAM the pages hold while the entry owns them, and which bytes of the state come from
// them. The bytes the chunks do not cover - the mRoPE header, the n-gram history, the per-layer tails - live
// in the entry's own small HostBuffer; together they are exactly what a HostBuffer entry keeps in one buffer,
// so a paged entry's file is byte for byte the same file (see PromptCache::write_paged_state).
// Built by Qwen4ExpSession::paged_state, which is also what checked it tiles the state.
struct PagedState {
    int64_t parked_id = -1;       // the parked generation (its record stays with the session)
    size_t state_bytes = 0;       // the whole state: what the file's state region holds
    size_t page_bytes = 0;        // physical RAM the entry holds by holding this record
    std::string record_name;      // the tally category it is charged under (log lines)
    std::vector<PagedChunk> chunks;

    bool empty() const { return parked_id < 0; }
    size_t bytes() const { return page_bytes; }  // the RAM figure the cache counts (what page_bytes is for)
};

}  // namespace strix
