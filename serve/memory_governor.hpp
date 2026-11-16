#pragma once

// The memory governor: the one object that owns the RAM budget on this unified-memory machine - the floor
// MemAvailable must stay above, and the single /proc/meminfo reading every consumer's decision is made from. On
// Strix Halo the GPU allocates from the same physical RAM as everything else, and components that each read that
// number at their own moment raced: a rule saw "enough, evict nothing" and a check a few MiB later saw "short,
// refuse", with all of the tier's entries still free to leave. The rule is therefore: read once, decide on that
// one reading.
//
// Consumers (priority order): a slot's K/V pages (a live conversation), in-flight exports / loads, RAM-tier
// entries (evictable to disk), spare buffers (free to drop). A request that does not fit takes from the lower
// priorities first - the reclaim itself lives with the memory (PromptCache::make_room_locked frees spares and RAM
// entries today; parking will add an idle slot's K/V pages); this object owns the floor, the reading and the
// rule, so every consumer answers "is there room?" the same way and a refusal is real.
//
// Reservations: a reading can't see an allocation that was granted but hasn't landed yet, so two callers on two
// slots could each pass the rule on the same reading and both allocate - together past the floor (strixite #42).
// try_reserve checks and books the bytes in one step; every reading subtracts the booked bytes until the caller's
// allocation has landed and it calls release.
// Thread-safe: the booking has its own lock; available() reads /proc/meminfo, which needs no lock, and the rule is a
// pure compare.

#include "common/check.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

namespace strix {

class MemoryGovernor {
public:
    // floor_bytes: keep MemAvailable at or above this (the RAM margin - 4 GiB served; a machine at ~0
    // MemAvailable hard-hangs).
    // avail_fn: the MemAvailable reading, bytes (null: the caller must set one - tests fake it through the
    // owner's option).
    explicit MemoryGovernor(uint64_t floor_bytes, std::function<uint64_t()> avail_fn = nullptr)
        : floor_bytes_(floor_bytes), avail_fn_(std::move(avail_fn)) {}

    void set_avail_fn(std::function<uint64_t()> avail_fn) { avail_fn_ = std::move(avail_fn); }

    uint64_t floor() const { return floor_bytes_; }

    // One reading of MemAvailable, less the bytes booked by try_reserve and not yet released. A decision sequence
    // (rule, reclaim, check) takes this once and passes the value around - never reads again mid-decision (the
    // header note on why).
    uint64_t available() const {
        STRIX_CHECK(avail_fn_ != nullptr, "MemoryGovernor::available: no MemAvailable function wired");
        const uint64_t raw = avail_fn_(), booked = reserved_.load();
        return raw > booked ? raw - booked : 0;
    }

    // The rule on a fresh reading and, if it allows `bytes`, those bytes booked - one step, so a second caller's
    // check already sees them. False: nothing booked. The caller releases them once its allocation has landed.
    bool try_reserve(uint64_t bytes) {
        STRIX_CHECK(bytes < (1ull << 40), "MemoryGovernor::try_reserve: ", bytes, " bytes, expected < 1 TiB");
        std::lock_guard<std::mutex> lock(reserve_mu_);
        if (!room(available(), bytes)) return false;
        reserved_ += bytes;
        return true;
    }

    // Gives back bytes try_reserve booked (their allocation has landed, or won't happen).
    void release(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(reserve_mu_);
        STRIX_CHECK(bytes <= reserved_.load(), "MemoryGovernor::release: ", bytes, " bytes, but only ", reserved_.load(),
                    " are booked");
        reserved_ -= bytes;
    }

    uint64_t reserved() const { return reserved_.load(); }  // bytes booked and not yet released

    // The RAM rule: an allocation of `extra` bytes is allowed against this reading. avail: the caller's reading
    // (available()); extra 0 is the floor check the margin watch runs. Pure: the caller decides what a "no"
    // reclaims, then re-decides on the same or a fresh reading per its own protocol.
    bool room(uint64_t avail, uint64_t extra) const { return avail >= floor_bytes_ + extra; }

    // room() with a fresh reading - for callers with no decision yet in flight (the watch, the prefault check).
    bool room_now(uint64_t extra) const { return room(available(), extra); }

    // The bytes a request for `extra` falls short by on this reading (0: it fits) - what a reclaim loop frees.
    uint64_t shortfall(uint64_t avail, uint64_t extra) const { return room(avail, extra) ? 0 : floor_bytes_ + extra - avail; }

private:
    uint64_t floor_bytes_;
    std::function<uint64_t()> avail_fn_;
    std::mutex reserve_mu_;               // try_reserve's check-and-book and release
    std::atomic<uint64_t> reserved_{0};  // booked bytes not yet released
};

}  // namespace strix
