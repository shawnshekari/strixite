#pragma once

// PLE n-gram rows for the forward. NgramRowSource is what Qwen4ExpModel reads rows through; two sources:
//   NgramTableRows      - the converted table file (formats/ngram_table.hpp): rows read on demand from SSD
//                         (the table stays on SSD), deduplicated and fetched with up to 32 concurrent preads, and an
//                         LRU cache of recently used rows (n-gram ids are heavily skewed - frequent n-grams repeat
//                         across a conversation and its prefills).
//   NgramRowsFromShards - interim (runtime/ngram_rows.hpp): straight from the HF checkpoint's shards.
// The caller picks one explicitly (Qwen4ExpModel: a table file or a checkpoint directory) - not a fallback.

#include "formats/ngram_table.hpp"
#include "runtime/ple_hash.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strix {

class NgramRowSource {
public:
    virtual ~NgramRowSource() = default;
    virtual const PleHashParams &hash_params() const = 0;
    virtual int64_t row_dim() const = 0;
    // rows [n] table row ids (ple_ngram_ids output) -> out [n, row_dim] BF16 bits. Throws on an id outside the table.
    virtual void gather(const int64_t *rows, int64_t n, uint16_t *out) const = 0;
    // A hint: rows [n] a gather will soon ask for - a source with a row cache reads the missing ones into it now,
    // so that gather finds them there. A source without a cache does nothing (the gather reads them as always).
    virtual void prefetch(const int64_t *rows, int64_t n) const { (void)rows, (void)n; }
};

// Concurrent row fetch shared by both sources: fetch(u) for u in [0, n) on the caller's thread and up to
// max_threads - 1 of a persistent pool of reader threads (shared by concurrent calls); returns when all n are done
// and rethrows the first failure.
void parallel_rows(size_t n, size_t max_threads, const std::function<void(size_t)> &fetch);

class NgramTableRows : public NgramRowSource {
public:
    // cache_rows: rows kept in the LRU cache (0: no cache), row_dim BF16 values each (320 B for the shipping table:
    // the default 2^18 rows are ~84 MB).
    static constexpr int64_t kDefaultCacheRows = 1 << 18;
    explicit NgramTableRows(const std::string &path, int64_t cache_rows = kDefaultCacheRows);
    const PleHashParams &hash_params() const override { return file_.info().params; }
    int64_t row_dim() const override { return file_.info().row_dim; }
    void gather(const int64_t *rows, int64_t n, uint16_t *out) const override;
    void prefetch(const int64_t *rows, int64_t n) const override;
    int64_t cache_rows() const;
    // RAM per cache row: the row's BF16 values + its slot's LRU links and index entry (an unordered_map node and
    // bucket, ~40 B - estimated, not measured).
    uint64_t cache_bytes_per_row() const { return (uint64_t)row_dim() * 2 + 24 + 40; }
    // Empties the cache and reallocates it at `rows` (0..2^26), freeing the old arrays - the startup memory plan's
    // shrink (serve/memory_plan.hpp). Thread-safe; gathers meanwhile wait.
    void resize_cache(int64_t rows) const;
    const NgramTableFile &file() const { return file_; }
    // Gathers: rows requested, distinct rows per gather (summed), of those served from the cache / read from the
    // file. Prefetches: rows read from the file into the cache (a later gather counts them as hits).
    struct Stats {
        uint64_t requested = 0, unique = 0, hits = 0, reads = 0, prefetched = 0;
    };
    Stats stats() const;
    // For measurements (tools/bench_replay --ple-after-prefill): empties the row cache (its counters stay), as after a
    // server restart; and asks the kernel to drop the table file's clean pages from the page cache, as under memory
    // pressure (posix_fadvise DONTNEED - advisory: pages another process holds mapped may stay).
    void clear_cache() const;
    void drop_page_cache() const;

private:
    NgramTableFile file_;
    mutable int64_t cap_;  // under mu_ (resize_cache)
    // LRU over cap_ slots: slot -> row id, doubly linked most- to least-recent.
    mutable std::mutex mu_;
    mutable std::unordered_map<int64_t, int32_t> where_;
    mutable std::vector<int64_t> slot_row_;
    mutable std::vector<int32_t> prev_, next_;
    mutable std::vector<uint16_t> data_;
    mutable int32_t head_ = -1, tail_ = -1, used_ = 0;
    mutable Stats stats_;
    void touch(int32_t slot) const;
    // Sorted distinct ids of rows [n] (validated). fill: the cached ones copied to buf [uniq, row_dim] and touched
    // (under mu_), the missing ones read from the file into buf without the lock, then added to the cache (under
    // mu_; a row another caller added meanwhile is kept). Returns how many were read from the file.
    std::vector<int64_t> unique_rows(const int64_t *rows, int64_t n, const char *fn) const;
    size_t fill(const std::vector<int64_t> &uniq, uint16_t *buf) const;
    int32_t take_slot(int64_t row) const;
};

}  // namespace strix
