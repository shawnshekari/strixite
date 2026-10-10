#include "runtime/ngram_table.hpp"

#include "common/check.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace strix {

namespace {

// Persistent row readers for parallel_rows: starting a std::thread per read cost more than the read itself (a 16-row
// gather ~1 ms of thread start-up vs ~0.17 ms of SSD reads in parallel). A batch is n indices; the caller works on its
// own batch too and returns when every index is done. Concurrent batches (a gather and a prefetch) share the threads.
class ReadPool {
public:
    static ReadPool &get() {
        static ReadPool pool(32);
        return pool;
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread &t : threads_) t.join();
    }
    void run(size_t n, size_t max_helpers, const std::function<void(size_t)> &fn) {
        Batch b;
        b.fn = &fn, b.n = n, b.helpers_left = std::min(max_helpers, threads_.size());
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(&b);
        }
        cv_.notify_all();
        work(b);  // the caller too
        std::unique_lock<std::mutex> lock(mu_);
        // Off the queue before it goes out of scope; then wait for helpers still inside it.
        queue_.erase(std::find(queue_.begin(), queue_.end(), &b));
        done_cv_.wait(lock, [&] { return b.active == 0; });
        if (b.err) std::rethrow_exception(b.err);
    }

private:
    struct Batch {
        const std::function<void(size_t)> *fn = nullptr;
        size_t n = 0, helpers_left = 0, active = 0;  // helpers_left / active under mu_
        std::atomic<size_t> next{0};
        std::exception_ptr err;  // under mu_
    };
    explicit ReadPool(size_t threads) {
        for (size_t k = 0; k < threads; ++k) threads_.emplace_back([this] { loop(); });
    }
    void work(Batch &b) {  // indices until none are left; the first failure is kept, the rest still run
        for (size_t u; (u = b.next.fetch_add(1)) < b.n;) {
            try {
                (*b.fn)(u);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu_);
                if (!b.err) b.err = std::current_exception();
            }
        }
    }
    void loop() {
        for (;;) {
            Batch *b = nullptr;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] {
                    if (stop_) return true;
                    for (Batch *q : queue_)
                        if (q->helpers_left > 0 && q->next.load() < q->n) return true;
                    return false;
                });
                if (stop_) return;
                // `next` moves without the lock (the caller works on its own batch): the batch that woke this thread
                // may be used up by now - then wait again.
                for (Batch *q : queue_)
                    if (q->helpers_left > 0 && q->next.load() < q->n) {
                        b = q;
                        break;
                    }
                if (b == nullptr) continue;
                --b->helpers_left, ++b->active;
            }
            work(*b);
            {
                std::lock_guard<std::mutex> lock(mu_);
                --b->active;
            }
            done_cv_.notify_all();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::vector<Batch *> queue_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

}  // namespace

void parallel_rows(size_t n, size_t max_threads, const std::function<void(size_t)> &fetch) {
    STRIX_CHECK(max_threads >= 1, "parallel_rows: max_threads = 0");
    STRIX_CHECK(fetch != nullptr, "parallel_rows: empty fetch function");
    if (n == 0) return;
    if (n == 1 || max_threads == 1) {
        for (size_t u = 0; u < n; ++u) fetch(u);
        return;
    }
    ReadPool::get().run(n, std::min(max_threads, n) - 1, fetch);
}

namespace {
constexpr size_t kReadThreads = 32;  // concurrent row reads (SSD queue depth)
}  // namespace

NgramTableRows::NgramTableRows(const std::string &path, int64_t cache_rows) : file_(path), cap_(cache_rows) {
    STRIX_CHECK(cache_rows >= 0 && cache_rows <= (1ll << 26), "NgramTableRows: cache_rows = ", cache_rows,
                ", expected 0..2^26");
    check_ple_hash_params(file_.info().params);
    slot_row_.resize((size_t)cap_), prev_.resize((size_t)cap_), next_.resize((size_t)cap_);
    data_.resize((size_t)(cap_ * file_.info().row_dim));
    where_.reserve((size_t)cap_);
}

void NgramTableRows::touch(int32_t s) const {  // move slot s to the front (most recent)
    if (head_ == s) return;
    if (prev_[(size_t)s] >= 0) next_[(size_t)prev_[(size_t)s]] = next_[(size_t)s];
    if (next_[(size_t)s] >= 0) prev_[(size_t)next_[(size_t)s]] = prev_[(size_t)s];
    if (tail_ == s) tail_ = prev_[(size_t)s];
    prev_[(size_t)s] = -1, next_[(size_t)s] = head_;
    if (head_ >= 0) prev_[(size_t)head_] = s;
    head_ = s;
    if (tail_ < 0) tail_ = s;
}

int32_t NgramTableRows::take_slot(int64_t row) const {  // a free slot, else the least recent one, now holding row
    int32_t s;
    if (used_ < cap_) {
        s = used_++;
        prev_[(size_t)s] = next_[(size_t)s] = -1;
    } else {
        s = tail_;
        where_.erase(slot_row_[(size_t)s]);
    }
    slot_row_[(size_t)s] = row;
    where_[row] = s;
    touch(s);
    return s;
}

std::vector<int64_t> NgramTableRows::unique_rows(const int64_t *rows, int64_t n, const char *fn) const {
    STRIX_CHECK(rows || n == 0, "NgramTableRows::", fn, ": null rows with n = ", n);
    STRIX_CHECK(n >= 0, "NgramTableRows::", fn, ": n = ", n);
    const NgramTableInfo &info = file_.info();
    for (int64_t i = 0; i < n; ++i)
        STRIX_CHECK(rows[i] >= 0 && rows[i] < info.rows, "NgramTableRows::", fn, ": row id ", rows[i], " (entry ", i,
                    ") outside the table's ", info.rows, " rows");
    std::vector<int64_t> uniq(rows, rows + n);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    return uniq;
}

size_t NgramTableRows::fill(const std::vector<int64_t> &uniq, uint16_t *buf) const {
    const NgramTableInfo &info = file_.info();
    const size_t D = (size_t)info.row_dim, rb = info.row_bytes(), vb = D * 2;  // stored / decoded (BF16) row bytes
    std::vector<size_t> miss;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t u = 0; u < uniq.size(); ++u) {
            auto it = cap_ > 0 ? where_.find(uniq[u]) : where_.end();
            if (it == where_.end()) {
                miss.push_back(u);
                continue;
            }
            std::memcpy(buf + u * D, data_.data() + (size_t)it->second * D, vb);
            touch(it->second);
        }
    }
    // The reads run without the lock: a gather doesn't wait behind a prefetch's reads (or the other way round).
    parallel_rows(miss.size(), kReadThreads, [&](size_t m) {
        const size_t u = miss[m];
        thread_local std::vector<uint8_t> raw;
        raw.resize(rb);
        uint8_t *p = raw.data();
        uint64_t at = kNgramTableDataOffset + (uint64_t)uniq[u] * rb;
        size_t left = rb;
        while (left > 0) {
            const ssize_t got = ::pread(file_.fd(), p, left, (off_t)at);
            if (got < 0 && errno == EINTR) continue;
            STRIX_CHECK(got > 0, "NgramTableRows: read of row ", uniq[u], " from '", file_.path(), "' failed: ",
                        got < 0 ? std::strerror(errno) : "end of file");
            p += got, left -= (size_t)got, at += (uint64_t)got;
        }
        ngram_decode_row(info, raw.data(), buf + u * D);  // the cache and the forward see BF16
    });
    if (!miss.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t u = 0, k = 0; cap_ > 0 && k < miss.size(); ++k) {  // (cap_ under the lock: resize_cache)
            u = miss[k];
            auto it = where_.find(uniq[u]);  // another gather / prefetch may have read it meanwhile
            if (it != where_.end()) touch(it->second);
            else std::memcpy(data_.data() + (size_t)take_slot(uniq[u]) * D, buf + u * D, vb);
        }
    }
    return miss.size();
}

void NgramTableRows::gather(const int64_t *rows, int64_t n, uint16_t *out) const {
    STRIX_CHECK(out || n == 0, "NgramTableRows::gather: null out with n = ", n);
    const std::vector<int64_t> uniq = unique_rows(rows, n, "gather");
    const size_t D = (size_t)file_.info().row_dim, vb = D * 2;
    std::vector<uint16_t> buf(uniq.size() * D);
    const size_t reads = fill(uniq, buf.data());
    {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.requested += (uint64_t)n, stats_.unique += uniq.size(), stats_.hits += uniq.size() - reads;
        stats_.reads += reads;
    }
    for (int64_t i = 0; i < n; ++i) {
        const size_t u = (size_t)(std::lower_bound(uniq.begin(), uniq.end(), rows[i]) - uniq.begin());
        std::memcpy(out + i * (int64_t)D, buf.data() + u * D, vb);
    }
}

void NgramTableRows::prefetch(const int64_t *rows, int64_t n) const {
    const std::vector<int64_t> uniq = unique_rows(rows, n, "prefetch");
    if (cap_ == 0 || uniq.empty()) return;  // nowhere to keep them: the gather reads them as always
    std::vector<uint16_t> buf(uniq.size() * (size_t)file_.info().row_dim);
    const size_t reads = fill(uniq, buf.data());
    std::lock_guard<std::mutex> lock(mu_);
    stats_.prefetched += reads;
}

void NgramTableRows::clear_cache() const {
    std::lock_guard<std::mutex> lock(mu_);
    where_.clear();
    head_ = tail_ = -1, used_ = 0;
}

int64_t NgramTableRows::cache_rows() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cap_;
}

void NgramTableRows::resize_cache(int64_t rows) const {
    STRIX_CHECK(rows >= 0 && rows <= (1ll << 26), "NgramTableRows::resize_cache: rows = ", rows, ", expected 0..2^26");
    std::lock_guard<std::mutex> lock(mu_);
    // Swapped with fresh containers so the old memory is returned now (clear() / resize() keep the capacity).
    std::unordered_map<int64_t, int32_t>().swap(where_);
    std::vector<int64_t>((size_t)rows).swap(slot_row_);
    std::vector<int32_t>((size_t)rows).swap(prev_);
    std::vector<int32_t>((size_t)rows).swap(next_);
    std::vector<uint16_t>((size_t)(rows * file_.info().row_dim)).swap(data_);
    where_.reserve((size_t)rows);
    cap_ = rows;
    head_ = tail_ = -1, used_ = 0;
}

void NgramTableRows::drop_page_cache() const {
    const int rc = ::posix_fadvise(file_.fd(), 0, 0, POSIX_FADV_DONTNEED);  // returns the error number, not errno
    STRIX_CHECK(rc == 0, "NgramTableRows::drop_page_cache: posix_fadvise(DONTNEED) on '", file_.path(), "' failed: ",
                std::strerror(rc));
}

NgramTableRows::Stats NgramTableRows::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

}  // namespace strix
