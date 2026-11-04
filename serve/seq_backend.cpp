#include "serve/seq_backend.hpp"

#include "common/check.hpp"
#include "serve/log.hpp"  // the exclusion is announced in the request's log block

#include <algorithm>
#include <chrono>

namespace strix {

namespace {
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

GpuArbiter::GpuArbiter(LmBackend &be)
    : be_(be), epoch_((size_t)be.n_seqs(), 0), last_call_batch_((size_t)be.n_seqs(), 0),
      window_((size_t)be.n_seqs()), solo_announced_((size_t)be.n_seqs(), 0) {
    STRIX_CHECK(be.n_seqs() >= 1, "GpuArbiter: the backend holds ", be.n_seqs(), " conversations, expected >= 1");
}

void GpuArbiter::check_ok_locked(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter: seq ", seq, ", expected 0..", epoch_.size() - 1);
    STRIX_CHECK(!failed_ && epoch_[(size_t)seq] == failures_, "GpuArbiter: seq ", seq,
                ": a call on the shared GPU state failed (on this or another slot) since this slot's last reset - "
                "reset it first");
}

void GpuArbiter::fail_locked() {
    failed_ = true, ++failures_;
    owner_ = -1, resolving_.clear();
    for (auto &[s, p] : pending_)
        if (!p.done) p.done = true, p.error = std::make_exception_ptr(std::runtime_error(
                                         "GpuArbiter: the batched verify was abandoned - another slot's call failed"));
}

bool GpuArbiter::can_run_locked(int seq, Kind kind) const {
    if (busy_) return false;
    if (kind == Kind::Hint) return true;
    if (kind == Kind::Resolve) return resolving_.count(seq) > 0;
    if (!resolving_.empty()) return false;  // the batch's resolutions first (its shared workspaces)
    if (owner_ >= 0 && owner_ != seq) return false;  // another seq's chain runs uninterrupted
    if (owner_ == seq) return true;                  // its own chain (or the forward that ends it)
    // A new chain waits for another seq's call that has waited through a decode step (see calls_waiting_).
    if (kind == Kind::Draft)
        for (int s : calls_waiting_)
            if (s != seq && stats_.verify_batches > last_call_batch_[(size_t)s]) return false;
    // A batch ready to go runs before a new call (between a prefill's chunks the other slots' decode steps run).
    return !lead_ready_locked(now_s());
}

bool GpuArbiter::poisoned_locked(int seq) const {
    return failed_ || epoch_[(size_t)seq] != failures_;
}

bool GpuArbiter::run(int seq, Kind kind, const std::function<void()> &f) {
    std::unique_lock<std::mutex> lock(mu_);
    // A Hint is advisory (reads, prefetches): after a failed call there is nothing trustworthy left to read,
    // so it skips f and reports false instead of throwing - an advisory call must never be what takes a worker
    // down (the production abort: a post-request snapshot_pos Hint threw after another slot's clean failure).
    if (kind == Kind::Hint && poisoned_locked(seq)) return false;
    check_ok_locked(seq);
    STRIX_CHECK(seq >= 0 && seq < (int)last_call_batch_.size(), "GpuArbiter::run: seq ", seq, ", expected 0..",
                last_call_batch_.size() - 1);
    const bool call = kind == Kind::Call && owner_ != seq;  // a chain-ending forward isn't waiting for a turn
    const bool chain_start = kind == Kind::Draft && owner_ != seq;
    auto waiting = call ? calls_waiting_.insert(seq) : calls_waiting_.end();
    auto drafting = chain_start ? drafts_waiting_.insert(seq) : drafts_waiting_.end();
    cv_.wait(lock, [&] { return failed_ || can_run_locked(seq, kind); });
    if (call) {
        calls_waiting_.erase(waiting);
        last_call_batch_[(size_t)seq] = stats_.verify_batches;
    }
    if (chain_start) drafts_waiting_.erase(drafting);
    if (kind == Kind::Hint && poisoned_locked(seq)) return false;  // poisoned while waiting: same skip
    check_ok_locked(seq);
    STRIX_CHECK(kind != Kind::Resolve || resolving_.count(seq), "GpuArbiter: seq ", seq,
                " resolves a verify it has no pending run of");
    busy_ = true;
    if (kind == Kind::Draft) owner_ = seq;
    else if (kind == Kind::Call && owner_ == seq) owner_ = -1;  // a forward ends the chain (no draft passed)
    lock.unlock();
    try {
        f();
    } catch (...) {
        lock.lock();
        busy_ = false;
        if (owner_ == seq) owner_ = -1;
        if (be_.failed()) fail_locked();
        lock.unlock();
        cv_.notify_all();
        throw;
    }
    lock.lock();
    busy_ = false;
    if (kind == Kind::Resolve) resolving_.erase(seq);
    lock.unlock();
    cv_.notify_all();
    return true;  // f ran
}

bool GpuArbiter::lead_ready_locked(double now) const {
    if (busy_ || owner_ >= 0 || !resolving_.empty()) return false;
    double first = 0;
    int waiting = 0;
    for (const auto &[s, p] : pending_)
        if (!p.done) first = waiting++ ? std::min(first, p.since) : p.since;
    if (waiting == 0) return false;
    bool host_side = false;  // a decoding seq still to come that is busy on the host (sampling, parsing, streaming)
    for (int s : decoding_) {
        if (pending_.count(s) && !pending_.at(s).done) continue;
        // Waiting for the GPU to start its chain: it comes in a few ms after its chain - wait for it, no timeout. (The
        // timeout alone fired while the other slots' chains queued one after another: four slots averaged 2 runs a
        // batch, and the ready batch then held off the waiting chains.)
        if (drafts_waiting_.count(s)) return false;
        host_side = true;
    }
    return !host_side || now - first >= batch_wait_locked();  // every decoding seq is in, or the host-side wait ran out
}

double GpuArbiter::batch_wait_locked() const {
    return std::clamp(0.1 * last_batch_seconds_, kBatchWaitMinSeconds, kBatchWaitMaxSeconds);
}

void GpuArbiter::run_batch(std::unique_lock<std::mutex> &lock) {
    // Pack the waiting runs, oldest first, into the row budget; the rest wait for the next batch.
    std::vector<int> order;
    for (const auto &[s, p] : pending_)
        if (!p.done) order.push_back(s);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return pending_[a].since < pending_[b].since; });
    std::vector<int> members;
    std::vector<LmBackend::VerifyRun> runs;
    int64_t rows = 0;
    const int64_t budget = be_.max_verify_rows();
    bool cands = false;
    int64_t n_valid = 0;
    for (int s : order) {
        const Pending &p = pending_[s];
        const int64_t r = (int64_t)p.run.ids.size();
        if (!runs.empty() && (rows + r > budget || p.cands != cands || p.n_valid != n_valid)) continue;
        // A run whose slot carries mRoPE position state is not batchable at all: with another run in the forward
        // it throws (LmBackend::batchable, issue #29) and the throw takes every run of the batch down with it.
        // So it goes by itself: as the batch when it is the oldest waiting run, otherwise this pack stops in
        // front of it - the batch's runs go now, it goes alone right after, never behind newer arrivals.
        if (!p.batchable) {
            if (!runs.empty()) break;
            runs.push_back(p.run), members.push_back(s), rows += r;
            break;
        }
        if (runs.empty()) cands = p.cands, n_valid = p.n_valid;
        runs.push_back(p.run), members.push_back(s), rows += r;
    }
    busy_ = true;
    lock.unlock();
    std::vector<LogitRows> out;
    std::exception_ptr err;
    const double t0 = now_s();
    try {
        out = be_.verify_batch(runs, cands, n_valid);
        STRIX_CHECK(out.size() == runs.size(), "GpuArbiter: verify_batch returned ", out.size(), " results for ",
                    runs.size(), " runs");
    } catch (...) {
        err = std::current_exception();
    }
    lock.lock();
    busy_ = false;
    last_batch_seconds_ = now_s() - t0;
    if (err && be_.failed()) {
        for (int m : members) pending_[m].done = true, pending_[m].error = err;  // the batch's own error
        fail_locked();  // the rest: abandoned
    } else {
        for (size_t i = 0; i < members.size(); ++i) {
            Pending &p = pending_[members[i]];
            p.done = true;
            if (err) p.error = err;
            else p.rows = std::move(out[i]), resolving_.insert(members[i]);
        }
        if (!err) {
            ++stats_.verify_batches, stats_.verify_runs += (int64_t)members.size();
            stats_.max_runs = std::max(stats_.max_runs, (int64_t)members.size());
            // Who ran with whom, per run: shared is the batch's whole point, solo_mrope is the run that could not
            // have joined any batch at all. Both land in the slot's decode window (the request's block reads it).
            for (int m : members) {
                VerifyWindow &w = window_[(size_t)m];
                if (members.size() > 1) ++w.shared;
                else if (!pending_[m].batchable) ++w.solo_mrope, ++stats_.verify_solo_mrope;
            }
        }
    }
    cv_.notify_all();
}

LogitRows GpuArbiter::verify(int seq, const LmBackend::VerifyRun &run, bool cands, int64_t n_valid) {
    std::unique_lock<std::mutex> lock(mu_);
    check_ok_locked(seq);
    STRIX_CHECK(!pending_.count(seq) && !resolving_.count(seq), "GpuArbiter: seq ", seq,
                " verifies again before its last verify was resolved");
    STRIX_CHECK(run.seq == seq, "GpuArbiter: seq ", seq, " sends a run of seq ", run.seq);
    if (owner_ == seq) owner_ = -1;  // the chain is done
    Pending &me = pending_[seq];
    // Asked here, not while packing: the run's own thread still owns this slot's state (it has no run in a
    // batch yet - checked above), so reading it costs no GPU token.
    me.run = run, me.cands = cands, me.n_valid = n_valid, me.batchable = be_.batchable(seq), me.since = now_s();
    if (!me.batchable && !solo_announced_[(size_t)seq]) {  // said once per conversation, not once per step
        solo_announced_[(size_t)seq] = 1;
        slog(LogLevel::Info, "batching off: this conversation carries mRoPE position state (it ran an image, or "
                             "resumed one) - a multi-run forward takes scalar positions only, so its verifies run "
                             "alone (strix_verify_solo_mrope_total)");
    }
    cv_.notify_all();
    for (;;) {
        if (me.done) break;
        if (lead_ready_locked(now_s())) {
            run_batch(lock);
            continue;
        }
        cv_.wait_for(lock, std::chrono::duration<double>(kBatchWaitMinSeconds / 2));
    }
    Pending done = std::move(me);
    pending_.erase(seq);
    if (done.error) std::rethrow_exception(done.error);
    return std::move(done.rows);
}

void GpuArbiter::reset(int seq) {
    std::unique_lock<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::reset: seq ", seq, ", expected 0..",
                epoch_.size() - 1);
    cv_.wait(lock, [&] { return !busy_; });
    busy_ = true;
    const bool whole = failed_;
    const bool drop = !whole && resolving_.count(seq) > 0;  // a request failed between its verify and its resolution
    lock.unlock();
    std::exception_ptr err;
    try {
        if (whole) be_.reset();
        else {
            if (drop) be_.drop_verify(seq);
            be_.reset(seq);
        }
    } catch (...) {
        err = std::current_exception();
    }
    lock.lock();
    busy_ = false;
    if (!err) {
        if (whole) failed_ = false;  // the other seqs' epochs stay behind: each resets before its next call
        epoch_[(size_t)seq] = failures_;
        resolving_.erase(seq);
        solo_announced_[(size_t)seq] = 0;  // a reset clears the mRoPE state: the exclusion could apply again
        window_[(size_t)seq] = VerifyWindow{};
        if (owner_ == seq) owner_ = -1;
    } else if (be_.failed()) {
        fail_locked();
    }
    lock.unlock();
    cv_.notify_all();
    if (err) std::rethrow_exception(err);
}

GpuArbiter::VerifyWindow GpuArbiter::verify_window(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    return (size_t)seq < window_.size() ? window_[(size_t)seq] : VerifyWindow{};
}

void GpuArbiter::set_decoding(int seq, bool on) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        // The window opens with the request's decode: what its log block reports is what this decode did.
        if (on) decoding_.insert(seq), window_[(size_t)seq] = VerifyWindow{};
        else decoding_.erase(seq);
    }
    cv_.notify_all();
}

bool GpuArbiter::stale(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::stale: seq ", seq, ", expected 0..", epoch_.size() - 1);
    return failed_ || epoch_[(size_t)seq] != failures_;
}

int GpuArbiter::decoding_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)decoding_.size();
}

bool GpuArbiter::others_decoding(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::others_decoding: seq ", seq, ", expected 0..",
                epoch_.size() - 1);
    for (int s : decoding_)
        if (s != seq) return true;
    return false;
}

GpuArbiter::Stats GpuArbiter::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

// ---------------------------------------------------------------- SeqBackend

SeqBackend::SeqBackend(GpuArbiter &arb, int seq) : arb_(arb), seq_(seq) {
    STRIX_CHECK(seq >= 0 && seq < arb.backend().n_seqs(), "SeqBackend: seq ", seq, ", expected 0..",
                arb.backend().n_seqs() - 1);
}

int64_t SeqBackend::pos() const {
    int64_t p = 0;  // poisoned: 0 (its next Call fails the request anyway; nothing trusts positions mid-failure)
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { p = arb_.backend().pos(seq_); });
    return p;
}

void SeqBackend::reset() {
    arb_.reset(seq_);
    verify_rows_ = 0;
}

std::vector<float> SeqBackend::forward(const std::vector<int32_t> &ids, bool want_logits) {
    return forward(ids, (int64_t)(want_logits ? 1 : 0));
}

std::vector<float> SeqBackend::forward(const std::vector<int32_t> &ids, int64_t n_logits) {
    return forward_rows(ids, n_logits, false, 0).full;
}

void SeqBackend::set_lookahead(const std::vector<int32_t> &next_ids) {
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { arb_.backend().set_lookahead(next_ids); });
}

Top2 SeqBackend::forward_mtp_top2(int32_t token_id, int64_t step) {
    Top2 t;
    arb_.run(seq_, GpuArbiter::Kind::Draft, [&] { t = arb_.backend().forward_mtp_top2(seq_, token_id, step); });
    return t;
}

std::vector<float> SeqBackend::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) {
    return forward_verify_rows(ids, n_logits, false, 0).full;
}

LogitRows SeqBackend::forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                   const LogitMasks *masks) {
    LogitRows l;
    arb_.run(seq_, GpuArbiter::Kind::Call,
             [&] { l = arb_.backend().forward_rows(seq_, ids, n_logits, cands, n_valid, masks); });
    return l;
}

LogitRows SeqBackend::forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands,
                                          int64_t n_valid, const LogitMasks *masks) {
    STRIX_CHECK(n_logits == (int64_t)ids.size(), "SeqBackend::forward_verify_rows: ", n_logits, " logits rows for ",
                ids.size(), " ids - a batched verify returns every row");
    LogitRows l = arb_.verify(seq_, {seq_, ids, masks}, cands, n_valid);
    verify_rows_ = (int64_t)ids.size();
    return l;
}

LogitRows SeqBackend::forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                            const std::vector<int64_t> &pos3, int64_t n_logits, bool cands,
                                            int64_t n_valid, const LogitMasks *masks) {
    STRIX_CHECK(seq_ == 0, "SeqBackend::forward_image_prefill: an image prompt on slot ", seq_,
                ", expected slot 0 (image positions are slot 0's only; the engine routes image requests there)");
    LogitRows l;
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] {
        l = arb_.backend().forward_image_prefill(ids, image, pos3, n_logits, cands, n_valid, masks);
    });
    return l;
}

void SeqBackend::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { arb_.backend().prefetch_ple(seq_, ids, first); });
}

void SeqBackend::keep_verify() {
    STRIX_CHECK(verify_rows_ >= 1, "SeqBackend::keep_verify: slot ", seq_, " has no verify to keep");
    const int64_t rows = verify_rows_;
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().keep_verify(seq_, rows, -1); });
    verify_rows_ = 0;
}

void SeqBackend::drop_verify() {
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().drop_verify(seq_); });
    verify_rows_ = 0;
}

void SeqBackend::keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) {
    STRIX_CHECK(verify_rows_ >= 1 && (int64_t)ids.size() == verify_rows_ && rows >= 1 && rows < verify_rows_,
                "SeqBackend::keep_verify_prefix: keep ", rows, " of ", ids.size(), " ids, the pending verify has ",
                verify_rows_, " rows");
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().keep_verify(seq_, rows, -1); });
    verify_rows_ = 0;
}

void SeqBackend::save_snapshot(int slot) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().save_snapshot(seq_, slot); });
}

int64_t SeqBackend::snapshot_pos(int slot) const {
    int64_t p = -1;  // poisoned: -1 - no snapshot (a turn_pos of -1 = none; the engine never saves against it)
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { p = arb_.backend().snapshot_pos(seq_, slot); });
    return p;
}

bool SeqBackend::can_restore_snapshot(int slot) const {
    bool ok = false;  // poisoned: false - refuse the restore (the caller's next Call fails cleanly)
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { ok = arb_.backend().can_restore_snapshot(seq_, slot); });
    return ok;
}

void SeqBackend::restore_snapshot(int slot) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().restore_snapshot(seq_, slot); });
}

void SeqBackend::export_snapshot(int slot, HostBuffer &out, int64_t from) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().export_snapshot(seq_, slot, out, from); });
}

uint64_t SeqBackend::snapshot_bytes(int slot, int64_t from) const {
    uint64_t b = 0;  // poisoned: 0 (never reached: the save path's snapshot_pos / can_restore hints skip first)
    arb_.run(seq_, GpuArbiter::Kind::Hint, [&] { b = arb_.backend().snapshot_bytes(seq_, slot, from); });
    return b;
}

void SeqBackend::import_state(const HostBuffer &state, int64_t n, int64_t from) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().import_state(seq_, state, n, from); });
}

}  // namespace strix
