#pragma once

// Several conversations on one backend (feat/multi-sequence step 5): the engine runs one worker thread per slot,
// each with a SeqBackend - an LmBackend whose calls address that slot's seq of the shared backend - and a GpuArbiter
// that orders every call of every slot on the one GPU stream and batches the slots' MTP verifies into one forward.
//
// The arbiter keeps the shared backend's rules (Qwen4ExpSession, "batched forwards"):
//   - one call at a time (the session is single-threaded);
//   - a draft chain runs uninterrupted: once a seq drafts step 0 it owns the GPU until its verify or a forward
//     (another seq's draft would overwrite the head's shared streams mid-chain);
//   - a verify joins a batch: the batch runs when every decoding seq has its verify in (or batch_wait_locked() after
//     the first one came in - a seq busy elsewhere does not hold the others), packed into the backend's row budget;
//   - after a batched verify only its resolutions (keep / drop) and hints run until every member resolved;
//   - a call that fails midway leaves every seq's state suspect (LmBackend::failed): every seq's next call refuses
//     until it reset()s, and the first reset after the failure resets the whole backend.
// One slot (n_seqs() == 1): the engine uses the backend directly, no arbiter - single-conversation serving is
// unchanged.

#include "serve/engine.hpp"

#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace strix {

class GpuArbiter {
public:
    explicit GpuArbiter(LmBackend &be);
    LmBackend &backend() { return be_; }
    const LmBackend &backend() const { return be_; }

    enum class Kind {
        Call,     // a plain call (a forward, a snapshot, an export / import, a reset); ends the seq's draft chain
        Draft,    // a draft step: step 0 starts the seq's chain, the GPU stays the seq's until it verifies / forwards
        Resolve,  // keep / drop of the seq's run of the batched verify awaiting resolution
        Hint,     // prefetch / lookahead / reads: never blocked by a chain or a pending batch
    };
    // Runs f for seq under the GPU turn (see the rules above). Returns whether f ran.
    // A failed call poisons the shared backend until each seq resets it: Call / Draft / Resolve throw then (a real
    // call on suspect state must fail the request cleanly), a Hint - advisory reads and prefetches - skips f and
    // returns false: an advisory call must never be what takes a slot's worker down (serve/engine.cpp).
    bool run(int seq, Kind kind, const std::function<void()> &f);
    // seq's verify, joined into a batch with the other decoding seqs' (see above); returns seq's rows.
    LogitRows verify(int seq, const LmBackend::VerifyRun &run, bool cands, int64_t n_valid);
    // seq resets its state: after a failure the first one resets the whole backend; a pending resolution of seq's is
    // dropped first.
    void reset(int seq);
    // The seq runs a request's decode (it will verify soon) / is done with it (its verifies are not waited for).
    void set_decoding(int seq, bool on);
    // A call failed midway since seq's last reset (its state is suspect: reset before using it).
    bool stale(int seq) const;
    // Another seq than this one is running a request's decode (set_decoding): the engine then prefills in smaller
    // chunks (Engine::Options::shared_chunk) so those decode steps run between them.
    bool others_decoding(int seq) const;
    // How many seqs are running a request's decode (the engine sizes its draft chains so their verifies fit one batch).
    int decoding_count() const;

    // How long a waiting verify holds its batch for a decoding seq that hasn't sent its own yet: a tenth of the last
    // batch's time (that seq's host work between steps - sampling, parsing, streaming - is short next to a forward),
    // within [kBatchWaitMinSeconds, kBatchWaitMaxSeconds]. A fixed 2 ms let two slots fall out of step (each then
    // verified alone: 14 of 47 batches held both on a two-client run).
    static constexpr double kBatchWaitMinSeconds = 0.002, kBatchWaitMaxSeconds = 0.020;
    double batch_wait_locked() const;

    struct Stats {
        int64_t verify_batches = 0, verify_runs = 0;  // batched verify forwards and the runs they carried
        int64_t max_runs = 0;                         // the most runs one batch carried
        // Runs that went alone on a slot that can never share a forward, because its state carries mRoPE position
        // state (an image in the conversation): the batching this build refuses to do, counted rather than
        // inferred from batches and runs. What strix_verify_solo_mrope_total and the block's "batching" row read.
        int64_t verify_solo_mrope = 0;
    };
    Stats stats() const;

    // One conversation's last decode window (its set_decoding(true) .. (false)): how its verifies ran. A request's
    // block reads it once the window is closed.
    struct VerifyWindow {
        int64_t shared = 0;      // verifies that shared a forward with another slot's run
        int64_t solo_mrope = 0;  // verifies that ran alone because the slot cannot share one (mRoPE)
    };
    VerifyWindow verify_window(int seq) const;

private:
    struct Pending {
        LmBackend::VerifyRun run;
        bool cands = false;
        int64_t n_valid = 0;
        // Asked of the backend as the run comes in (LmBackend::batchable): false when the slot's state carries
        // mRoPE position state, which a multi-run forward refuses - this run then goes by itself.
        bool batchable = true;
        double since = 0;
        bool done = false;
        LogitRows rows;
        std::exception_ptr error;
    };
    void check_ok_locked(int seq) const;
    bool poisoned_locked(int seq) const;  // a failure since seq's last reset (check_ok_locked's condition, inverted)
    void fail_locked();  // a call failed with the backend failed(): every seq refuses until its reset
    bool can_run_locked(int seq, Kind kind) const;
    bool lead_ready_locked(double now) const;  // the pending verifies should run as a batch now
    void run_batch(std::unique_lock<std::mutex> &lock);

    LmBackend &be_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool busy_ = false;    // a call or a batch is on the GPU
    int owner_ = -1;       // the seq whose draft chain is running
    std::map<int, Pending> pending_;  // verifies waiting for their batch (and their results until taken)
    std::set<int> resolving_;         // seqs of the last batch that have not kept / dropped yet
    std::set<int> decoding_;
    bool failed_ = false;             // a failure since: every seq resets before its next call
    std::vector<uint64_t> epoch_;     // per seq: the failure count at its last reset
    uint64_t failures_ = 0;
    Stats stats_;
    double last_batch_seconds_ = 0;  // the last batched verify's forward time
    // Fairness between a prefill and the decoding seqs: a decoding seq resolves and starts its next chain on the same
    // thread before a waiting call's thread wakes, so a prefill's next chunk never got the GPU until every decode
    // ended. A seq's call waiting (calls_waiting_) blocks other seqs' new chains once a batch ran since that seq's
    // last call (last_call_batch_): chunk, decode step, chunk, ...
    std::multiset<int> calls_waiting_;
    // Seqs waiting for the GPU to start a draft chain: a batch waits for a decoding seq among them (its verify is a
    // chain away) instead of timing out on it.
    std::multiset<int> drafts_waiting_;
    std::vector<int64_t> last_call_batch_;  // per seq: stats_.verify_batches when its last call ran
    std::vector<VerifyWindow> window_;  // per seq: its last decode window, zeroed when the window opens
    std::vector<char> solo_announced_;  // per seq: the exclusion has been logged (its reset lets it be logged again)
};

// One slot's view of the shared backend: LmBackend's single-conversation calls, mapped to seq's.
class SeqBackend : public LmBackend {
public:
    SeqBackend(GpuArbiter &arb, int seq);
    int seq() const { return seq_; }
    // This slot last decode window, for the batching row of its request block.
    GpuArbiter::VerifyWindow verify_window() const { return arb_.verify_window(seq_); }

    int64_t capacity() const override { return be().capacity(); }
    int64_t max_chunk() const override { return be().max_chunk(); }
    int64_t logits_row() const override { return be().logits_row(); }
    int64_t pos() const override;
    void reset() override;
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override;
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits) override;
    void set_lookahead(const std::vector<int32_t> &next_ids) override;
    bool has_mtp() const override { return be().has_mtp(); }
    Top2 forward_mtp_top2(int32_t token_id, int64_t step) override;
    std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) override;
    LogitRows forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                           const LogitMasks *masks = nullptr) override;
    LogitRows forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                  const LogitMasks *masks = nullptr) override;
    bool has_vision() const override { return be().has_vision(); }
    int64_t max_positions() const override { return be().max_positions(); }
    LogitRows forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                    const std::vector<int64_t> &pos3, int64_t n_logits, bool cands, int64_t n_valid,
                                    const LogitMasks *masks = nullptr) override;
    BackendStats backend_stats() const override { return arb_.backend().backend_stats(); }
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first) override;
    void keep_verify() override;
    void drop_verify() override;
    void keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) override;
    void save_snapshot(int slot) override;
    int64_t snapshot_pos(int slot) const override;
    bool can_restore_snapshot(int slot) const override;
    void restore_snapshot(int slot) override;
    void export_snapshot(int slot, HostBuffer &out, int64_t from = 0) override;
    uint64_t snapshot_bytes(int slot, int64_t from = 0) const override;
    void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) override;
    std::string state_fingerprint() const override { return be().state_fingerprint(); }
    std::string describe() const override { return be().describe() + ", seq " + std::to_string(seq_); }

    // The engine marks a request's decode (GpuArbiter::set_decoding).
    void set_decoding(bool on) { arb_.set_decoding(seq_, on); }
    bool stale() const { return arb_.stale(seq_); }  // reset() before the next request (GpuArbiter::stale)

private:
    const LmBackend &be() const { return arb_.backend(); }
    GpuArbiter &arb_;
    const int seq_;
    int64_t verify_rows_ = 0;  // the pending verify's rows (keep_verify keeps them all)
};

}  // namespace strix
