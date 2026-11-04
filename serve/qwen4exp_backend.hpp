#pragma once

// LmBackend on the real model: one Qwen4ExpSession (BF16 activations, WMMA prefill - the shipping settings) and one
// Qwen4ExpSnapshot slot.

#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"

namespace strix {

class Qwen4ExpBackend : public LmBackend {
public:
    // mtp_vocab: the draft's vocabulary (Qwen4ExpSession::set_mtp_vocab), 0 = the whole one.
    // precision: the kernel-step choices (runtime/qwen4exp.hpp PrecisionOptions; all exact by default).
    // kv: the attention layers' K/V cache format (runtime/qwen4exp.hpp KvFormat).
    // n_seqs: conversations held at once (Qwen4ExpSession's slots, each at the full capacity); 1 = today's serving.
    Qwen4ExpBackend(const Qwen4ExpModel &model, int64_t capacity, int64_t chunk, bool use_mtp = true,
                    int64_t mtp_vocab = 0, const PrecisionOptions &precision = {}, KvFormat kv = KvFormat::Act,
                    int64_t n_seqs = 1);
    int64_t capacity() const override { return session_.capacity(); }
    int64_t max_chunk() const override { return session_.max_tokens(); }
    int64_t logits_row() const override { return model_.dims().vocab; }
    int64_t pos() const override { return session_.pos(); }
    uint64_t state_bytes_at(int64_t n) const { return session_.state_bytes(n); }  // a whole saved state at n tokens
    // Paged slot K/V (Qwen4ExpSession::set_kv_budget / kv_full_bytes): the startup plan's deep-slot term and the
    // budget its growth asks first.
    uint64_t kv_full_bytes() const { return session_.kv_full_bytes(); }
    void set_kv_budget(std::function<void(uint64_t)> f) { session_.set_kv_budget(std::move(f)); }
    void reset() override { session_.reset(); }
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override {
        return session_.forward(ids, want_logits ? 1 : 0);
    }
    void set_lookahead(const std::vector<int32_t> &next_ids) override { session_.set_lookahead(next_ids); }
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits) override {
        return session_.forward(ids, n_logits);
    }
    bool has_mtp() const override { return session_.has_mtp(); }
    std::vector<float> forward_mtp(int32_t token_id, int64_t step) override {
        return session_.forward_mtp(token_id, step);
    }
    Top2 forward_mtp_top2(int32_t token_id, int64_t step) override {  // reduced on the GPU, one small copy back
        const Qwen4ExpSession::MtpTop2 p = session_.forward_mtp_top2(token_id, step);
        Top2 t;
        t.best = p.best, t.best_v = p.best_v, t.second_v = p.second_v, t.nan = p.nan;
        return t;
    }
    std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) override {
        return session_.forward_verify(ids, n_logits);
    }
    // Full rows are masked on the host (apply_masks); candidates on the GPU, in logits_topk's first level.
    LogitRows forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                           const LogitMasks *masks = nullptr) override {
        return rows_of(ids, n_logits, cands, n_valid, masks, false);
    }
    LogitRows forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                  const LogitMasks *masks = nullptr) override {
        return rows_of(ids, n_logits, cands, n_valid, masks, true);
    }
    // Vision (Phase D): the model carries the tower when it was built with a vision.strixw; the image
    // forward mirrors rows_of, with the block's triples set first (the session checks the continuation).
    bool has_vision() const override { return model_.has_vision(); }
    int64_t max_positions() const override { return model_.dims().max_positions(); }
    LogitRows forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                    const std::vector<int64_t> &pos3, int64_t n_logits, bool cands,
                                    int64_t n_valid, const LogitMasks *masks = nullptr) override;
    void keep_verify() override { session_.keep_verify(); }
    void drop_verify() override { session_.drop_verify(); }
    BackendStats backend_stats() const override;
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first) override { session_.prefetch_ple(ids, first); }
    void keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) override {
        (void)ids;  // the session saved its own
        session_.keep_verify_prefix(rows);
    }
    // Several conversations (LmBackend's seq calls): the session's slots, the batched verify in one forward.
    int n_seqs() const override { return (int)session_.n_slots(); }
    int64_t pos(int seq) const override { return session_.pos(seq); }
    LogitRows forward_rows(int seq, const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                           const LogitMasks *masks = nullptr) override;
    Top2 forward_mtp_top2(int seq, int32_t token_id, int64_t step) override {
        const Qwen4ExpSession::MtpTop2 p = session_.forward_mtp_top2_seq(seq, token_id, step);
        Top2 t;
        t.best = p.best, t.best_v = p.best_v, t.second_v = p.second_v, t.nan = p.nan;
        return t;
    }
    void prefetch_ple(int seq, const std::vector<int32_t> &ids, int64_t first) override {
        session_.prefetch_ple(seq, ids, first);
    }
    std::vector<LogitRows> verify_batch(const std::vector<VerifyRun> &runs, bool cands, int64_t n_valid) override;
    int64_t max_verify_rows() const override { return Qwen4ExpSession::kMaxLogits; }
    // A conversation that has run an image carries mRoPE position state for good (issue #29): its verifies run
    // alone. Safe to ask off the GPU token - a slot's position state is written only by that slot's forwards,
    // and this one has no run in a batch yet (GpuArbiter::verify checks that before asking).
    bool batchable(int seq) const override { return session_.batchable(seq); }
    // carry -1 (none) is the session's 0; a carried token id 0 is then unchecked (Qwen4ExpSession::keep_verify).
    void keep_verify(int seq, int64_t rows, int32_t carry) override { session_.keep_verify(seq, rows, carry < 0 ? 0 : carry); }
    void drop_verify(int seq) override { session_.drop_verify(seq); }
    void save_snapshot(int slot) override { save_snapshot(0, slot); }
    int64_t snapshot_pos(int slot) const override { return snapshot_pos(0, slot); }
    bool can_restore_snapshot(int slot) const override { return can_restore_snapshot(0, slot); }
    void restore_snapshot(int slot) override { restore_snapshot(0, slot); }
    void export_snapshot(int slot, HostBuffer &out, int64_t from = 0) override { export_snapshot(0, slot, out, from); }
    uint64_t snapshot_bytes(int slot, int64_t from = 0) const override { return snapshot_bytes(0, slot, from); }
    void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) override { import_state(0, state, n, from); }
    // Per conversation: its own state and snapshot slots.
    void reset(int seq) override { session_.reset(seq); }
    bool failed() const override { return session_.broken(); }
    void save_snapshot(int seq, int slot) override { session_.save(seq, snap(seq, slot)); }
    int64_t snapshot_pos(int seq, int slot) const override { return snap(seq, slot).pos(); }
    bool can_restore_snapshot(int seq, int slot) const override { return session_.can_restore(seq, snap(seq, slot)); }
    void restore_snapshot(int seq, int slot) override { session_.restore(seq, snap(seq, slot)); }
    void export_snapshot(int seq, int slot, HostBuffer &out, int64_t from = 0) override;
    uint64_t snapshot_bytes(int seq, int slot, int64_t from = 0) const override;
    void import_state(int seq, const HostBuffer &state, int64_t n, int64_t from = 0) override;
    std::string state_fingerprint() const override;
    std::string describe() const override;

private:
    LogitRows candidate_rows(int64_t rows) const;  // session_.candidates() of the last forward, checked
    // A batch's per-run masks as one [rows, words] set for want_candidates (runs without masks: every id allowed);
    // empty when no run has any.
    LogitMasks batch_masks(const std::vector<VerifyRun> &runs) const;
    LogitRows rows_of(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                      const LogitMasks *masks, bool verify);
    const Qwen4ExpModel &model_;
    Qwen4ExpSession session_;
    std::vector<std::vector<Qwen4ExpSnapshot>> snapshots_;  // [seq][snapshot slot]
    Qwen4ExpSnapshot &snap(int seq, int slot);
    const Qwen4ExpSnapshot &snap(int seq, int slot) const;
};

}  // namespace strix
