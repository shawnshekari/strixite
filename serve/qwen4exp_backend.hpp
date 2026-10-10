#pragma once

// LmBackend on the real model: one Qwen4ExpSession (BF16 activations, WMMA prefill - the shipping settings) and one
// Qwen4ExpSnapshot slot.

#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"

namespace strix {

class Qwen4ExpBackend : public LmBackend {
public:
    // mtp_vocab: the draft's vocabulary (Qwen4ExpSession::set_mtp_vocab), 0 = the whole one.
    Qwen4ExpBackend(const Qwen4ExpModel &model, int64_t capacity, int64_t chunk, bool use_mtp = true,
                    int64_t mtp_vocab = 0);
    int64_t capacity() const override { return session_.capacity(); }
    int64_t max_chunk() const override { return session_.max_tokens(); }
    int64_t logits_row() const override { return model_.dims().vocab; }
    int64_t pos() const override { return session_.pos(); }
    uint64_t state_bytes_at(int64_t n) const { return session_.state_bytes(n); }  // a whole saved state at n tokens
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
    void keep_verify() override { session_.keep_verify(); }
    void drop_verify() override { session_.drop_verify(); }
    BackendStats backend_stats() const override;
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first) override { session_.prefetch_ple(ids, first); }
    void keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) override {
        (void)ids;  // the session saved its own
        session_.keep_verify_prefix(rows);
    }
    void save_snapshot(int slot) override { session_.save(snap(slot)); }
    int64_t snapshot_pos(int slot) const override { return snap(slot).pos(); }
    bool can_restore_snapshot(int slot) const override { return session_.can_restore(snap(slot)); }
    void restore_snapshot(int slot) override { session_.restore(snap(slot)); }
    void export_snapshot(int slot, HostBuffer &out, int64_t from = 0) override;
    uint64_t snapshot_bytes(int slot, int64_t from = 0) const override;
    void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) override;
    std::string state_fingerprint() const override;
    std::string describe() const override;

private:
    LogitRows candidate_rows(int64_t rows) const;  // session_.candidates() of the last forward, checked
    LogitRows rows_of(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                      const LogitMasks *masks, bool verify);
    const Qwen4ExpModel &model_;
    Qwen4ExpSession session_;
    Qwen4ExpSnapshot snapshots_[4];
    Qwen4ExpSnapshot &snap(int slot);
    const Qwen4ExpSnapshot &snap(int slot) const;
};

}  // namespace strix
