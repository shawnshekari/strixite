#include "serve/qwen4exp_backend.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"
#include "runtime/ngram_table.hpp"

#include <algorithm>
#include <cstdio>

namespace strix {

Qwen4ExpBackend::Qwen4ExpBackend(const Qwen4ExpModel &model, int64_t capacity, int64_t chunk, bool use_mtp,
                                 int64_t mtp_vocab, const PrecisionOptions &precision, KvFormat kv, int64_t n_seqs)
    : model_(model),
      session_(model, capacity, chunk, PrefillMath::WmmaBf16, use_mtp && model.has_mtp(), kv, n_seqs) {
    for (int k = 0; k < (int)n_seqs; ++k) {  // each conversation's own snapshot slots (a snapshot belongs to one state)
        snapshots_.emplace_back();
        for (int i = 0; i < 4; ++i) snapshots_.back().push_back(session_.make_snapshot(k));
    }
    STRIX_CHECK(mtp_vocab >= 0, "Qwen4ExpBackend: mtp_vocab ", mtp_vocab, ", expected >= 0 (0 = the whole vocabulary)");
    session_.set_precision(precision);
    if (mtp_vocab > 0) session_.set_mtp_vocab(mtp_vocab);
}

Qwen4ExpSnapshot &Qwen4ExpBackend::snap(int seq, int slot) {
    check_seq(seq, "snapshot");
    STRIX_CHECK(slot == kTurnSlot || slot == kSystemSlot || slot == kDraftSlot || slot == kUserSlot,
                "Qwen4ExpBackend: snapshot slot ", slot, " of seq ", seq);
    return snapshots_[(size_t)seq][(size_t)slot];
}
const Qwen4ExpSnapshot &Qwen4ExpBackend::snap(int seq, int slot) const {
    check_seq(seq, "snapshot");
    STRIX_CHECK(slot == kTurnSlot || slot == kSystemSlot || slot == kDraftSlot || slot == kUserSlot,
                "Qwen4ExpBackend: snapshot slot ", slot, " of seq ", seq);
    return snapshots_[(size_t)seq][(size_t)slot];
}

const NgramTableRows *Qwen4ExpBackend::ngram_table() const {
    if (!model_.has_ngram_rows()) return nullptr;
    return dynamic_cast<const NgramTableRows *>(&model_.ngram_rows());
}

BackendStats Qwen4ExpBackend::backend_stats() const {
    BackendStats s;
    const Qwen4ExpSession::PleStats p = session_.ple_stats();
    s.ple_gathers = p.gathers, s.ple_waits = p.waits;
    s.ple_gather_seconds = p.gather_seconds, s.ple_wait_seconds = p.wait_seconds;
    s.ple_wait_max_seconds = p.wait_max_seconds;
    s.ple_prefetches = p.prefetches, s.ple_prefetch_skipped = p.prefetch_skipped, s.ple_prefetch_seconds = p.prefetch_seconds;
    if (const NgramTableRows *t = ngram_table()) {
        const NgramTableRows::Stats n = t->stats();
        s.ngram_rows_requested = (int64_t)n.requested, s.ngram_rows_unique = (int64_t)n.unique;
        s.ngram_rows_cached = (int64_t)n.hits, s.ngram_rows_read = (int64_t)n.reads;
        s.ngram_rows_prefetched = (int64_t)n.prefetched;
    }
    const Qwen4ExpSession::VisionStats v = session_.vision_stats();
    s.vision_images = v.images, s.vision_patches = v.patches, s.vision_merged_tokens = v.merged_tokens;
    s.vision_tower_seconds = v.tower_seconds, s.vision_tower_max_seconds = v.tower_max_seconds;
    return s;
}

void Qwen4ExpBackend::export_snapshot(int seq, int slot, HostBuffer &out, int64_t from) {
    const Qwen4ExpSnapshot &s = snap(seq, slot);
    STRIX_CHECK(s.pos() >= 1, "Qwen4ExpBackend::export_snapshot: slot ", slot, " of seq ", seq, " was never saved");
    out.resize(session_.state_bytes(s.pos(), from));
    session_.export_state(seq, s, out.data(), out.size(), from);
}

uint64_t Qwen4ExpBackend::snapshot_bytes(int seq, int slot, int64_t from) const {
    const Qwen4ExpSnapshot &s = snap(seq, slot);
    STRIX_CHECK(s.pos() >= 1, "Qwen4ExpBackend::snapshot_bytes: slot ", slot, " of seq ", seq, " was never saved");
    return (uint64_t)session_.state_bytes(s.pos(), from);
}

void Qwen4ExpBackend::import_state(int seq, const HostBuffer &state, int64_t n, int64_t from) {
    session_.import_state(seq, state.data(), state.size(), n, from);
}

std::string Qwen4ExpBackend::state_fingerprint() const {
    const StrixwFile &f = model_.weights().file();
    // The layout string is long; its hash keeps the fingerprint short (the header holds < 192 chars).
    const std::string &layout = f.meta("layout");
    char buf[160];
    // v2: the MTP head's state follows the trunk's when MTP is on ("mtp" - a state from a session without it lacks it).
    // YaRN: the K caches and indexer block keys hold keys rotated with the factor's table, so a state is only valid
    // under the same factor (" yarn<f>"; nothing without YaRN, so the plain entries stay valid).
    char yarn[32] = "";
    if (model_.dims().yarn_factor > 1.0f) std::snprintf(yarn, sizeof yarn, " yarn%g", (double)model_.dims().yarn_factor);
    // Precision: a state built with a non-exact step is not the one the exact math builds (" gdn=fast"; nothing when
    // all exact, so the existing entries stay valid).
    const std::string prec = session_.precision().exact() ? std::string() : " " + session_.precision().describe();
    // 8-bit K/V (" kv8"): another state layout - a BF16 entry is dropped at load, never misread (nothing for the
    // activation dtype's K/V, so those entries stay valid).
    const int n = std::snprintf(buf, sizeof buf, "qwen4exp-state-v2 %s layout-%016llx %s L%lld%s%s%s%s",
                                f.meta("source_fingerprint").c_str(), (unsigned long long)strix_hash64(layout.data(), layout.size()),
                                kernels::act_name(model_.act()), (long long)model_.dims().layers,
                                session_.has_mtp() ? " mtp" : "", yarn, prec.c_str(),
                                session_.kv_format() == KvFormat::Q8 ? " kv8" : "");
    STRIX_CHECK(n > 0 && n < (int)sizeof buf, "Qwen4ExpBackend::state_fingerprint: ", n, " characters, expected < ", sizeof buf);
    return buf;
}

std::string Qwen4ExpBackend::describe() const {
    const StrixwFile &f = model_.weights().file();
    return "qwen4exp, " + std::to_string(model_.dims().layers) + " layers, layout " + f.meta("layout") + ", " +
           kernels::act_name(model_.act()) + ", capacity " + std::to_string(capacity()) + ", chunk " +
           std::to_string(max_chunk()) + ", precision " + session_.precision().describe() + ", kv " +
           kv_format_name(session_.kv_format()) +
           (has_mtp() ? ", mtp (draft vocabulary " + std::to_string(session_.mtp_vocab()) + ")" : "") +
           (model_.has_vision() ? ", vision tower on" : "") + [&] {
               const NgramTableRows *t = ngram_table();
               return t ? ", n-gram row cache " + std::to_string(t->cache_rows()) + " rows" : std::string();
           }();
}

LogitRows Qwen4ExpBackend::rows_of(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                   const LogitMasks *masks, bool verify) {
    if (masks != nullptr) masks->check(n_logits, logits_row(), "Qwen4ExpBackend::forward_rows");
    if (!cands || n_logits == 0) {
        LogitRows l = LogitRows::from_full(verify ? session_.forward_verify(ids, n_logits) : session_.forward(ids, n_logits),
                                           logits_row());
        if (masks != nullptr) apply_masks(l, *masks);
        return l;
    }
    if (masks != nullptr) session_.want_candidates(n_valid, masks->bits.data(), masks->rows, masks->words);
    else session_.want_candidates(n_valid);
    if (verify) session_.forward_verify(ids, n_logits);
    else session_.forward(ids, n_logits);
    return candidate_rows(n_logits);
}

LogitRows Qwen4ExpBackend::forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                                 const std::vector<int64_t> &pos3, int64_t n_logits, bool cands,
                                                 int64_t n_valid, const LogitMasks *masks) {
    STRIX_CHECK(model_.has_vision(), "Qwen4ExpBackend::forward_image_prefill: the model was loaded without "
                "a vision.strixw - image prompts are refused (config vision)");
    if (masks != nullptr)
        masks->check(n_logits, logits_row(), "Qwen4ExpBackend::forward_image_prefill");
    STRIX_CHECK(!image.prefill.pixel_values.empty(), "Qwen4ExpBackend::forward_image_prefill: the image block at row ",
                image.row_start, " has no pixel_values - the engine decodes a known image before its forward");
    session_.set_positions(pos3);  // the walk's triples for exactly this forward's rows (Phase A checks)
    ImageInput in;
    in.pixel_values = image.prefill.pixel_values;  // one copy per image forward (the block runs once)
    in.grid_t = image.prefill.grid_t, in.grid_h = image.prefill.grid_h, in.grid_w = image.prefill.grid_w;
    if (!cands || n_logits == 0) {
        LogitRows l = LogitRows::from_full(session_.forward(ids, n_logits, &in), logits_row());
        if (masks != nullptr) apply_masks(l, *masks);
        return l;
    }
    if (masks != nullptr) session_.want_candidates(n_valid, masks->bits.data(), masks->rows, masks->words);
    else session_.want_candidates(n_valid);
    session_.forward(ids, n_logits, &in);
    return candidate_rows(n_logits);
}

LogitRows Qwen4ExpBackend::candidate_rows(int64_t rows) const {
    const Qwen4ExpSession::Candidates &c = session_.candidates();
    constexpr int64_t K = kernels::kLogitCands;
    static_assert(K == Sampler::kCandidates, "the GPU's candidates per row are what the sampler takes");
    STRIX_CHECK(c.rows == rows && c.cand.size() == (size_t)(rows * K) && c.nan.size() == (size_t)rows,
                "Qwen4ExpBackend: the forward returned ", c.rows, " candidate rows (", c.cand.size(), " candidates, ",
                c.nan.size(), " NaN flags), expected ", rows, " rows of ", K);
    LogitRows l;
    l.rows = rows, l.row = logits_row(), l.cands = K;
    l.cand_v.resize(c.cand.size()), l.cand_id.resize(c.cand.size()), l.nan.resize((size_t)rows);
    for (size_t i = 0; i < c.cand.size(); ++i) l.cand_v[i] = c.cand[i].v, l.cand_id[i] = c.cand[i].id;
    for (size_t r = 0; r < (size_t)rows; ++r) l.nan[r] = c.nan[r] != 0;
    return l;
}

LogitRows Qwen4ExpBackend::forward_rows(int seq, const std::vector<int32_t> &ids, int64_t n_logits, bool cands,
                                        int64_t n_valid, const LogitMasks *masks) {
    if (seq == 0) return forward_rows(ids, n_logits, cands, n_valid, masks);  // the one-seq path, unchanged
    check_seq(seq, "forward_rows");
    if (masks != nullptr) masks->check(n_logits, logits_row(), "Qwen4ExpBackend::forward_rows");
    std::vector<Qwen4ExpSession::BatchRun> run{{seq, ids, n_logits}};
    if (!cands || n_logits == 0) {
        LogitRows l = LogitRows::from_full(session_.forward_batch(run), logits_row());
        if (masks != nullptr) apply_masks(l, *masks);
        return l;
    }
    if (masks != nullptr) session_.want_candidates(n_valid, masks->bits.data(), masks->rows, masks->words);
    else session_.want_candidates(n_valid);
    session_.forward_batch(run);
    return candidate_rows(n_logits);
}

LogitMasks Qwen4ExpBackend::batch_masks(const std::vector<VerifyRun> &runs) const {
    LogitMasks m;
    for (const VerifyRun &r : runs)
        if (r.masks != nullptr) {
            r.masks->check((int64_t)r.ids.size(), logits_row(), "Qwen4ExpBackend::verify_batch");
            m.words = std::max(m.words, r.masks->words);
        }
    if (m.words == 0) return m;
    for (const VerifyRun &r : runs) m.rows += (int64_t)r.ids.size();
    m.bits.assign((size_t)(m.rows * m.words), ~0u);  // runs without masks: every id allowed
    int64_t at = 0;
    for (const VerifyRun &r : runs) {
        if (r.masks != nullptr)
            for (int64_t i = 0; i < (int64_t)r.ids.size(); ++i)
                std::copy(r.masks->row(i), r.masks->row(i) + r.masks->words, m.bits.begin() + (size_t)((at + i) * m.words));
        at += (int64_t)r.ids.size();
    }
    return m;
}

std::vector<LogitRows> Qwen4ExpBackend::verify_batch(const std::vector<VerifyRun> &runs, bool cands, int64_t n_valid) {
    STRIX_CHECK(!runs.empty() && (int64_t)runs.size() <= n_seqs(), "Qwen4ExpBackend::verify_batch: ", runs.size(),
                " runs, expected 1..", n_seqs());
    std::vector<Qwen4ExpSession::BatchRun> batch;
    int64_t rows = 0;
    for (const VerifyRun &r : runs) {
        check_seq(r.seq, "verify_batch");
        batch.push_back({r.seq, r.ids, (int64_t)r.ids.size()});
        rows += (int64_t)r.ids.size();
    }
    const LogitMasks masks = batch_masks(runs);
    std::vector<LogitRows> out;
    if (!cands) {
        const LogitRows all = LogitRows::from_full(session_.forward_batch_verify(batch), logits_row());
        STRIX_CHECK(all.rows == rows, "Qwen4ExpBackend::verify_batch: ", all.rows, " logits rows, expected ", rows);
        int64_t at = 0;
        for (const VerifyRun &r : runs) {
            const int64_t n = (int64_t)r.ids.size();
            LogitRows l;
            l.rows = n, l.row = all.row;
            l.full.assign(all.full.begin() + at * all.row, all.full.begin() + (at + n) * all.row);
            if (r.masks != nullptr) apply_masks(l, *r.masks);
            out.push_back(std::move(l));
            at += n;
        }
        return out;
    }
    if (!masks.empty()) session_.want_candidates(n_valid, masks.bits.data(), masks.rows, masks.words);
    else session_.want_candidates(n_valid);
    session_.forward_batch_verify(batch);
    const LogitRows all = candidate_rows(rows);
    int64_t at = 0;
    for (const VerifyRun &r : runs) {
        const int64_t n = (int64_t)r.ids.size(), K = all.cands;
        LogitRows l;
        l.rows = n, l.row = all.row, l.cands = K;
        l.cand_v.assign(all.cand_v.begin() + at * K, all.cand_v.begin() + (at + n) * K);
        l.cand_id.assign(all.cand_id.begin() + at * K, all.cand_id.begin() + (at + n) * K);
        l.nan.assign(all.nan.begin() + at, all.nan.begin() + at + n);
        out.push_back(std::move(l));
        at += n;
    }
    return out;
}

}  // namespace strix
