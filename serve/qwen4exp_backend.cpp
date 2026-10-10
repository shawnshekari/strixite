#include "serve/qwen4exp_backend.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"
#include "runtime/ngram_table.hpp"

#include <cstdio>

namespace strix {

Qwen4ExpBackend::Qwen4ExpBackend(const Qwen4ExpModel &model, int64_t capacity, int64_t chunk, bool use_mtp,
                                 int64_t mtp_vocab)
    : model_(model), session_(model, capacity, chunk, PrefillMath::WmmaBf16, use_mtp && model.has_mtp()),
      snapshots_{session_.make_snapshot(), session_.make_snapshot(), session_.make_snapshot(),
                 session_.make_snapshot()} {
    STRIX_CHECK(mtp_vocab >= 0, "Qwen4ExpBackend: mtp_vocab ", mtp_vocab, ", expected >= 0 (0 = the whole vocabulary)");
    if (mtp_vocab > 0) session_.set_mtp_vocab(mtp_vocab);
}

Qwen4ExpSnapshot &Qwen4ExpBackend::snap(int slot) {
    STRIX_CHECK(slot == kTurnSlot || slot == kSystemSlot || slot == kDraftSlot || slot == kUserSlot,
                "Qwen4ExpBackend: snapshot slot ", slot);
    return snapshots_[slot];
}
const Qwen4ExpSnapshot &Qwen4ExpBackend::snap(int slot) const {
    STRIX_CHECK(slot == kTurnSlot || slot == kSystemSlot || slot == kDraftSlot || slot == kUserSlot,
                "Qwen4ExpBackend: snapshot slot ", slot);
    return snapshots_[slot];
}

BackendStats Qwen4ExpBackend::backend_stats() const {
    BackendStats s;
    const Qwen4ExpSession::PleStats p = session_.ple_stats();
    s.ple_gathers = p.gathers, s.ple_waits = p.waits;
    s.ple_gather_seconds = p.gather_seconds, s.ple_wait_seconds = p.wait_seconds;
    s.ple_wait_max_seconds = p.wait_max_seconds;
    s.ple_prefetches = p.prefetches, s.ple_prefetch_skipped = p.prefetch_skipped, s.ple_prefetch_seconds = p.prefetch_seconds;
    if (const auto *t = dynamic_cast<const NgramTableRows *>(&model_.ngram_rows())) {  // the checkpoint source keeps none
        const NgramTableRows::Stats n = t->stats();
        s.ngram_rows_requested = (int64_t)n.requested, s.ngram_rows_unique = (int64_t)n.unique;
        s.ngram_rows_cached = (int64_t)n.hits, s.ngram_rows_read = (int64_t)n.reads;
        s.ngram_rows_prefetched = (int64_t)n.prefetched;
    }
    return s;
}

void Qwen4ExpBackend::export_snapshot(int slot, HostBuffer &out, int64_t from) {
    const Qwen4ExpSnapshot &s = snap(slot);
    STRIX_CHECK(s.pos() >= 1, "Qwen4ExpBackend::export_snapshot: slot ", slot, " was never saved");
    out.resize(session_.state_bytes(s.pos(), from));
    session_.export_state(s, out.data(), out.size(), from);
}

uint64_t Qwen4ExpBackend::snapshot_bytes(int slot, int64_t from) const {
    const Qwen4ExpSnapshot &s = snap(slot);
    STRIX_CHECK(s.pos() >= 1, "Qwen4ExpBackend::snapshot_bytes: slot ", slot, " was never saved");
    return (uint64_t)session_.state_bytes(s.pos(), from);
}

void Qwen4ExpBackend::import_state(const HostBuffer &state, int64_t n, int64_t from) {
    session_.import_state(state.data(), state.size(), n, from);
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
    const int n = std::snprintf(buf, sizeof buf, "qwen4exp-state-v2 %s layout-%016llx %s L%lld%s%s",
                                f.meta("source_fingerprint").c_str(), (unsigned long long)strix_hash64(layout.data(), layout.size()),
                                kernels::act_name(model_.act()), (long long)model_.dims().layers,
                                session_.has_mtp() ? " mtp" : "", yarn);
    STRIX_CHECK(n > 0 && n < (int)sizeof buf, "Qwen4ExpBackend::state_fingerprint: ", n, " characters, expected < ", sizeof buf);
    return buf;
}

std::string Qwen4ExpBackend::describe() const {
    const StrixwFile &f = model_.weights().file();
    return "qwen4exp, " + std::to_string(model_.dims().layers) + " layers, layout " + f.meta("layout") + ", " +
           kernels::act_name(model_.act()) + ", capacity " + std::to_string(capacity()) + ", chunk " +
           std::to_string(max_chunk()) +
           (has_mtp() ? ", mtp (draft vocabulary " + std::to_string(session_.mtp_vocab()) + ")" : "") + [&] {
               const auto *t = dynamic_cast<const NgramTableRows *>(&model_.ngram_rows());
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

}  // namespace strix
