#include "serve/sampler.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

namespace strix {

Sampler::Sampler(const SamplingParams &p, int64_t n_valid) : p_(p), n_valid_(n_valid), rng_(p.seed) {
    STRIX_CHECK(std::isfinite(p.temperature) && p.temperature >= 0 && p.temperature <= 100,
                "Sampler: temperature ", p.temperature, ", expected 0..100");
    STRIX_CHECK(p.top_k >= 0, "Sampler: top_k ", p.top_k, ", expected >= 0 (0 = off)");
    STRIX_CHECK(std::isfinite(p.top_p) && p.top_p > 0 && p.top_p <= 1, "Sampler: top_p ", p.top_p, ", expected (0, 1]");
    STRIX_CHECK(n_valid >= 1, "Sampler: n_valid ", n_valid, ", expected >= 1");
}

int32_t Sampler::sample(const std::vector<float> &logits, float *out_margin) {
    return sample(logits.data(), logits.size(), out_margin);
}

LogitRows LogitRows::from_full(std::vector<float> logits, int64_t row) {
    STRIX_CHECK(row >= 1 && logits.size() % (size_t)row == 0, "LogitRows::from_full: ", logits.size(),
                " logits are not whole rows of ", row);
    LogitRows l;
    l.rows = (int64_t)logits.size() / row, l.row = row, l.full = std::move(logits);
    return l;
}

const uint32_t *LogitMasks::row(int64_t r) const {
    STRIX_CHECK(r >= 0 && r < rows && bits.size() == (size_t)(rows * words), "LogitMasks::row: row ", r, " of ", rows,
                " (", bits.size(), " words for ", rows, " rows of ", words, ")");
    return bits.data() + r * words;
}

void LogitMasks::check(int64_t want_rows, int64_t logits_row, const char *where) const {
    STRIX_CHECK(rows == want_rows, where, ": ", rows, " mask rows for ", want_rows, " logits rows");
    STRIX_CHECK(words * 32 >= logits_row, where, ": mask rows of ", words, " words cover ", words * 32,
                " ids, the logits row has ", logits_row);
    STRIX_CHECK(bits.size() == (size_t)(rows * words), where, ": ", bits.size(), " mask words for ", rows, " rows of ",
                words);
}

void apply_masks(LogitRows &l, const LogitMasks &masks) {
    STRIX_CHECK(l.cands == 0, "apply_masks: candidate rows (", l.cands, " a row) are masked by their backend, before "
                "they are reduced - only full rows here");
    masks.check(l.rows, l.row, "apply_masks");
    STRIX_CHECK(l.full.size() == (size_t)(l.rows * l.row), "apply_masks: ", l.full.size(), " floats for ", l.rows,
                " rows of ", l.row);
    for (int64_t r = 0; r < l.rows; ++r) {
        const uint32_t *m = masks.row(r);
        float *x = l.full.data() + r * l.row;
        for (int64_t i = 0; i < l.row; ++i)
            if (!((m[i >> 5] >> (i & 31)) & 1u) && x[i] == x[i]) x[i] = -std::numeric_limits<float>::infinity();
    }
}

LogitRows LogitRows::row_of(int64_t r) const {
    STRIX_CHECK(r >= 0 && r < rows, "LogitRows::row_of: row ", r, " of ", rows);
    LogitRows o;
    o.rows = 1, o.row = row, o.cands = cands;
    if (cands == 0) {
        STRIX_CHECK(full.size() == (size_t)(rows * row), "LogitRows::row_of: ", full.size(), " floats for ", rows,
                    " rows of ", row);
        o.full.assign(full.begin() + r * row, full.begin() + (r + 1) * row);
    } else {
        STRIX_CHECK(cand_v.size() == (size_t)(rows * cands) && cand_id.size() == cand_v.size() && nan.size() == (size_t)rows,
                    "LogitRows::row_of: candidate arrays ", cand_v.size(), " / ", cand_id.size(), " / ", nan.size(),
                    " for ", rows, " rows of ", cands);
        o.cand_v.assign(cand_v.begin() + r * cands, cand_v.begin() + (r + 1) * cands);
        o.cand_id.assign(cand_id.begin() + r * cands, cand_id.begin() + (r + 1) * cands);
        o.nan = {nan[(size_t)r]};
    }
    return o;
}

Top2 top2(const float *x, int64_t n) {
    STRIX_CHECK(x != nullptr && n >= 1, "top2: ", n, " logits at ", (const void *)x, ", expected >= 1");
    Top2 t;
    t.best_v = x[0], t.second_v = -std::numeric_limits<float>::infinity(), t.nan = x[0] != x[0];
    auto consider = [&](int64_t i) {
        const float v = x[i];
        t.nan |= v != v;
        if (v > t.best_v) t.second_v = t.best_v, t.best_v = v, t.best = (int32_t)i;
        else if (v > t.second_v) t.second_v = v;
    };
    constexpr int64_t kBlock = 64;
    int64_t i = 1;
    for (; i < std::min<int64_t>(n, kBlock); ++i) consider(i);
    for (; i + kBlock <= n; i += kBlock) {
        const float thr = t.second_v;
        int hit = 0;
        for (int64_t j = 0; j < kBlock; ++j) hit |= (x[i + j] > thr) | (x[i + j] != x[i + j]);
        if (hit)
            for (int64_t j = 0; j < kBlock; ++j) consider(i + j);
    }
    for (; i < n; ++i) consider(i);
    return t;
}

int32_t Sampler::sample(const float *logits, size_t size, float *out_margin) {
    STRIX_CHECK(logits != nullptr, "Sampler::sample: null logits");
    STRIX_CHECK((int64_t)size >= n_valid_, "Sampler::sample: ", size, " logits, expected >= ", n_valid_);
    const int64_t n = n_valid_;
    if (p_.temperature == 0) {
        const Top2 t = top2(logits, n);
        STRIX_CHECK(!t.nan, "Sampler::sample: NaN logit(s) in the row of ", n);
        if (out_margin) *out_margin = (n > 1) ? (t.best_v - t.second_v) : std::numeric_limits<float>::infinity();
        return t.best;
    }
    // Candidates, sorted by logit descending; ties keep the lower id first (as a full sort would).
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    const int64_t k = p_.top_k > 0 ? std::min<int64_t>(p_.top_k, n) : n;
    if (k == n) return sample_all(logits, out_margin);
    if (k > kSmallK) return sample_large_k(logits, k, out_margin);
    // Top-k in one pass over the row: the best max(k, 2) so far (2 for the margin), sorted. Once the list is full, a
    // logit enters only if it beats the last one - ids ascend, so an equal logit never does and the lower id stays
    // ahead. Replaces filling and partial-sorting all 248,320 candidates per row (bench_sampler).
    const size_t m = (size_t)std::min<int64_t>(n, std::max<int64_t>(k, 2));
    cand_.clear();
    bool nan = false;
    auto consider = [&](int64_t i) {
        const float v = logits[i];
        if (cand_.size() == m) {
            if (!(v > cand_.back().first)) {
                nan |= v != v;
                return;
            }
            cand_.pop_back();
        } else if (v != v) {
            nan = true;
            return;
        }
        const std::pair<float, int32_t> c{v, (int32_t)i};
        cand_.insert(std::upper_bound(cand_.begin(), cand_.end(), c, better), c);
    };
    // Blocks of 64: a branch-free (vectorized) test whether anything in the block could enter - above the last
    // candidate, or NaN - and the element loop only for those blocks. Past the first few blocks almost none can.
    constexpr int64_t kBlock = 64;
    int64_t i = 0;
    for (; i < std::min<int64_t>(n, (int64_t)m); ++i) consider(i);  // fill the list first
    for (; i + kBlock <= n; i += kBlock) {
        const float thr = cand_.back().first;
        const float *b = logits + i;
        int hit = 0;
        for (int64_t j = 0; j < kBlock; ++j) hit |= (b[j] > thr) | (b[j] != b[j]);
        if (hit)
            for (int64_t j = 0; j < kBlock; ++j) consider(i + j);
    }
    for (; i < n; ++i) consider(i);
    STRIX_CHECK(!nan, "Sampler::sample: NaN logit(s) in the row of ", n);
    if (out_margin) {
        *out_margin = (n > 1) ? (cand_[0].first - cand_[1].first) : std::numeric_limits<float>::infinity();
    }
    cand_.resize((size_t)k);
    return draw_from_candidates(k);
}

bool ban_tokens(LogitRows &l, int64_t r, const std::vector<int32_t> &ids) {
    STRIX_CHECK(r >= 0 && r < l.rows, "ban_tokens: row ", r, " of ", l.rows);
    STRIX_CHECK(!ids.empty(), "ban_tokens: no ids");
    const auto banned = [&](int32_t id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    if (l.cands == 0) {
        STRIX_CHECK(l.full.size() == (size_t)(l.rows * l.row), "ban_tokens: ", l.full.size(), " floats for ", l.rows,
                    " rows of ", l.row);
        float *x = l.full.data() + r * l.row;
        const int32_t top = (int32_t)(std::max_element(x, x + l.row) - x);  // NaN rows fail in the sampler
        for (const int32_t id : ids) {
            STRIX_CHECK(id >= 0 && id < l.row, "ban_tokens: id ", id, ", expected 0..", l.row - 1);
            x[id] = -std::numeric_limits<float>::infinity();
        }
        return banned(top);
    }
    STRIX_CHECK(l.cand_v.size() == (size_t)(l.rows * l.cands) && l.cand_id.size() == l.cand_v.size(),
                "ban_tokens: candidate arrays ", l.cand_v.size(), " / ", l.cand_id.size(), " for ", l.rows, " rows of ",
                l.cands);
    float *v = l.cand_v.data() + r * l.cands;
    int32_t *id = l.cand_id.data() + r * l.cands;
    const bool top = banned(id[0]);
    int64_t w = 0;
    for (int64_t i = 0; i < l.cands; ++i)
        if (!banned(id[i])) v[w] = v[i], id[w] = id[i], ++w;
    for (; w < l.cands; ++w) v[w] = -std::numeric_limits<float>::infinity(), id[w] = INT32_MAX;
    return top;
}

bool Sampler::takes_candidates() const {
    if (n_valid_ < kCandidates) return false;
    return p_.temperature == 0 || (p_.top_k >= 1 && p_.top_k <= kCandidates);
}

int32_t Sampler::sample(const LogitRows &l, int64_t r, float *out_margin) {
    STRIX_CHECK(r >= 0 && r < l.rows, "Sampler::sample: row ", r, " of ", l.rows);
    if (l.cands == 0) {
        STRIX_CHECK(l.full.size() == (size_t)(l.rows * l.row), "Sampler::sample: ", l.full.size(), " floats for ",
                    l.rows, " rows of ", l.row);
        return sample(l.full.data() + r * l.row, (size_t)l.row, out_margin);
    }
    STRIX_CHECK(l.cand_v.size() == (size_t)(l.rows * l.cands) && l.cand_id.size() == l.cand_v.size() &&
                    l.nan.size() == (size_t)l.rows,
                "Sampler::sample: candidate arrays ", l.cand_v.size(), " / ", l.cand_id.size(), " / ", l.nan.size(),
                " for ", l.rows, " rows of ", l.cands);
    return sample_candidates(l.cand_v.data() + r * l.cands, l.cand_id.data() + r * l.cands, l.cands, l.nan[(size_t)r] != 0,
                             out_margin);
}

int32_t Sampler::sample_candidates(const float *v, const int32_t *id, int64_t m, bool nan, float *out_margin) {
    STRIX_CHECK(v != nullptr && id != nullptr, "Sampler::sample_candidates: null candidates");
    STRIX_CHECK(takes_candidates(), "Sampler::sample_candidates: temperature ", p_.temperature, ", top_k ", p_.top_k,
                " need the whole row (candidates serve greedy or top_k 1..", kCandidates, ")");
    const int64_t n = n_valid_;
    STRIX_CHECK(m >= std::min<int64_t>(n, kCandidates), "Sampler::sample_candidates: ", m, " candidates, expected >= ",
                std::min<int64_t>(n, kCandidates));
    // The full row's checks, in the same order: NaN anywhere in the row, then the top logit (draw_from_candidates).
    STRIX_CHECK(!nan, "Sampler::sample: NaN logit(s) in the row of ", n);
    // Real candidates first, then (masked rows only) padding (-inf, INT32_MAX) to the end.
    bool padding = false;
    for (int64_t i = 0; i < std::min<int64_t>(m, kCandidates); ++i) {
        const bool pad = id[i] == INT32_MAX && v[i] == -std::numeric_limits<float>::infinity();
        STRIX_CHECK(pad || (!padding && id[i] >= 0 && id[i] < n), "Sampler::sample_candidates: candidate ", i,
                    " has id ", id[i], " (value ", v[i], "), expected 0..", n - 1,
                    padding ? " - a real candidate after padding" : "");
        padding |= pad;
    }
    STRIX_CHECK(id[0] != INT32_MAX, "Sampler::sample_candidates: the row has no candidate at all (every id masked)");
    if (p_.temperature == 0) {
        if (out_margin) *out_margin = (n > 1) ? (v[0] - v[1]) : std::numeric_limits<float>::infinity();
        return id[0];
    }
    const int64_t k = std::min<int64_t>(p_.top_k, n);
    const size_t keep = (size_t)std::min<int64_t>(n, std::max<int64_t>(k, 2));  // as sample(): the margin needs 2
    cand_.resize(keep);
    for (size_t i = 0; i < keep; ++i) cand_[i] = {v[i], id[i]};
    if (out_margin) *out_margin = (n > 1) ? (cand_[0].first - cand_[1].first) : std::numeric_limits<float>::infinity();
    cand_.resize((size_t)k);
    return draw_from_candidates(k);
}

// Softmax at the temperature over cand_[0, k) (sorted), top-p, then the draw.
int32_t Sampler::draw_from_candidates(int64_t k) {
    STRIX_CHECK(std::isfinite(cand_[0].first), "Sampler::sample: top logit is ", cand_[0].first);
    // Softmax at the temperature over the kept set.
    std::vector<double> prob((size_t)k);
    const double top = cand_[0].first / p_.temperature;
    double sum = 0;
    for (int64_t i = 0; i < k; ++i) {
        STRIX_CHECK(!std::isnan(cand_[(size_t)i].first), "Sampler::sample: NaN logit for token ", cand_[(size_t)i].second);
        prob[(size_t)i] = std::exp(cand_[(size_t)i].first / p_.temperature - top);
        sum += prob[(size_t)i];
    }
    // Top-p: the smallest prefix whose probability reaches top_p (at least one token).
    int64_t keep = k;
    if (p_.top_p < 1) {
        double acc = 0;
        for (int64_t i = 0; i < k; ++i) {
            acc += prob[(size_t)i] / sum;
            if (acc >= p_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double kept = 0;
    for (int64_t i = 0; i < keep; ++i) kept += prob[(size_t)i];
    const double r = std::uniform_real_distribution<double>(0, kept)(rng_);
    double acc = 0;
    for (int64_t i = 0; i < keep; ++i) {
        acc += prob[(size_t)i];
        if (r < acc) return cand_[(size_t)i].second;
    }
    return cand_[(size_t)(keep - 1)].second;
}

// A large top-k (above kSmallK, where the one-pass list's inserts would go quadratic): fill every candidate and
// partial-sort the best k - the sampler's original way. The softmax / top-p / draw is the same tail as sample().
int32_t Sampler::sample_large_k(const float *logits, int64_t k, float *out_margin) {
    const int64_t n = n_valid_;
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    cand_.resize((size_t)n);
    bool nan = false;
    for (int64_t i = 0; i < n; ++i) {
        nan |= logits[i] != logits[i];
        cand_[(size_t)i] = {logits[i], (int32_t)i};
    }
    STRIX_CHECK(!nan, "Sampler::sample: NaN logit(s) in the row of ", n);
    std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(), better);
    if (out_margin) *out_margin = cand_[0].first - cand_[1].first;  // k > kSmallK >= 2
    cand_.resize((size_t)k);
    return draw_from_candidates(k);
}

// top_k off: the softmax over the whole row, then only the part of the sorted order the draw needs - the top-p
// nucleus, then the prefix holding the drawn mass. sorted_prefix sorts the tokens above a probability threshold and
// lowers it until the prefix is covered: every token left out has a lower logit than every one taken, so the sorted
// set is exactly the head of the full sort. A peaked row sorts a handful instead of all 248,320 (bench_sampler:
// top_k0). Same distribution as sorting everything; only the order of summation (and so the last bits of the
// normalizer) differs.
int32_t Sampler::sample_all(const float *logits, float *out_margin) {
    const int64_t n = n_valid_;
    float top = -std::numeric_limits<float>::infinity(), second = top;
    bool nan = false;
    for (int64_t i = 0; i < n; ++i) {
        const float v = logits[i];
        nan |= v != v;
        if (v > top) second = top, top = v;
        else if (v > second) second = v;
    }
    STRIX_CHECK(!nan, "Sampler::sample: NaN logit(s) in the row of ", n);
    STRIX_CHECK(std::isfinite(top), "Sampler::sample: top logit is ", top);
    if (out_margin) *out_margin = (n > 1) ? (top - second) : std::numeric_limits<float>::infinity();
    probs_.resize((size_t)n);
    const double t_top = top / p_.temperature;
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += probs_[(size_t)i] = std::exp(logits[i] / p_.temperature - t_top);
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    // The shortest sorted prefix whose running sum (of f(prob)) makes done(acc) true; the whole row if none does.
    auto sorted_prefix = [&](const auto &f, const auto &done) -> size_t {
        for (double t = 0x1p-10;; t = t > 0x1p-80 ? t * 0x1p-10 : 0) {
            cand_.clear();
            for (int64_t i = 0; i < n; ++i)
                if (probs_[(size_t)i] >= t) cand_.push_back({logits[i], (int32_t)i});
            std::sort(cand_.begin(), cand_.end(), better);
            double acc = 0;
            for (size_t j = 0; j < cand_.size(); ++j) {
                acc += f(probs_[(size_t)cand_[j].second]);
                if (done(acc)) return j + 1;
            }
            if (t == 0) return cand_.size();
        }
    };
    double kept = sum;
    if (p_.top_p < 1) {
        // Top-p: the smallest prefix whose probability reaches top_p (at least one token), then draw within it.
        const size_t keep = sorted_prefix([&](double p) { return p / sum; }, [&](double acc) { return acc >= p_.top_p; });
        kept = 0;
        for (size_t j = 0; j < keep; ++j) kept += probs_[(size_t)cand_[j].second];
        const double r = std::uniform_real_distribution<double>(0, kept)(rng_);
        double acc = 0;
        for (size_t j = 0; j < keep; ++j) {
            acc += probs_[(size_t)cand_[j].second];
            if (r < acc) return cand_[j].second;
        }
        return cand_[keep - 1].second;
    }
    const double r = std::uniform_real_distribution<double>(0, kept)(rng_);
    const size_t at = sorted_prefix([](double p) { return p; }, [&](double acc) { return r < acc; });
    return cand_[at - 1].second;
}

}  // namespace strix
