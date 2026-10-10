#pragma once

// Next-token sampling on the host from one row of FP32 logits, in the
// order transformers' generate applies its warpers: temperature, then top-k, then top-p (nucleus), then a draw
// from the renormalized distribution. temperature 0 = greedy (argmax, lowest id on a tie). Only ids below
// `n_valid` can come out: the LM head is padded past the tokenizer's vocabulary (248320 rows, 248077 tokens), and a
// padding row must never be sampled.

#include <cstdint>
#include <random>
#include <vector>

namespace strix {

struct SamplingParams {
    double temperature = 1.0;  // >= 0
    int64_t top_k = 20;        // 0 = off
    double top_p = 0.95;       // (0, 1]
    uint64_t seed = 0;
};

// The largest logit (lowest id on a tie) and the runner-up's value (-inf if n == 1; equal to the best on a tie) over
// x[0, n), and whether any of it is NaN. Skips 64-logit blocks with nothing above the runner-up, branch-free.
struct Top2 {
    int32_t best = 0;
    float best_v = 0, second_v = 0;
    bool nan = false;
};
Top2 top2(const float *x, int64_t n);

// A forward's logits as the sampler takes them: `rows` rows of full logits (`row` floats each), or each row's top
// `cands` candidates - value descending, the lower id first on a tie (kernels/logits_topk) - with a per-row NaN flag.
struct LogitRows {
    int64_t rows = 0, row = 0;        // full rows: row = floats per row
    std::vector<float> full;          // [rows, row]; empty for candidates
    int64_t cands = 0;                // candidates per row; 0 = full rows
    std::vector<float> cand_v;        // [rows, cands]
    std::vector<int32_t> cand_id;     // [rows, cands]
    std::vector<uint8_t> nan;         // [rows]
    bool empty() const { return rows == 0; }
    static LogitRows from_full(std::vector<float> logits, int64_t row);  // logits.size() a multiple of row
    LogitRows row_of(int64_t r) const;                                  // one row, either kind
};

// Per-row allowed-token masks for a forward's logits rows (structured output, response_format):
// bit (id % 32) of word (id / 32) of row r set = id allowed. A backend gives back only allowed ids - candidates from
// the allowed ids alone (kernels/logits_topk's mask), full rows with every disallowed logit set to -inf.
struct LogitMasks {
    int64_t rows = 0, words = 0;  // words per row: >= ceil(logits row / 32)
    std::vector<uint32_t> bits;   // [rows, words]
    bool empty() const { return rows == 0; }
    const uint32_t *row(int64_t r) const;  // checked
    void check(int64_t want_rows, int64_t logits_row, const char *where) const;  // rows and width fit the forward
};
// Full rows: every logit whose id the row's mask disallows becomes -inf (candidate rows: refused - their backend
// applies the mask before reducing).
void apply_masks(LogitRows &rows, const LogitMasks &masks);
// Takes ids out of row r after the forward (the think-block guard, Engine::Options::think_end_guard): full rows get
// -inf there; candidate rows drop them and end in padding (-inf, INT32_MAX) - top-k then draws from one fewer
// candidate per removed id (the row's 21st-best is not there to move up; after top-p 0.95 its share is nearly always
// cut anyway). Returns whether one of ids was the row's top token (it would most likely have been sampled).
bool ban_tokens(LogitRows &rows, int64_t r, const std::vector<int32_t> &ids);

class Sampler {
public:
    static constexpr int64_t kCandidates = 20;  // what sample_candidates needs a row (kernels::kLogitCands)

    Sampler(const SamplingParams &p, int64_t n_valid);  // throws on out-of-range params
    int32_t sample(const std::vector<float> &logits, float *out_margin = nullptr);
    int32_t sample(const float *logits, size_t size, float *out_margin = nullptr);
    // Row r of either kind.
    int32_t sample(const LogitRows &logits, int64_t r, float *out_margin = nullptr);
    // Whether a row's top kCandidates are enough for these params: greedy, or top_k in 1..kCandidates.
    bool takes_candidates() const;
    // sample() from a row's top candidates (v / id, m of them, sorted as in LogitRows; m >= kCandidates or every
    // valid id) and whether the row had a NaN: the same result, margin, errors and random draws as sample() on the
    // whole row. Requires takes_candidates(). A masked row (structured output) may end in padding entries
    // (-inf, INT32_MAX) when fewer ids are allowed than candidates: they carry no probability, as the -inf logits of
    // the same row masked in full do.
    int32_t sample_candidates(const float *v, const int32_t *id, int64_t m, bool nan, float *out_margin = nullptr);

private:
    SamplingParams p_;
    int64_t n_valid_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int32_t>> cand_;
    std::vector<double> probs_;
    static constexpr int64_t kSmallK = 64;  // top-k up to this: the one-pass sorted list
    int32_t sample_all(const float *logits, float *out_margin);  // top_k off
    int32_t sample_large_k(const float *logits, int64_t k, float *out_margin);
    int32_t draw_from_candidates(int64_t k);
};

}  // namespace strix
