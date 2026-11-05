#pragma once

// A request's logit_bias (OpenAI's request field) added to its logits rows on the GPU, in place, ahead of logits_topk:
// the candidates then come from the biased rows, so a +100 on a token far outside a row's top 20 still reaches the
// sampler, and a -100 leaves its token out the way the full row would (serve/sampler.hpp, LogitBias). Full rows on
// the host get the same addition from apply_masks - this kernel is only the candidates path.
//
// An entry adds v to logits[row * ld + id] for every row in [row0, row0 + rows): a request's bias is the same on each
// of its rows (a verify's token rows, a batched verify's run), so one entry covers them. One thread per entry; the
// caller guarantees no two entries touch the same logit (ids unique within a run, runs' row ranges disjoint), so
// there are no write races.

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

struct LogitBiasEntry {
    int32_t row0;  // first logits row it applies to
    int32_t rows;  // rows from row0 (>= 1)
    int32_t id;    // token id (column), < ld
    float v;       // the bias, finite
};
static_assert(sizeof(LogitBiasEntry) == 16, "LogitBiasEntry is uploaded from a host array as 16 bytes");

// logits [n_rows, ld] FP32 on the device; entries [n] on the device (1 <= n), each inside [0, n_rows) x [0, ld) -
// the host checks that before the upload (Qwen4ExpSession::want_candidates), the kernel does not.
void logits_bias(float *logits, int64_t n_rows, int64_t ld, const LogitBiasEntry *entries, int64_t n,
                 hipStream_t stream);

}  // namespace strix::kernels
