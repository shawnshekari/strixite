#pragma once

// The MTP draft's pick on the GPU: the largest logit (lowest index on a tie), the runner-up's value (the multiset
// second - equal to the best on a tie, -inf for one logit) and whether any logit is NaN - serve/sampler.hpp top2()'s
// result for non-NaN rows - written with the forward's error words into one small struct, so a draft step ends with
// one ~64-byte copy to the host instead of five copies and the whole draft vocabulary's logits (256 KB for 65536 ids)
// scanned on the CPU. A trace of the draft loop had ~0.3 ms of each ~1.2 ms draft
// step idle around those copies and the host scan.

#include "kernels/embedding.hpp"

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// Device and host layout (one copy): the pick, then the router / expert / attention error words (3 each, as
// read_back lays them out) and the embedding error slot.
struct MtpPick {
    int32_t best;
    float best_v, second_v;
    uint32_t nan;  // 1 if any logit was NaN (best / values are then meaningless)
    uint32_t err[9];
    uint32_t pad;
    EmbeddingError emb;
};
static_assert(sizeof(MtpPick) == 64, "MtpPick is copied as one 64-byte block");

// logits [n] FP32 (1 <= n <= 2^31 - 1); the error words and slot are the forward's device ones (read, not reset);
// out on the device. One block.
void mtp_pick(const float *logits, int64_t n, const uint32_t *router_err, const uint32_t *expert_err,
              const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out, hipStream_t stream);

// Batched MTP pick: B independent sets of logits [B * n], producing B MtpPick structs in out.
void mtp_pick_batch(const float *logits, int64_t n, int64_t B, const uint32_t *router_err, const uint32_t *expert_err,
                    const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out, hipStream_t stream);

}  // namespace strix::kernels
