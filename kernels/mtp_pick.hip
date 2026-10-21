#include "kernels/mtp_pick.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"

#include <climits>
#include <cmath>

namespace strix::kernels {

namespace {

constexpr int kThreads = 1024;
constexpr int kLanes = 32;  // shuffle width (gfx11 wave32)

// A partial top-2: the best value and its index (INT_MAX = nothing seen yet), the runner-up value, NaN seen.
struct Part {
    float b1, b2;
    int i1;
    uint32_t nan;
};

__device__ inline Part merge(const Part &a, const Part &b) {
    if (b.i1 == INT_MAX) return {a.b1, a.b2, a.i1, a.nan | b.nan};
    if (a.i1 == INT_MAX) return {b.b1, b.b2, b.i1, a.nan | b.nan};
    const bool a_wins = a.b1 > b.b1 || (a.b1 == b.b1 && a.i1 < b.i1);
    const Part &w = a_wins ? a : b, &l = a_wins ? b : a;
    // The loser's best is at least its own runner-up, so it and the winner's runner-up are the candidates.
    return {w.b1, fmaxf(w.b2, l.b1), w.i1, a.nan | b.nan};
}

__device__ inline Part shfl_down(const Part &p, int off) {
    return {__shfl_down(p.b1, off, kLanes), __shfl_down(p.b2, off, kLanes), __shfl_down(p.i1, off, kLanes),
            (uint32_t)__shfl_down((int)p.nan, off, kLanes)};
}

__global__ void __launch_bounds__(kThreads)
    mtp_pick_kernel(const float *x, int64_t n, const uint32_t *router_err, const uint32_t *expert_err,
                    const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out) {
    __shared__ Part wave_part[kThreads / kLanes];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    // Each thread's indices ascend, so a strict > keeps the lower index of equal values (top2's order).
    Part p{-INFINITY, -INFINITY, INT_MAX, 0u};
    for (int64_t i = tid; i < n; i += kThreads) {
        const float v = x[i];
        if (v != v) {
            p.nan = 1u;
            continue;
        }
        if (p.i1 == INT_MAX || v > p.b1) {
            p.b2 = p.b1;  // -inf before the first logit
            p.b1 = v, p.i1 = (int)i;
        } else if (v > p.b2) {
            p.b2 = v;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) p = merge(p, shfl_down(p, off));
    if (lane == 0) wave_part[wave] = p;
    __syncthreads();
    if (wave != 0) return;
    p = lane < kThreads / kLanes ? wave_part[lane] : Part{-INFINITY, -INFINITY, INT_MAX, 0u};
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) p = merge(p, shfl_down(p, off));
    if (lane == 0) {
        out->best = p.i1 == INT_MAX ? 0 : p.i1;  // every logit NaN: top2 would answer 0 (and flag it)
        out->best_v = p.b1, out->second_v = p.b2, out->nan = p.nan;
        out->pad = 0;
        out->emb = *emb_err;
    }
    if (lane < 3) {
        out->err[lane] = router_err[lane];
        out->err[3 + lane] = expert_err[lane];
        out->err[6 + lane] = attn_err[lane];
    }
}

}  // namespace

void mtp_pick(const float *logits, int64_t n, const uint32_t *router_err, const uint32_t *expert_err,
              const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out, hipStream_t stream) {
    STRIX_CHECK(logits != nullptr && router_err != nullptr && expert_err != nullptr && attn_err != nullptr &&
                    emb_err != nullptr && out != nullptr,
                "mtp_pick: null pointer (logits=", (const void *)logits, ", router_err=", (const void *)router_err,
                ", expert_err=", (const void *)expert_err, ", attn_err=", (const void *)attn_err,
                ", emb_err=", (const void *)emb_err, ", out=", (const void *)out, ")");
    STRIX_CHECK(n >= 1 && n <= INT_MAX, "mtp_pick: n = ", n, " logits, expected 1..", INT_MAX);
    mtp_pick_kernel<<<1, kThreads, 0, stream>>>(logits, n, router_err, expert_err, attn_err, emb_err, out);
    STRIX_HIP_CHECK_LAUNCH("mtp_pick n=", n);
}

}  // namespace strix::kernels
