#pragma once

// A dense projection's weight in whichever format the layout gave it (the converter's per-class
// --layout): Q4 (formats/q4), Q5 / Q6 (formats/q5, q6) or Q8 (formats/q8), row-major, resident on the device. linear_qw dispatches to
// the matching kernel - the per-tensor format is read from the strixw file at load, never guessed.

#include "common/check.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q5.hpp"
#include "kernels/linear_q6.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"

#include <cstdint>

namespace strix {

struct QWeightView {
    int bits = 0;  // 4, 5, 6 or 8; 0 = not loaded
    Q4DeviceView q4;
    Q8DeviceView q8;
    Q6DeviceView q6;
    Q5DeviceView q5;
    int64_t N() const { return bits == 8 ? q8.N : bits == 6 ? q6.N : bits == 5 ? q5.N : q4.N; }
    int64_t K() const { return bits == 8 ? q8.K : bits == 6 ? q6.K : bits == 5 ? q5.K : q4.K; }
};

inline QWeightView qweight(const Q4DeviceView &v) { return {4, v, {}, {}, {}}; }
inline QWeightView qweight(const Q8DeviceView &v) { return {8, {}, v, {}, {}}; }
inline QWeightView qweight(const Q6DeviceView &v) { return {6, {}, {}, v, {}}; }
inline QWeightView qweight(const Q5DeviceView &v) { return {5, {}, {}, {}, v}; }

// The same for a chunk-major weight (hc_mix_up's W_up).
struct QChunkMajorView {
    int bits = 0;  // 4 or 8; 0 = not loaded
    Q4ChunkMajorView q4;
    Q8ChunkMajorView q8;
};
inline QChunkMajorView qweight(const Q4ChunkMajorView &v) { return {4, v, {}}; }
inline QChunkMajorView qweight(const Q8ChunkMajorView &v) { return {8, {}, v}; }

namespace kernels {

// y[M, N] = x[M, K] * dequant(W)^T with W's own kernel (linear_q4 / q5 / q6 / q8; their requirements apply).
inline void linear_qw(const void *x, const QWeightView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4(x, w.q4, y, M, act, out_act, stream); return;
        case 5: linear_q5(x, w.q5, y, M, act, out_act, stream); return;
        case 6: linear_q6(x, w.q6, y, M, act, out_act, stream); return;
        case 8: linear_q8(x, w.q8, y, M, act, out_act, stream); return;
        default: STRIX_FAIL("linear_qw: weight has ", w.bits, " bits (not loaded?), expected 4, 5, 6 or 8");
    }
}
inline void linear_qw(const void *x, const QWeightView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_qw(x, w, y, M, act, act, stream);
}
// The same on the matrix units (kernels/linear_wmma: BF16 inputs, a lower-precision path behind a switch).
inline void linear_qw_wmma(const void *x, const QWeightView &w, void *y, int64_t M, Act act, Act out_act,
                           hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4_wmma(x, w.q4, y, M, act, out_act, stream); return;
        case 5: linear_q5_wmma(x, w.q5, y, M, act, out_act, stream); return;
        case 6: linear_q6_wmma(x, w.q6, y, M, act, out_act, stream); return;
        case 8: linear_q8_wmma(x, w.q8, y, M, act, out_act, stream); return;
        default: STRIX_FAIL("linear_qw_wmma: weight has ", w.bits, " bits (not loaded?), expected 4, 5, 6 or 8");
    }
}

// MoE experts stacked [E*N, K] in Q4 or Q5 (the expert kernels' formats; kernels/linear_q4.hpp, moe_grouped.hpp).
inline void linear_qw_experts_gather(const void *x, const QWeightView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4_experts_gather(x, w.q4, E, ids, M, A, y, act, err, stream); return;
        case 5: linear_q5_experts_gather(x, w.q5, E, ids, M, A, y, act, err, stream); return;
        default: STRIX_FAIL("linear_qw_experts_gather: experts have ", w.bits, " bits, expected 4 or 5");
    }
}
// The gather with the SwiGLU in its epilogue: w's expert rows [gate | up], h [M, A, N/2] (bit-identical to the
// gather then kernels::swiglu).
inline void linear_qw_experts_gather_swiglu(const void *x, const QWeightView &w, int64_t E, const int32_t *ids,
                                            int64_t M, int64_t A, void *h, Act act, uint32_t *err,
                                            hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4_experts_gather_swiglu(x, w.q4, E, ids, M, A, h, act, err, stream); return;
        case 5: linear_q5_experts_gather_swiglu(x, w.q5, E, ids, M, A, h, act, err, stream); return;
        default: STRIX_FAIL("linear_qw_experts_gather_swiglu: experts have ", w.bits, " bits, expected 4 or 5");
    }
}
inline void linear_qw_experts_combine(const void *h, const QWeightView &w, int64_t E, const int32_t *ids,
                                      const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                                      hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4_experts_combine(h, w.q4, E, ids, coef, M, A, y, act, err, stream); return;
        case 5: linear_q5_experts_combine(h, w.q5, E, ids, coef, M, A, y, act, err, stream); return;
        default: STRIX_FAIL("linear_qw_experts_combine: experts have ", w.bits, " bits, expected 4 or 5");
    }
}
inline void linear_qw_experts_gather_grouped(const void *x, const QWeightView &w, int64_t E, const void *ws,
                                             size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                             hipStream_t stream) {
    switch (w.bits) {
        case 4: linear_q4_experts_gather_grouped(x, w.q4, E, ws, ws_bytes, M, A, y, act, math, stream); return;
        case 5: linear_q5_experts_gather_grouped(x, w.q5, E, ws, ws_bytes, M, A, y, act, math, stream); return;
        default: STRIX_FAIL("linear_qw_experts_gather_grouped: experts have ", w.bits, " bits, expected 4 or 5");
    }
}
// The grouped gather with the SwiGLU in its epilogue (MoeMath::WmmaF16 kinds; kernels/moe_grouped.hpp).
inline void linear_qw_experts_gather_grouped_swiglu(const void *x, const QWeightView &w, int64_t E, const void *ws,
                                                    size_t ws_bytes, int64_t M, int64_t A, void *h, Act act,
                                                    MoeMath math, hipStream_t stream, bool h_f16 = false) {
    switch (w.bits) {
        case 4: linear_q4_experts_gather_grouped_swiglu(x, w.q4, E, ws, ws_bytes, M, A, h, act, math, stream, h_f16); return;
        case 5: linear_q5_experts_gather_grouped_swiglu(x, w.q5, E, ws, ws_bytes, M, A, h, act, math, stream, h_f16); return;
        default: STRIX_FAIL("linear_qw_experts_gather_grouped_swiglu: experts have ", w.bits, " bits, expected 4 or 5");
    }
}
inline void linear_qw_experts_combine_grouped(const void *h, const QWeightView &w, int64_t E, const int32_t *ids,
                                              const float *coef, const void *ws, size_t ws_bytes, float *partial,
                                              int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                              hipStream_t stream, const MoeSharedAdd *shared = nullptr,
                                              bool h_f16 = false) {
    switch (w.bits) {
        case 4:
            linear_q4_experts_combine_grouped(h, w.q4, E, ids, coef, ws, ws_bytes, partial, M, A, y, act, math, stream,
                                              shared, h_f16);
            return;
        case 5:
            linear_q5_experts_combine_grouped(h, w.q5, E, ids, coef, ws, ws_bytes, partial, M, A, y, act, math, stream,
                                              shared, h_f16);
            return;
        default: STRIX_FAIL("linear_qw_experts_combine_grouped: experts have ", w.bits, " bits, expected 4 or 5");
    }
}

}  // namespace kernels

}  // namespace strix
