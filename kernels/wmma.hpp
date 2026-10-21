#pragma once

// RDNA3 / RDNA3.5 (gfx11) matrix-core tile, wave32: D[16x16] = A[16x16] * B[16x16] + C, BF16 inputs, FP32
// accumulation (v_wmma_f32_16x16x16_bf16). Device-only helpers for the tiled GEMMs; the lane layout below is
// checked end to end (kernels::wmma_tile_product), since everything built on it depends on it:
//   A (M x K): lane l < 16 holds row l's 16 K values; lanes 16..31 hold the same (the ISA reads both halves).
//   B (K x N): lane l < 16 holds column l's 16 K values - i.e. row l of B^T, which is how a weight W [N, K]
//              is stored, so a W tile loads like an A tile; lanes 16..31 again duplicate.
//   C / D:     8 floats per lane; element v of lane l is row 2v + l / 16, column l % 16.
// The FP32 accumulation is not bit-exact IEEE: up to 2^-23 of sum|a*b| per 16x16x16 step (measured on gfx1100) -
// FP32 noise, far below the BF16 input rounding (2^-9).
// Lower-precision path: the inputs are BF16-rounded; only kernels behind
// an explicit switch use it.

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// Host entry point for the layout test: c [16, 16] FP32 = a [16, 16] * b^T, with a and bt [16, 16] as BF16 bits
// row-major (bt = B transposed, the weight layout). Device pointers; synchronous on the null stream.
void wmma_tile_product(const uint16_t *a, const uint16_t *bt, float *c);

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
typedef __bf16 wmma_bf16x16 __attribute__((ext_vector_type(16)));
typedef float wmma_f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t wmma_u32x8 __attribute__((ext_vector_type(8)));

__device__ inline wmma_f32x8 wmma_bf16(wmma_bf16x16 a, wmma_bf16x16 b, wmma_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// One 16-value K run of row (lane % 16) of a row-major BF16 tile (bits) with row stride ld (elements, even): an A
// fragment, or a B fragment from W [N, K] rows. p must be 4-byte aligned at every row.
__device__ inline wmma_bf16x16 wmma_load_rows(const uint16_t *p, int ld) {
    const uint32_t *r = reinterpret_cast<const uint32_t *>(p + (threadIdx.x & 15) * ld);
    wmma_u32x8 v;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = r[i];
    return __builtin_bit_cast(wmma_bf16x16, v);
}

// wmma_load_rows for 16-byte aligned rows (p and ld * 2 bytes multiples of 16): two 128-bit LDS reads per lane
// instead of eight 32-bit ones (the compiler can't prove the alignment itself).
__device__ inline wmma_bf16x16 wmma_load_rows_a16(const uint16_t *p, int ld) {
    const uint4 *r = reinterpret_cast<const uint4 *>(p + (threadIdx.x & 15) * ld);
    const uint4 lo = r[0], hi = r[1];
    const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(wmma_bf16x16, v);
}

// F16 inputs: the same tile and lane layout as the BF16 instruction, F16 A / B,
// FP32 accumulation (v_wmma_f32_16x16x16_f16).
typedef _Float16 wmma_f16x16 __attribute__((ext_vector_type(16)));

__device__ inline wmma_f32x8 wmma_f16(wmma_f16x16 a, wmma_f16x16 b, wmma_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// An F16 A fragment from 16-byte aligned LDS rows (as wmma_load_rows_a16, F16 bits).
__device__ inline wmma_f16x16 wmma_load_rows_a16_f16(const uint16_t *p, int ld) {
    const uint4 *r = reinterpret_cast<const uint4 *>(p + (threadIdx.x & 15) * ld);
    const uint4 lo = r[0], hi = r[1];
    const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(wmma_f16x16, v);
}

// Row of accumulator element v on this lane (its column is lane % 16).
__device__ inline int wmma_c_row(int v) { return 2 * v + (((int)threadIdx.x & 31) >> 4); }
#endif

}  // namespace strix::kernels
