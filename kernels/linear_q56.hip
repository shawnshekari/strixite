// Q5 and Q6 (formats/q5.hpp, q6.hpp): a 4-bit low plane plus an HB-bit high plane per row; one kernel template.
#include "kernels/linear_q5.hpp"
#include "kernels/linear_q6.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerPass = 8;  // x rows accumulated together per W pass
constexpr int kChunk = 32;       // codes per lane per load (16 bytes of low nibbles + 4 * HB bytes of high bits)
constexpr int64_t kMaxN = 0x7fffffff;
constexpr int64_t kMaxM = 65535ll * kRowsPerPass;
constexpr int64_t kMaxK = 1 << 24;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// 32 consecutive activations starting at off, as FP32.
__device__ inline void load_x32(const float *x, int64_t off, float (&v)[kChunk]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) {
        const float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
__device__ inline void load_x32(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 8; ++i) {
        const uint4 u = p[i];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

// A chunk's high bits (32 codes x HB): HB 2 - 8 bytes (code i at bits 2 (i % 16) of word i / 16); HB 1 - 4 bytes
// (code i at bit i).
template <int HB>
struct HighBits {
    uint32_t w[HB];
    __device__ static HighBits load(const uint8_t *plane, int64_t k) {
        HighBits h;
        if constexpr (HB == 2) {
            const uint2 v = *reinterpret_cast<const uint2 *>(plane + k / 4);
            h.w[0] = v.x, h.w[1] = v.y;
        } else {
            h.w[0] = *reinterpret_cast<const uint32_t *>(plane + k / 8);
        }
        return h;
    }
    __device__ uint32_t at(int i) const {
        if constexpr (HB == 2) return (w[i / 16] >> (2 * (i % 16))) & 0x3u;
        else return (w[0] >> i) & 0x1u;
    }
};

// One output row per KS waves of a block (as linear_q4_kernel): wave w handles row w / KS and the chunks with
// index = w % KS (mod KS); partial sums meet in LDS. Code i of a chunk = low nibble i | high bits i << 4.
template <typename TX, typename TY, int HB, int G, int MR>
__global__ void __launch_bounds__(kThreads) linear_q56_kernel(const TX *x, const uint8_t *q, const uint16_t *scale,
                                                             const uint16_t *minv, TY *y, int64_t M, int64_t N,
                                                             int64_t K, int KS) {
    __shared__ float part[kThreads / 32][MR];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS;
    const int ks = wave % KS;
    const int64_t m0 = (int64_t)blockIdx.y * MR;
    const int rows = (int)((M - m0) < MR ? (M - m0) : MR);

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) acc[r] = 0.f;

    if (n < N) {  // no early return: every wave reaches the barrier below
        const uint8_t *row = q + n * (K / 2 + K * HB / 8);
        const uint4 *lo_row = reinterpret_cast<const uint4 *>(row);
        const uint8_t *hi_plane = row + K / 2;
        const uint16_t *srow = scale + n * (K / G);
        const uint16_t *mrow = minv + n * (K / G);
        for (int64_t k = ((int64_t)ks * 32 + lane) * kChunk; k < K; k += (int64_t)KS * 32 * kChunk) {
            const uint4 lo = lo_row[k / kChunk];
            const HighBits<HB> hb = HighBits<HB>::load(hi_plane, k);
            const uint32_t lw[4] = {lo.x, lo.y, lo.z, lo.w};
            const float s = bf16_bits_to_f32(srow[k / G]);
            const float mn = bf16_bits_to_f32(mrow[k / G]);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (r < rows) {
                    float xv[kChunk];
                    load_x32(x, (m0 + r) * K + k, xv);
                    float dot = 0.f, xsum = 0.f;
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        uint32_t word = lw[w];
#pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            const int idx = 8 * w + b;
                            const uint32_t c = ((word >> (4 * b)) & 0xFu) | (hb.at(idx) << 4);
                            const float xval = xv[idx];
                            dot = fmaf(xval, (float)c, dot);
                            xsum += xval;
                        }
                    }
                    acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down(acc[r], off, 32);
    }
    if (KS == 1) {
        if (lane == 0 && n < N)
            for (int r = 0; r < rows; ++r) store(y, (m0 + r) * N + n, acc[r]);
        return;
    }
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < MR; ++r) part[wave][r] = acc[r];
    __syncthreads();
    if (lane == 0 && ks == 0 && n < N)
        for (int r = 0; r < rows; ++r) {
            float v = 0.f;
            for (int i = 0; i < KS; ++i) v += part[wave + i][r];
            store(y, (m0 + r) * N + n, v);
        }
}

// Split-K factor: waves per output row, ~2048 codes a wave - linear_q4's rule (the same 32 codes per lane step).
int split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K / (2048 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

struct View {
    const uint8_t *q;
    const uint16_t *scale, *minv;
    int64_t N, K, G;
};

template <typename TX, typename TY, int HB, int MR>
void launch_mr(const char *fn, const TX *x, const View &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    const int waves = kThreads / device_warp_size(kThreads);
    STRIX_CHECK(KS >= 1 && waves % KS == 0, fn, ": split-K factor ", KS, " must divide the ", waves, " waves of a block");
    const int64_t rows_per_block = waves / KS;
    const dim3 grid((unsigned)((w.N + rows_per_block - 1) / rows_per_block), (unsigned)((M + MR - 1) / MR)),
        block(kThreads);
    switch (w.G) {
        case 32: linear_q56_kernel<TX, TY, HB, 32, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 64: linear_q56_kernel<TX, TY, HB, 64, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 128: linear_q56_kernel<TX, TY, HB, 128, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        default: STRIX_FAIL(fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    }
}

template <typename TX, typename TY, int HB>
void launch(const char *fn, const TX *x, const View &w, TY *y, int64_t M, hipStream_t stream) {
    // Rows of x per weight pass as linear_q4: fewer accumulators for short M (MTP verify, M = 2-5).
    if (M == 1) launch_mr<TX, TY, HB, 1>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M == 2) launch_mr<TX, TY, HB, 2>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M <= 4) launch_mr<TX, TY, HB, 4>(fn, x, w, y, M, split_k(w.K), stream);
    else launch_mr<TX, TY, HB, kRowsPerPass>(fn, x, w, y, M, split_k(w.K), stream);
}

// HB high bits: Q6 (2, K % 64) or Q5 (1, K % 128) - K / 2 + K * HB / 8 row bytes stay 16-byte aligned.
template <int HB>
void linear_q56(const char *fn, const void *x, const View &w, void *y, int64_t M, Act act, Act out_act,
                hipStream_t stream) {
    constexpr int64_t kKAlign = HB == 2 ? 64 : 128;
    STRIX_CHECK(x != nullptr && y != nullptr && w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn,
                ": null pointer (x=", x, ", y=", y, ", q=", (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, fn, ": M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= kMaxN, fn, ": N = ", w.N, ", expected 1..", kMaxN);
    STRIX_CHECK(w.K >= w.G && w.K <= kMaxK && w.K % w.G == 0 && w.K % kKAlign == 0, fn, ": K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and of ", kKAlign, ", and <= ", kMaxK);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, fn, ": code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": x ", x, " must be 16-byte aligned for vector loads");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    const uintptr_t xe = xb + (uintptr_t)(M * w.K) * es, ye = yb + (uintptr_t)(M * w.N) * oes;
    STRIX_CHECK(ye <= xb || xe <= yb, fn, ": output y [", y, ", +", M * w.N * (int64_t)oes, " B) overlaps input x [",
                x, ", +", M * w.K * (int64_t)es, " B)");
    if (act == Act::F32) launch<float, float, HB>(fn, static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
    else if (out_act == Act::F32)
        launch<uint16_t, float, HB>(fn, static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
    else launch<uint16_t, uint16_t, HB>(fn, static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K, " G=", w.G);
}

}  // namespace

void linear_q6(const void *x, const Q6DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q6(x, w, y, M, act, act, stream);
}
void linear_q6(const void *x, const Q6DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    linear_q56<2>("linear_q6", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}
void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q5(x, w, y, M, act, act, stream);
}
void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    linear_q56<1>("linear_q5", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}

}  // namespace strix::kernels
