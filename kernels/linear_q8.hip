#include "kernels/linear_q8.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerPass = 8;  // x rows accumulated together per W pass
constexpr int kChunk = 16;       // codes per lane per load (16 bytes)
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

// 16 consecutive activations starting at off, as FP32.
__device__ inline void load_x16(const float *x, int64_t off, float (&v)[kChunk]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) {
        const float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
__device__ inline void load_x16(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
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

// BF16 activations: the products go through v_dot2_f32_bf16 (two bf16 products summed into an FP32 accumulator in
// one instruction). The codes (0..255, exact in BF16) are converted to packed BF16 pairs once per chunk and shared by
// every row; x stays in its packed BF16 pairs - no per-row conversion - and the group sum of x is a second dot2
// against (1, 1). At M = 5 the scalar form was instruction-bound (35-37% of wave cycles waiting to issue, 3+ VALU
// per weight per row); this is ~1. Products are exact in FP32; only the summation order
// differs from the scalar form.
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
// Codes 2p and 2p+1 of a chunk (bytes of `words`) as a packed BF16 pair.
__device__ inline uint32_t code_pair(const uint32_t (&words)[4], int p) {
    const uint32_t w = words[p / 2], sh = 16 * (p % 2);
    const float lo = (float)((w >> sh) & 0xFFu), hi = (float)((w >> (sh + 8)) & 0xFFu);
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}
constexpr uint32_t kOnePair = 0x3F803F80u;  // BF16 (1, 1)

// One output row per KS waves of a block (as linear_q4_kernel): wave w handles row w / KS and the chunks with
// index = w % KS (mod KS), so the KS waves of a row read adjacent 512-byte runs; partial sums meet in LDS.
template <typename TX, typename TY, int G, int MR>
__global__ void __launch_bounds__(kThreads) linear_q8_kernel(const TX *x, const uint8_t *q, const uint16_t *scale,
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
        const uint4 *qrow = reinterpret_cast<const uint4 *>(q + n * K);
        const uint16_t *srow = scale + n * (K / G);
        const uint16_t *mrow = minv + n * (K / G);
        for (int64_t k = ((int64_t)ks * 32 + lane) * kChunk; k < K; k += (int64_t)KS * 32 * kChunk) {
            const uint4 packed = qrow[k / kChunk];
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const float s = bf16_bits_to_f32(srow[k / G]);
            const float mn = bf16_bits_to_f32(mrow[k / G]);
            if constexpr (std::is_same_v<TX, uint16_t>) {
                uint32_t cp[kChunk / 2];
#pragma unroll
                for (int p = 0; p < kChunk / 2; ++p) cp[p] = code_pair(words, p);
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (r < rows) {
                        const uint4 *xp = reinterpret_cast<const uint4 *>(x + (m0 + r) * K + k);
                        const uint4 u0 = xp[0], u1 = xp[1];
                        const uint32_t xw[kChunk / 2] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                        float dot = 0.f, xsum = 0.f;
#pragma unroll
                        for (int p = 0; p < kChunk / 2; ++p) {
                            dot = dot2(xw[p], cp[p], dot);
                            xsum = dot2(xw[p], kOnePair, xsum);
                        }
                        acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                    }
                }
                continue;
            }
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (r < rows) {
                    float xv[kChunk];
                    load_x16(x, (m0 + r) * K + k, xv);
                    float dot = 0.f, xsum = 0.f;
#pragma unroll
                    for (int i = 0; i < kChunk; ++i) {
                        dot = fmaf(xv[i], (float)((words[i / 4] >> (8 * (i % 4))) & 0xFFu), dot);
                        xsum += xv[i];
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

// Split-K factor: waves per output row. Swept at the target's shapes (M = 1, KS 1/2/4/8): K <= 2560 is best at KS 1 (3-8% over KS 2 on the small, latency-bound matrices - the
// shared expert, [512-1280, 2560] - and equal on the large ones); K = 6144 at KS 4 (KS 1 is 30% slower), which the
// ~2048 code bytes per wave rule gives.
int split_k(int64_t K) {
    if (K <= 2560) return 1;
    int ks = 1;
    while (ks < 8 && K / (1024 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

template <typename TX, typename TY, int MR>
void launch_mr(const TX *x, const Q8DeviceView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    const int waves = kThreads / device_warp_size(kThreads);
    STRIX_CHECK(KS >= 1 && waves % KS == 0, "linear_q8: split-K factor ", KS, " must divide the ", waves,
                " waves of a block");
    const int64_t rows_per_block = waves / KS;
    const dim3 grid((unsigned)((w.N + rows_per_block - 1) / rows_per_block), (unsigned)((M + MR - 1) / MR)),
        block(kThreads);
    switch (w.G) {
        case 32: linear_q8_kernel<TX, TY, 32, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 64: linear_q8_kernel<TX, TY, 64, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 128: linear_q8_kernel<TX, TY, 128, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        default: STRIX_FAIL("linear_q8: group size ", w.G, " unsupported (32, 64 or 128)");
    }
}

template <typename TX, typename TY>
void launch(const TX *x, const Q8DeviceView &w, TY *y, int64_t M, hipStream_t stream) {
    // Rows of x per weight pass: as many as there are, up to kRowsPerPass - fewer accumulators for short M
    // (MTP verify, M = 2-5) keep the occupancy of the M = 1 kernel.
    if (M == 1) launch_mr<TX, TY, 1>(x, w, y, M, split_k(w.K), stream);
    else if (M == 2) launch_mr<TX, TY, 2>(x, w, y, M, split_k(w.K), stream);
    else if (M <= 4) launch_mr<TX, TY, 4>(x, w, y, M, split_k(w.K), stream);
    else launch_mr<TX, TY, kRowsPerPass>(x, w, y, M, split_k(w.K), stream);
}

}  // namespace

void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q8(x, w, y, M, act, act, stream);
}

void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    STRIX_CHECK(x != nullptr && y != nullptr && w.q != nullptr && w.scale != nullptr && w.minv != nullptr,
                "linear_q8: null pointer (x=", x, ", y=", y, ", q=", (const void *)w.q, ", scale=", w.scale,
                ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, "linear_q8: group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, "linear_q8: M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= kMaxN, "linear_q8: N = ", w.N, ", expected 1..", kMaxN);
    STRIX_CHECK(w.K >= w.G && w.K <= kMaxK && w.K % w.G == 0, "linear_q8: K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and <= ", kMaxK);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, "linear_q8: code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, "linear_q8: x ", x, " must be 16-byte aligned for vector loads");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "linear_q8: unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), "linear_q8: output dtype ",
                act_name(out_act), " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    const uintptr_t xe = xb + (uintptr_t)(M * w.K) * es, ye = yb + (uintptr_t)(M * w.N) * oes;
    STRIX_CHECK(ye <= xb || xe <= yb, "linear_q8: output y [", y, ", +", M * w.N * (int64_t)oes,
                " B) overlaps input x [", x, ", +", M * w.K * (int64_t)es, " B)");
    if (act == Act::F32) launch(static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
    else if (out_act == Act::F32) launch(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
    else launch(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    STRIX_HIP_CHECK_LAUNCH("linear_q8 ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K,
                           " G=", w.G);
}

}  // namespace strix::kernels
