#include "kernels/linear_q4.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/qsplit.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerPass = 8;  // x rows accumulated together per W pass
constexpr int kChunk = 32;       // codes per lane per load (16 bytes)
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

// 32 consecutive activations of row `row` starting at k, as FP32.
__device__ inline void load_x32(const float *x, int64_t off, float (&v)[kChunk]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) {
        float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
__device__ inline void load_x32(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 8; ++i) {
        uint4 u = p[i];
        uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

// One output row per KS waves of a block: wave w handles row w / KS and the
// K chunks with index = w % KS (mod KS), interleaved so the KS waves of a row
// read adjacent 512-byte runs. Partial sums meet in LDS.
// TX/TY: activation types of x and y (the same except BF16 -> F32, for outputs kept FP32 such as the router logits).
template <typename TX, typename TY, int G, int MR>
__global__ void __launch_bounds__(kThreads) linear_q4_kernel(const TX *x, const uint8_t *q, const uint16_t *scale,
                                                             const uint16_t *minv, TY *y, int64_t M, int64_t N,
                                                             int64_t K, int KS) {
    __shared__ float part[kThreads / 32][MR];  // per wave (wave32 worst case)
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
        const uint4 *qrow = reinterpret_cast<const uint4 *>(q + n * (K / 2));
        const uint16_t *srow = scale + n * (K / G);
        const uint16_t *mrow = minv + n * (K / G);
        for (int64_t k = ((int64_t)ks * 32 + lane) * kChunk; k < K; k += (int64_t)KS * 32 * kChunk) {
            uint4 packed = qrow[k / kChunk];
            uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
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
                        uint32_t word = words[w];
#pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            const float qv = (float)((word >> (4 * b)) & 0xFu);
                            const float xval = xv[8 * w + b];
                            dot = fmaf(xval, qv, dot);
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

// Expert kernels (MoE): the weight is E experts stacked as one [E*N, K] Q4
// matrix, so expert e's rows are e*N .. e*N+N-1. Expert ids come from the
// device (router output); a bad id sets err (first one wins) and poisons its
// outputs with NaN instead of reading out of bounds.
__device__ inline bool expert_id_ok(int32_t id, int64_t E, uint32_t *err, int64_t slot) {
    if (id >= 0 && id < E) return true;
    if (atomicCAS(err, 0u, 1u) == 0u) {
        err[1] = (uint32_t)id;
        err[2] = (uint32_t)slot;
    }
    return false;
}

// Lane partial of dot(W row, x) over the chunks c = ks*warp + lane (mod
// KS*warp) of chunk range [0, chunks); chunk c covers codes 32c..32c+31.
// HB: the row's high-bit plane (kernels/qsplit.hpp) - 0 for Q4, 1 for Q5.
template <typename T, int G, int HB = 0>
__device__ inline float q4_row_partial(const uint8_t *qrow, const uint16_t *srow, const uint16_t *mrow, const T *xrow,
                                       int64_t chunks, int ks, int KS, int lane, int64_t K = 0) {
    float acc = 0.f;
    for (int64_t c = (int64_t)ks * 32 + lane; c < chunks; c += (int64_t)KS * 32) {
        uint4 packed = reinterpret_cast<const uint4 *>(qrow)[c];
        uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        const QHigh32<HB> hb = QHigh32<HB>::load(qrow, K, c * kChunk);
        const float s = bf16_bits_to_f32(srow[c * kChunk / G]);
        const float mn = bf16_bits_to_f32(mrow[c * kChunk / G]);
        float xv[kChunk];
        load_x32(xrow, c * kChunk, xv);
        float dot = 0.f, xsum = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = words[w];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const int idx = 8 * w + b;
                const float qv = (float)(((word >> (4 * b)) & 0xFu) | hb.at(idx));
                const float xval = xv[idx];
                dot = fmaf(xval, qv, dot);
                xsum += xval;
            }
        }
        acc = fmaf(s, dot, fmaf(mn, xsum, acc));
    }
    return acc;
}

// Wave-reduce each lane's partial, then combine the KS waves of a row in
// LDS; returns the row total on lane 0 of the row's first wave (ks == 0).
__device__ inline float reduce_row(float v, float *part, int wave, int ks, int KS) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    if (KS == 1) return v;
    if (threadIdx.x % 32 == 0) part[wave] = v;
    __syncthreads();
    if (ks != 0) return 0.f;
    float t = 0.f;
    for (int i = 0; i < KS; ++i) t += part[wave + i];
    return t;
}

// Gather: y[m, a, :] = x[m, :] * W[ids[m, a]]^T  (grid: rows, a, m).
template <typename T, int G, int HB>
__global__ void __launch_bounds__(kThreads)
    linear_q4_experts_gather_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                    const int32_t *ids, int64_t E, int64_t A, T *y, int64_t N, int64_t K, int KS,
                                    uint32_t *err) {
    __shared__ float part[kThreads / 32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ks = wave % KS;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS;
    const int64_t slot = (int64_t)blockIdx.z * A + blockIdx.y;
    const int32_t id = ids[slot];
    const bool ok = id >= 0 && id < E;
    float v = 0.f;
    if (ok && n < N) {  // no early return: every wave reaches reduce_row's barrier
        const int64_t row = (int64_t)id * N + n;
        v = q4_row_partial<T, G, HB>(q + row * qsplit_row_bytes<HB>(K), scale + row * (K / G), minv + row * (K / G),
                                     x + (int64_t)blockIdx.z * K, K / kChunk, ks, KS, lane, K);
    }
    v = reduce_row(v, part, wave, ks, KS);
    if (lane == 0 && ks == 0 && n < N) {
        if (!ok && n == 0) expert_id_ok(id, E, err, slot);
        store(y, slot * N + n, ok ? v : __int_as_float(0x7fc00000));
    }
}

// A value as its store and reload in dtype T would give it back (BF16: rounded to nearest even; F32: itself).
__device__ inline float as_stored(float v, const float *) { return v; }
__device__ inline float as_stored(float v, const uint16_t *) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }
__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }  // kernels/swiglu.hip's form

// Gather with the SwiGLU in its epilogue: W's expert rows are the merged [gate (I) | up (I)] projection, and a wave
// computes gate row j and up row j of one expert, then stores h[m, a, j] = silu(g) * u (y [M, A, I]). Bit-identical
// to the gather then swiglu(): g and u are rounded as their store / reload would, and h is formed as swiglu_bf16 /
// swiglu_f32 do. Saves the swiglu launch (and the gaps around it) and the [M, A, 2I] round trip.
template <typename T, int G, int HB>
__global__ void __launch_bounds__(kThreads)
    linear_q4_experts_gather_swiglu_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                           const int32_t *ids, int64_t E, int64_t A, T *y, int64_t N, int64_t K,
                                           int KS, uint32_t *err) {
    __shared__ float part_g[kThreads / 32], part_u[kThreads / 32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ks = wave % KS;
    const int64_t I = N / 2, j = (int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS;
    const int64_t slot = (int64_t)blockIdx.z * A + blockIdx.y;
    const int32_t id = ids[slot];
    const bool ok = id >= 0 && id < E;
    float vg = 0.f, vu = 0.f;
    if (ok && j < I) {  // no early return: every wave reaches reduce_row's barrier
        const T *xr = x + (int64_t)blockIdx.z * K;
        const int64_t rg = (int64_t)id * N + j, ru = rg + I;
        vg = q4_row_partial<T, G, HB>(q + rg * qsplit_row_bytes<HB>(K), scale + rg * (K / G), minv + rg * (K / G), xr,
                                      K / kChunk, ks, KS, lane, K);
        vu = q4_row_partial<T, G, HB>(q + ru * qsplit_row_bytes<HB>(K), scale + ru * (K / G), minv + ru * (K / G), xr,
                                      K / kChunk, ks, KS, lane, K);
    }
    vg = reduce_row(vg, part_g, wave, ks, KS);
    vu = reduce_row(vu, part_u, wave, ks, KS);
    if (lane == 0 && ks == 0 && j < I) {
        if (!ok && j == 0) expert_id_ok(id, E, err, slot);
        const float g = as_stored(vg, y), u = as_stored(vu, y);
        const float hv = as_stored(silu(g), y) * u;  // BF16: bf16(bf16(silu(g)) * u), as swiglu_bf16
        store(y, slot * I + j, ok ? hv : __int_as_float(0x7fc00000));
    }
}

// Combine: y[m, :] = sum_a coef[m, a] * h[m, a, :] * W[ids[m, a]]^T - one
// output row is a single A*K-long reduction; lanes walk its chunks flattened
// across experts, so short expert rows (K=640 is 20 chunks) still fill a wave.
template <typename T, int G, int HB>
__global__ void __launch_bounds__(kThreads)
    linear_q4_experts_combine_kernel(const T *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                     const int32_t *ids, const float *coef, int64_t E, int64_t A, T *y, int64_t N,
                                     int64_t K, int KS, uint32_t *err) {
    __shared__ float part[kThreads / 32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ks = wave % KS;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS;
    const int64_t m = blockIdx.y;
    const uint32_t kc = (uint32_t)(K / kChunk);  // chunks per expert row (32-bit: A*kc < 2^32, checked on launch)
    bool ok = true;
    for (int64_t a = 0; a < A; ++a) ok = ok && ids[m * A + a] >= 0 && ids[m * A + a] < E;
    float acc = 0.f;
    if (ok && n < N) {
        for (uint32_t c = (uint32_t)(ks * 32 + lane); c < (uint32_t)A * kc; c += (uint32_t)(KS * 32)) {
            const uint32_t a = c / kc, cc = c % kc;
            const int64_t row = (int64_t)ids[m * A + a] * N + n;
            const uint8_t *qrow = q + row * qsplit_row_bytes<HB>(K);
            uint4 packed = reinterpret_cast<const uint4 *>(qrow)[cc];
            uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const QHigh32<HB> hb = QHigh32<HB>::load(qrow, K, (int64_t)cc * kChunk);
            const float s = bf16_bits_to_f32(scale[row * (K / G) + (int64_t)cc * kChunk / G]);
            const float mn = bf16_bits_to_f32(minv[row * (K / G) + (int64_t)cc * kChunk / G]);
            float xv[kChunk];
            load_x32(h, (m * A + a) * K + (int64_t)cc * kChunk, xv);
            float dot = 0.f, xsum = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t word = words[w];
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const int idx = 8 * w + b;
                    const float qv = (float)(((word >> (4 * b)) & 0xFu) | hb.at(idx));
                    const float xval = xv[idx];
                    dot = fmaf(xval, qv, dot);
                    xsum += xval;
                }
            }
            acc = fmaf(coef[m * A + a], fmaf(s, dot, mn * xsum), acc);
        }
    }
    acc = reduce_row(acc, part, wave, ks, KS);
    if (lane == 0 && ks == 0 && n < N) {
        if (!ok && n == 0)
            for (int64_t a = 0; a < A; ++a)
                if (!expert_id_ok(ids[m * A + a], E, err, m * A + a)) break;
        store(y, m * N + n, ok ? acc : __int_as_float(0x7fc00000));
    }
}

int device_warp_size() { return kernels::device_warp_size(kThreads); }

template <typename TX, typename TY, int MR>
void launch_mr(const TX *x, const Q4DeviceView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    int waves = kThreads / device_warp_size();
    STRIX_CHECK(KS >= 1 && waves % KS == 0, "linear_q4: split-K factor ", KS, " must divide the ", waves,
                " waves of a block");
    int64_t rows_per_block = waves / KS;
    dim3 grid((unsigned)((w.N + rows_per_block - 1) / rows_per_block), (unsigned)((M + MR - 1) / MR)),
        block(kThreads);
    switch (w.G) {
        case 32: linear_q4_kernel<TX, TY, 32, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 64: linear_q4_kernel<TX, TY, 64, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 128: linear_q4_kernel<TX, TY, 128, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        default: STRIX_FAIL("linear_q4: group size ", w.G, " unsupported (32, 64 or 128)");
    }
}

// Split-K factor: waves per output row. One wave per row leaves long-K
// shapes with few, long waves and an underfilled last round on the GPU;
// splitting so each wave covers ~2048 codes measured best across the bench
// shapes, while short K loses to the LDS reduction.
int split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K / (2048 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

template <typename TX, typename TY>
void launch(const TX *x, const Q4DeviceView &w, TY *y, int64_t M, hipStream_t stream) {
    // Rows of x per weight pass: as many as there are, up to kRowsPerPass - fewer accumulators for short M
    // (MTP verify, M = 2-5) keep the occupancy of the M = 1 kernel.
    if (M == 1) launch_mr<TX, TY, 1>(x, w, y, M, split_k(w.K), stream);
    else if (M == 2) launch_mr<TX, TY, 2>(x, w, y, M, split_k(w.K), stream);
    else if (M <= 4) launch_mr<TX, TY, 4>(x, w, y, M, split_k(w.K), stream);
    else launch_mr<TX, TY, kRowsPerPass>(x, w, y, M, split_k(w.K), stream);
}

template <typename T, int HB>
void launch_experts(const T *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, const float *coef, int64_t M,
                    int64_t A, T *y, uint32_t *err, hipStream_t stream, bool swiglu) {
    const int64_t N = w.N / E;
    const bool combine = coef != nullptr;
    // Split by the contiguous row length (K), as for the dense kernel: combine's
    // A*K reduction is already A separate rows, and splitting it further
    // measured slower (2560x640 x10: 85% unsplit, 78% at 2 waves, 64% at 8).
    const int KS = split_k(w.K);
    const int waves = kThreads / device_warp_size();
    STRIX_CHECK(waves % KS == 0, "linear_q4 experts: split-K factor ", KS, " must divide the ", waves, " waves");
    const int64_t rows_per_block = waves / KS;
    const int64_t out_rows = swiglu ? N / 2 : N;  // swiglu: a wave per (gate, up) row pair
    const unsigned bx = (unsigned)((out_rows + rows_per_block - 1) / rows_per_block);
    dim3 block(kThreads);
#define STRIX_Q4_EXPERTS(G)                                                                                         \
    if (swiglu)                                                                                                     \
        linear_q4_experts_gather_swiglu_kernel<T, G, HB><<<dim3(bx, (unsigned)A, (unsigned)M), block, 0, stream>>>( \
            x, w.q, w.scale, w.minv, ids, E, A, y, N, w.K, KS, err);                                                \
    else if (combine)                                                                                               \
        linear_q4_experts_combine_kernel<T, G, HB><<<dim3(bx, (unsigned)M), block, 0, stream>>>(                    \
            x, w.q, w.scale, w.minv, ids, coef, E, A, y, N, w.K, KS, err);                                          \
    else                                                                                                            \
        linear_q4_experts_gather_kernel<T, G, HB><<<dim3(bx, (unsigned)A, (unsigned)M), block, 0, stream>>>(        \
            x, w.q, w.scale, w.minv, ids, E, A, y, N, w.K, KS, err);
    switch (w.G) {
        case 32: STRIX_Q4_EXPERTS(32) break;
        case 64: STRIX_Q4_EXPERTS(64) break;
        case 128: STRIX_Q4_EXPERTS(128) break;
        default: STRIX_FAIL("linear_q4 experts: group size ", w.G, " unsupported (32, 64 or 128)");
    }
#undef STRIX_Q4_EXPERTS
}

void check_weight_view(const char *fn, const Q4DeviceView &w) {
    STRIX_CHECK(w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn, ": null weight pointer (q=",
                (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.N >= 1 && w.N <= kMaxN, fn, ": N = ", w.N, ", expected 1..", kMaxN);
    STRIX_CHECK(w.K >= w.G && w.K <= kMaxK && w.K % w.G == 0, fn, ": K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and <= ", kMaxK);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, fn, ": code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
}

void check_no_overlap(const char *fn, const void *in, int64_t in_bytes, const void *out, int64_t out_bytes) {
    auto ib = reinterpret_cast<uintptr_t>(in), ob = reinterpret_cast<uintptr_t>(out);
    STRIX_CHECK(ob + (uintptr_t)out_bytes <= ib || ib + (uintptr_t)in_bytes <= ob, fn, ": output [", out, ", +",
                out_bytes, " B) overlaps input [", in, ", +", in_bytes, " B)");
}

// hb: the rows' high-bit plane (kernels/qsplit.hpp) - 0 for Q4, 1 for Q5 (w then holds a Q5 view's pointers and
// shape: the same fields).
// swiglu (gathers only): W holds [gate | up] rows per expert and y gets h = silu(gate) * up, [M, A, N/2].
void linear_q4_experts(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                       const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream,
                       int hb = 0, bool swiglu = false) {
    STRIX_CHECK(x != nullptr && y != nullptr && ids != nullptr && err != nullptr, fn, ": null pointer (x=", x,
                ", y=", y, ", ids=", (const void *)ids, ", err=", (const void *)err, ")");
    check_weight_view(fn, w);
    STRIX_CHECK(hb == 0 || hb == 1, fn, ": high bits ", hb, ", expected 0 (Q4) or 1 (Q5)");
    STRIX_CHECK(hb == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    STRIX_CHECK(E >= 1 && w.N % E == 0, fn, ": weight has ", w.N, " rows, not a multiple of E = ", E, " experts");
    STRIX_CHECK(M >= 1 && M <= 65535, fn, ": M (tokens) = ", M, ", expected 1..65535");
    STRIX_CHECK(A >= 1 && A <= 64 && A <= E, fn, ": A (active experts per token) = ", A, ", expected 1..min(64, E=",
                E, ")");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": input ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    const bool combine = coef != nullptr;
    STRIX_CHECK(!(swiglu && combine), fn, ": the SwiGLU epilogue is for gathers only");
    STRIX_CHECK(!swiglu || N % 2 == 0, fn, ": SwiGLU gather needs an even row count per expert ([gate | up]), got ",
                N);
    check_no_overlap(fn, x, (combine ? M * A : M) * w.K * es, y, (combine ? M : M * A) * (swiglu ? N / 2 : N) * es);
    auto go = [&](auto hbc) {
        constexpr int HB = decltype(hbc)::value;
        if (act == Act::F32)
            launch_experts<float, HB>(static_cast<const float *>(x), w, E, ids, coef, M, A, static_cast<float *>(y),
                                      err, stream, swiglu);
        else
            launch_experts<uint16_t, HB>(static_cast<const uint16_t *>(x), w, E, ids, coef, M, A,
                                         static_cast<uint16_t *>(y), err, stream, swiglu);
    };
    if (hb == 1) go(std::integral_constant<int, 1>{});
    else go(std::integral_constant<int, 0>{});
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " M=", M, " A=", A, " E=", E, " N=", N, " K=", w.K, " G=", w.G,
                           swiglu ? " swiglu" : "");
}

}  // namespace

void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q4(x, w, y, M, act, act, stream);
}

void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    STRIX_CHECK(x != nullptr && y != nullptr && w.q != nullptr && w.scale != nullptr && w.minv != nullptr,
                "linear_q4: null pointer (x=", x, ", y=", y, ", q=", (const void *)w.q, ", scale=", w.scale,
                ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, "linear_q4: group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, "linear_q4: M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= kMaxN, "linear_q4: N = ", w.N, ", expected 1..", kMaxN);
    STRIX_CHECK(w.K >= w.G && w.K <= kMaxK && w.K % w.G == 0, "linear_q4: K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and <= ", kMaxK);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, "linear_q4: code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, "linear_q4: x ", x, " must be 16-byte aligned for vector loads");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "linear_q4: unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), "linear_q4: output dtype ",
                act_name(out_act), " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    uintptr_t xe = xb + (uintptr_t)(M * w.K) * es, ye = yb + (uintptr_t)(M * w.N) * oes;
    STRIX_CHECK(ye <= xb || xe <= yb, "linear_q4: output y [", y, ", +", M * w.N * (int64_t)oes,
                " B) overlaps input x [", x, ", +", M * w.K * (int64_t)es, " B)");

    if (act == Act::F32) launch(static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
    else if (out_act == Act::F32) launch(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
    else launch(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    STRIX_HIP_CHECK_LAUNCH("linear_q4 ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K,
                           " G=", w.G);
}

}  // namespace strix::kernels

namespace strix::kernels {

void linear_q4_experts_gather(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                              int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream) {
    linear_q4_experts("linear_q4_experts_gather", x, w, E, ids, nullptr, M, A, y, act, err, stream);
}

void linear_q4_experts_gather_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream) {
    linear_q4_experts("linear_q4_experts_gather_swiglu", x, w, E, ids, nullptr, M, A, h, act, err, stream, 0, true);
}

void linear_q5_experts_gather_swiglu(const void *x, const Q5DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream) {
    linear_q4_experts("linear_q5_experts_gather_swiglu", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ids, nullptr, M,
                      A, h, act, err, stream, 1, true);
}

void linear_q4_experts_combine(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                               const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                               hipStream_t stream) {
    STRIX_CHECK(coef != nullptr, "linear_q4_experts_combine: null coef");
    linear_q4_experts("linear_q4_experts_combine", h, w, E, ids, coef, M, A, y, act, err, stream);
}

void linear_q5_experts_gather(const void *x, const Q5DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                              int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream) {
    linear_q4_experts("linear_q5_experts_gather", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ids, nullptr, M, A, y,
                      act, err, stream, 1);
}

void linear_q5_experts_combine(const void *h, const Q5DeviceView &w, int64_t E, const int32_t *ids,
                               const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                               hipStream_t stream) {
    STRIX_CHECK(coef != nullptr, "linear_q5_experts_combine: null coef");
    linear_q4_experts("linear_q5_experts_combine", h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ids, coef, M, A, y,
                      act, err, stream, 1);
}

void check_expert_error(const uint32_t *err, const char *what) {
    STRIX_CHECK(err != nullptr, "check_expert_error(", what, "): null err");
    uint32_t e[3];
    STRIX_HIP_CHECK(hipMemcpy(e, err, sizeof(e), hipMemcpyDeviceToHost), "reading expert error word for ", what);
    report_expert_error(e, what);
}

void report_expert_error(const uint32_t e[3], const char *what) {
    STRIX_CHECK(e[0] == 0, what, ": expert id ", (int32_t)e[1], " at slot ", e[2],
                " (token * A + a) is out of range; outputs for that token are NaN");
}

}  // namespace strix::kernels
