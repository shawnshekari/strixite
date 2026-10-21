#include "kernels/hc.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kH = 4;              // streams (qwen4_exp hc_count); the kernels keep per-stream sums in registers
constexpr int kChunk = 32;         // codes per chunk: one 16-byte load at Q4, two at Q8
constexpr int64_t kMaxR = 1024;    // low rank, held in LDS by hc_mix_up
constexpr int64_t kMaxTokens = 65535;  // grid.y
constexpr int64_t kMaxD = 1 << 20;
constexpr size_t kMaxDownLds = 32768;  // hc_mix_down's chunk sums (float2 per 32 inputs): H*d up to 131072

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline float load(const float *p, int64_t i) { return p[i]; }
__device__ inline float load(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

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

// A 32-code chunk's BITS words (4 at Q4, 8 at Q8): 16-byte loads from p (16-byte aligned), code i out of them.
template <int BITS> __device__ inline void load_chunk(const uint8_t *p, uint32_t (&w)[BITS]) {
    const uint4 *v = reinterpret_cast<const uint4 *>(p);
#pragma unroll
    for (int i = 0; i < BITS / 4; ++i) {
        const uint4 u = v[i];
        w[4 * i] = u.x, w[4 * i + 1] = u.y, w[4 * i + 2] = u.z, w[4 * i + 3] = u.w;
    }
}
template <int BITS> __device__ inline float code(const uint32_t (&w)[BITS], int i) {
    constexpr int per = 32 / BITS;
    return (float)((w[i / per] >> (BITS * (i % per))) & ((1u << BITS) - 1u));
}

// hc_mix_down: RW weight rows per wave sharing each X load, KS waves per row group (split-K over the H*d
// inputs), grid.y = token. Each lane accumulates, per stream, its chunks' partial dots with the RW (folded)
// weight rows and its sum of X^2 - a 32-code chunk lies in one stream since d % 32 == 0. A chunk's sum and sum of
// squares are the same for every row: computed once per block into LDS (per row they were ~40% of the loop's
// VALU work at T = 5). Partials meet by shuffles, then in LDS across the KS waves. The token stays in the grid:
// sharing each weight load across a verify's tokens was slower here (28.2 vs 26.9 us at T = 5, Q8 - the T blocks
// read the matrix at the same time, so L2 serves the repeats, and T x the waves hide the latency better).
template <typename T, int G, int RW, int BITS>
__global__ void __launch_bounds__(kThreads)
    hc_down_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, int64_t R, int64_t K,
                   int64_t d, int r, float eps, int KS, float *h, float *w_in, float *inv) {
    constexpr int kV = (RW + 1) * kH;              // per wave: RW rows' dots per stream, then the sums of squares
    __shared__ float part[kThreads / 32][kV];      // per wave (wave32 worst case)
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int64_t n0 = ((int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS) * RW;
    const int ks = wave % KS;
    const int64_t t = blockIdx.y;
    const T *xr = x + t * K;
    extern __shared__ float2 xs[];  // [K / kChunk]: each chunk's sum and sum of squares (dynamic LDS)
    const int64_t nc = K / kChunk;
    for (int64_t c = threadIdx.x; c < nc; c += blockDim.x) {
        float xv[kChunk];
        load_x32(xr, c * kChunk, xv);
        float xsum = 0.f, sq = 0.f;
#pragma unroll
        for (int i = 0; i < kChunk; ++i) xsum += xv[i], sq = fmaf(xv[i], xv[i], sq);
        xs[c] = make_float2(xsum, sq);
    }
    __syncthreads();
    float v[kV];
#pragma unroll
    for (int i = 0; i < kV; ++i) v[i] = 0.f;
    if (n0 < R) {  // no early return: every wave reaches the barrier below
        for (int64_t c = (int64_t)ks * 32 + lane; c < nc; c += (int64_t)KS * 32) {
            float xv[kChunk];
            load_x32(xr, c * kChunk, xv);
            const float2 ss = xs[c];
            const float xsum = ss.x, sq = ss.y;
            const int g = (int)(c * kChunk / d);
#pragma unroll
            for (int rw = 0; rw < RW; ++rw) {
                const int64_t n = n0 + rw;
                if (n >= R) break;
                uint32_t words[BITS];
                load_chunk<BITS>(q + n * (K * BITS / 8) + c * (kChunk * BITS / 8), words);
                const float sc = bf16_bits_to_f32(scale[n * (K / G) + c * kChunk / G]);
                const float mn = bf16_bits_to_f32(minv[n * (K / G) + c * kChunk / G]);
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kChunk; ++i) dot = fmaf(xv[i], code<BITS>(words, i), dot);
                const float pv = fmaf(sc, dot, mn * xsum);
#pragma unroll
                for (int s = 0; s < kH; ++s)  // static indices keep v in registers
                    if (s == g) v[rw * kH + s] += pv;
            }
#pragma unroll
            for (int s = 0; s < kH; ++s)
                if (s == g) v[RW * kH + s] += sq;
        }
    }
#pragma unroll
    for (int i = 0; i < kV; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v[i] += __shfl_down(v[i], off, 32);
    }
    if (KS > 1) {
        if (lane == 0)
#pragma unroll
            for (int i = 0; i < kV; ++i) part[wave][i] = v[i];
        __syncthreads();
        if (lane == 0 && ks == 0)
#pragma unroll
            for (int i = 0; i < kV; ++i) {
                v[i] = 0.f;
                for (int k = 0; k < KS; ++k) v[i] += part[wave + k][i];
            }
    }
    if (lane == 0 && ks == 0 && n0 < R) {
        float iv[kH];
#pragma unroll
        for (int s = 0; s < kH; ++s) iv[s] = 1.0f / sqrtf(v[RW * kH + s] / (float)d + eps);  // correctly rounded
#pragma unroll
        for (int rw = 0; rw < RW; ++rw) {
            const int64_t n = n0 + rw;
            if (n >= R) break;
            float a = 0.f;
#pragma unroll
            for (int s = 0; s < kH; ++s) a = fmaf(iv[s], v[rw * kH + s], a);
            const float z = a / (float)kH;
            if (n < r) h[t * r + n] = z / (1.0f + expf(-z));
            else w_in[t * kH + (n - r)] = 2.0f / (1.0f + expf(-z));
        }
        if (n0 == 0)
#pragma unroll
            for (int s = 0; s < kH; ++s) inv[t * kH + s] = iv[s];
    }
}

// hc_mix_up: a block covers kUpJ outputs j x the H streams for MT tokens; thread (s, jl) owns row s*d + j of W_up
// (the threads of a wave take consecutive rows of one stream, adjacent in the chunk-major layout, so each 16-byte
// load instruction is one contiguous run). It loads each weight chunk once and dots it with every token's h (in
// LDS; each chunk's sum of h, the same for every row, once per block), gates its stream's normalized value with the
// sigmoid, and the H streams of each j meet in LDS. A token's math doesn't depend on MT. At T = 5 (Q8, the served
// verify): 27.2 us with a block per token -> 25.5 us.
constexpr int kUpJ = kThreads / kH;  // 64

template <typename T, int G, int BITS, int MT>
__global__ void __launch_bounds__(kThreads)
    hc_up_kernel(const float *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const T *x,
                 const float *norm_w, const float *inv, int64_t d, int r, int64_t Tn, T *u) {
    __shared__ float hs[MT][kMaxR];
    __shared__ float contrib[MT][kH][kUpJ];
    const int64_t t0 = (int64_t)blockIdx.y * MT;
    const int nt = (int)(Tn - t0 < MT ? Tn - t0 : MT);  // tokens in this block, 1..MT
    __shared__ float hsums[MT][kMaxR / kChunk];  // each chunk's sum of h, the same for every row
    for (int tt = 0; tt < nt; ++tt)
        for (int i = threadIdx.x; i < r; i += blockDim.x) hs[tt][i] = h[(t0 + tt) * r + i];
    __syncthreads();
    for (int i = threadIdx.x; i < nt * (r / kChunk); i += blockDim.x) {
        const int tt = i / (r / kChunk), c = i % (r / kChunk);
        float hsum = 0.f;
        for (int k = 0; k < kChunk; ++k) hsum += hs[tt][c * kChunk + k];
        hsums[tt][c] = hsum;
    }
    __syncthreads();
    const int s = threadIdx.x / kUpJ, jl = threadIdx.x % kUpJ;
    const int64_t j = (int64_t)blockIdx.x * kUpJ + jl, N = kH * d;
    if (j < d) {
        const int64_t row = s * d + j;
        float acc[MT];
#pragma unroll
        for (int tt = 0; tt < MT; ++tt) acc[tt] = 0.f;
        // The next chunk's weights load while this one computes: a thread has only r / 32 chunks, loaded in turn
        // otherwise (T = 5: 27.1 -> 26.0 us, T = 1 unchanged; hc_mix_down didn't gain from the same).
        uint32_t nwords[BITS];
        float nsc, nmn;
        auto fetch = [&](int c) {
            load_chunk<BITS>(q + ((int64_t)c * N + row) * (kChunk * BITS / 8), nwords);
            const int64_t gi = (int64_t)(c * kChunk / G) * N + row;
            nsc = bf16_bits_to_f32(scale[gi]), nmn = bf16_bits_to_f32(minv[gi]);
        };
        fetch(0);
        for (int c = 0; c < r / kChunk; ++c) {
            uint32_t words[BITS];
#pragma unroll
            for (int i = 0; i < BITS; ++i) words[i] = nwords[i];
            const float sc = nsc, mn = nmn;
            if (c + 1 < r / kChunk) fetch(c + 1);
#pragma unroll
            for (int tt = 0; tt < MT; ++tt) {
                if (tt >= nt) break;
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kChunk; ++i) dot = fmaf(hs[tt][c * kChunk + i], code<BITS>(words, i), dot);
                acc[tt] = fmaf(sc, dot, fmaf(mn, hsums[tt][c], acc[tt]));
            }
        }
        const float nw = 1.0f + norm_w[row];
#pragma unroll
        for (int tt = 0; tt < MT; ++tt) {
            if (tt >= nt) break;
            const int64_t t = t0 + tt;
            const float z = 1.0f / (1.0f + expf(-acc[tt]));
            contrib[tt][s][jl] = z * (load(x, t * N + row) * inv[t * kH + s] * nw);
        }
    }
    __syncthreads();
    if (s == 0 && j < d)
        for (int tt = 0; tt < nt; ++tt) {
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < kH; ++k) sum += contrib[tt][k][jl];
            store(u, (t0 + tt) * d + j, sum / (float)kH);
        }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    hc_inject_kernel(T *x, const float *w_in, const T *y, int64_t n_total, int64_t d) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_total; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / (kH * d), rem = i % (kH * d), s = rem / d, j = rem % d;
        // One FP32 rounding (fma), then the activation dtype's - the reference's x + w*y exactly.
        store(x, i, fmaf(w_in[t * kH + s], load(y, t * d + j), load(x, i)));
    }
}

// Rows per wave (RW) and waves per row group (KS) for hc_mix_down, swept at the target's shape (324 x 10240,
// Q4 G64; RW 1/2/4 x KS 1/2/4/8): T = 1 best at RW 1, KS 2
// (13.2 us, 62% of read peak - the ~1.9 MB matrix is latency-bound like linear_q4's of that size); T = 4 at
// RW 2, KS 1 (21.3 us), where sharing each X load across rows starts to pay. Sharing X didn't help at T = 1.
int split_k(int64_t T) { return T == 1 ? 2 : 1; }
int rows_per_wave(int64_t T) { return T == 1 ? 1 : 2; }

// Tokens per block (MT) for hc_mix_up: every token of a block shares each weight load. The instantiated sizes; a
// verify is T = drafts + 1 = 5; above 8 the tokens go in blocks of 8 (a partial last block).
int tokens_per_block(int64_t T) { return T == 1 ? 1 : T == 2 ? 2 : T <= 4 ? 4 : T == 5 ? 5 : 8; }

// A weight's arrays whichever format it's in (the kernels take raw pointers + BITS).
struct QArrays {
    const uint8_t *q;
    const uint16_t *scale, *minv;
    int64_t N, K, G;
};

void check_weight(const char *fn, const QArrays &w, int bits) {
    STRIX_CHECK(w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn, ": null Q", bits,
                " weight pointer (q=", (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.K >= w.G && w.K % w.G == 0, fn, ": K = ", w.K, " must be a positive multiple of group size ", w.G);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, fn, ": code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
}

void check_shape(const char *fn, int64_t T, int64_t H, int64_t d) {
    STRIX_CHECK(H == kH, fn, ": H = ", H, " streams, only ", kH, " (qwen4_exp) is supported");
    STRIX_CHECK(T >= 1 && T <= kMaxTokens, fn, ": T = ", T, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(d >= kChunk && d <= kMaxD && d % kChunk == 0, fn, ": d = ", d, ", expected a multiple of ", kChunk,
                " in ", kChunk, "..", kMaxD);
}

size_t act_size(const char *fn, Act act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    return act == Act::F32 ? 4 : 2;
}

template <int BITS>
void mix_down(const void *x, const QArrays &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject, float eps,
              float *h, float *w_in, float *inv, Act act, hipStream_t stream) {
    const char *fn = "hc_mix_down";
    check_shape(fn, T, H, d);
    check_weight(fn, w, BITS);
    act_size(fn, act);
    STRIX_CHECK(x != nullptr && h != nullptr && inv != nullptr, fn, ": null pointer (x=", x, ", h=", (void *)h,
                ", inv=", (void *)inv, ")");
    STRIX_CHECK(inject == (w_in != nullptr), fn, ": w_in must be given exactly when inject (inject=", inject,
                ", w_in=", (void *)w_in, ")");
    STRIX_CHECK(r >= 1 && r <= kMaxR, fn, ": r = ", r, ", expected 1..", kMaxR);
    STRIX_CHECK(w.K == H * d, fn, ": weight K = ", w.K, " != H*d = ", H * d);
    const int64_t R = r + (inject ? H : 0);
    STRIX_CHECK(w.N == R, fn, ": weight has ", w.N, " rows, expected r", inject ? " + H" : "", " = ", R);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": x ", x, " must be 16-byte aligned");
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps must be finite and > 0, got ", eps);
    const int waves = kThreads / device_warp_size(kThreads), KS = split_k(T);
    STRIX_CHECK(waves % KS == 0, fn, ": split-K factor ", KS, " must divide the ", waves, " waves of a block");
    const int RW = rows_per_wave(T);
    const size_t lds = (size_t)(w.K / kChunk) * sizeof(float2);  // the chunk sums
    STRIX_CHECK(lds <= kMaxDownLds, fn, ": K = ", w.K, " needs ", lds, " bytes of LDS for its chunk sums, at most ",
                kMaxDownLds, " (H*d up to ", kMaxDownLds / sizeof(float2) * kChunk, ")");
    const int64_t rows_per_block = waves / KS * RW;
    dim3 grid((unsigned)((R + rows_per_block - 1) / rows_per_block), (unsigned)T), block(kThreads);
#define STRIX_HC_DOWN_RW(TT, G, RWV)                                                                               \
    hc_down_kernel<TT, G, RWV, BITS><<<grid, block, lds, stream>>>(static_cast<const TT *>(x), w.q, w.scale, w.minv, \
                                                                R, w.K, d, (int)r, eps, KS, h, w_in, inv)
#define STRIX_HC_DOWN(TT, G)                                                                    \
    switch (RW) {                                                                               \
        case 1: STRIX_HC_DOWN_RW(TT, G, 1); break;                                              \
        default: STRIX_HC_DOWN_RW(TT, G, 2); break;                                             \
    }
#define STRIX_HC_DOWN_G(TT)                                                                     \
    switch (w.G) {                                                                              \
        case 32: STRIX_HC_DOWN(TT, 32); break;                                                  \
        case 64: STRIX_HC_DOWN(TT, 64); break;                                                  \
        default: STRIX_HC_DOWN(TT, 128); break;                                                 \
    }
    if (act == Act::F32) STRIX_HC_DOWN_G(float) else STRIX_HC_DOWN_G(uint16_t)
#undef STRIX_HC_DOWN_G
#undef STRIX_HC_DOWN
#undef STRIX_HC_DOWN_RW
    STRIX_HIP_CHECK_LAUNCH(fn, " Q", BITS, " ", act_name(act), " T=", T, " R=", R, " K=", w.K, " G=", w.G);
}

template <int BITS>
void mix_up(const float *h, const QArrays &w_up, const void *x, const float *norm_w, const float *inv, int64_t T,
            int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    const char *fn = "hc_mix_up";
    check_shape(fn, T, H, d);
    check_weight(fn, w_up, BITS);
    const size_t es = act_size(fn, act);
    STRIX_CHECK(h != nullptr && x != nullptr && norm_w != nullptr && inv != nullptr && u != nullptr, fn,
                ": null pointer (h=", (const void *)h, ", x=", x, ", norm_w=", (const void *)norm_w,
                ", inv=", (const void *)inv, ", u=", u, ")");
    STRIX_CHECK(w_up.N == H * d, fn, ": W_up has ", w_up.N, " rows, expected H*d = ", H * d);
    STRIX_CHECK(w_up.K >= kChunk && w_up.K <= kMaxR && w_up.K % kChunk == 0, fn, ": W_up K (the rank r) = ", w_up.K,
                ", expected a multiple of ", kChunk, " up to ", kMaxR);
    auto ub = reinterpret_cast<uintptr_t>(u), xb = reinterpret_cast<uintptr_t>(x);
    STRIX_CHECK(ub + (uintptr_t)(T * d) * es <= xb || xb + (uintptr_t)(T * H * d) * es <= ub, fn,
                ": u overlaps the streams x");
    const int MT = tokens_per_block(T);
    dim3 grid((unsigned)((d + kUpJ - 1) / kUpJ), (unsigned)((T + MT - 1) / MT)), block(kThreads);
#define STRIX_HC_UP_MT(TT, G, MTV)                                                                                 \
    hc_up_kernel<TT, G, BITS, MTV><<<grid, block, 0, stream>>>(h, w_up.q, w_up.scale, w_up.minv,                   \
                                                              static_cast<const TT *>(x), norm_w, inv, d,           \
                                                              (int)w_up.K, T, static_cast<TT *>(u))
#define STRIX_HC_UP(TT, G)                                                                      \
    switch (MT) {                                                                               \
        case 1: STRIX_HC_UP_MT(TT, G, 1); break;                                                \
        case 2: STRIX_HC_UP_MT(TT, G, 2); break;                                                \
        case 4: STRIX_HC_UP_MT(TT, G, 4); break;                                                \
        case 5: STRIX_HC_UP_MT(TT, G, 5); break;                                                \
        default: STRIX_HC_UP_MT(TT, G, 8); break;                                               \
    }
#define STRIX_HC_UP_G(TT)                                                                       \
    switch (w_up.G) {                                                                           \
        case 32: STRIX_HC_UP(TT, 32); break;                                                    \
        case 64: STRIX_HC_UP(TT, 64); break;                                                    \
        default: STRIX_HC_UP(TT, 128); break;                                                   \
    }
    if (act == Act::F32) STRIX_HC_UP_G(float) else STRIX_HC_UP_G(uint16_t)
#undef STRIX_HC_UP_G
#undef STRIX_HC_UP
#undef STRIX_HC_UP_MT
    STRIX_HIP_CHECK_LAUNCH(fn, " Q", BITS, " ", act_name(act), " T=", T, " d=", d, " r=", w_up.K, " G=", w_up.G,
                           " MT=", MT);
}

}  // namespace

void hc_mix_down(const void *x, const Q4DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                 float eps, float *h, float *w_in, float *inv, Act act, hipStream_t stream) {
    mix_down<4>(x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, T, H, d, r, inject, eps, h, w_in, inv, act, stream);
}
void hc_mix_down(const void *x, const Q8DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                 float eps, float *h, float *w_in, float *inv, Act act, hipStream_t stream) {
    mix_down<8>(x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, T, H, d, r, inject, eps, h, w_in, inv, act, stream);
}

void hc_mix_up(const float *h, const Q4ChunkMajorView &w, const void *x, const float *norm_w, const float *inv,
               int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    mix_up<4>(h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, x, norm_w, inv, T, H, d, u, act, stream);
}
void hc_mix_up(const float *h, const Q8ChunkMajorView &w, const void *x, const float *norm_w, const float *inv,
               int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    mix_up<8>(h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, x, norm_w, inv, T, H, d, u, act, stream);
}

void hc_inject(void *x, const float *w_in, const void *y, int64_t T, int64_t H, int64_t d, Act act,
               hipStream_t stream) {
    const char *fn = "hc_inject";
    check_shape(fn, T, H, d);
    const size_t es = act_size(fn, act);
    STRIX_CHECK(x != nullptr && w_in != nullptr && y != nullptr, fn, ": null pointer (x=", x, ", w_in=",
                (const void *)w_in, ", y=", y, ")");
    auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(T * d) * es <= xb || xb + (uintptr_t)(T * H * d) * es <= yb, fn,
                ": y overlaps the streams x");
    const int64_t n = T * H * d;
    const unsigned blocks = (unsigned)std::min<int64_t>((n + kThreads - 1) / kThreads, 4096);
    if (act == Act::F32)
        hc_inject_kernel<float><<<blocks, kThreads, 0, stream>>>(static_cast<float *>(x), w_in,
                                                                 static_cast<const float *>(y), n, d);
    else
        hc_inject_kernel<uint16_t><<<blocks, kThreads, 0, stream>>>(static_cast<uint16_t *>(x), w_in,
                                                                    static_cast<const uint16_t *>(y), n, d);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", T, " d=", d);
}

}  // namespace strix::kernels
