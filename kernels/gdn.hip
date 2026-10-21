#include "kernels/gdn.hpp"

#include "common/hip_check.hpp"
#include "kernels/wmma.hpp"

#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
// Tokens per conv thread. Decode is one tile; prefill spreads over T/kTile
// tiles per channel so a long prompt isn't one serial loop per channel.
constexpr int64_t kTile = 16;
constexpr int64_t kW = kGdnConvWidth, kS = kGdnConvWidth - 1;  // taps, state rows
static_assert(kS == 3, "the state write below selects among exactly 3 old values");
constexpr int64_t kMaxT = 1 << 24;
constexpr int64_t kMaxChannels = 1 << 20;  // far above any real C; catches a shape mix-up

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <typename T> __device__ inline float load(const T *p, int64_t i);
template <> __device__ inline float load<float>(const float *p, int64_t i) { return p[i]; }
template <> __device__ inline float load<uint16_t>(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }

template <typename T> __device__ inline void store(T *p, int64_t i, float v);
template <> __device__ inline void store<float>(float *p, int64_t i, float v) { p[i] = v; }
template <> __device__ inline void store<uint16_t>(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }  // same form as the other kernels

struct Params {
    int64_t T, C, in_ts, b_col, a_col, Hv, tiles, conv_blocks;
    bool reset;
};

// Blocks [0, conv_blocks): one thread per (channel, token tile), channels
// fastest so a wave's loads are contiguous. Blocks after that: one thread per
// (token, value head) gate pair.
//
// In place (state_in == state_out) is race-free: only tile 0 reads the old
// state, and the same thread writes the new one after it has read it. When
// T >= 3 the new state is just the last 3 inputs; when T < 3 there is only
// one tile, which holds the old state it needs in registers.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    gdn_front_kernel(const T *in, const float *conv_w, const T *state_in, T *state_out, const float *A_log, const float *dt_bias, T *out,
                     float *beta, float *g, Params p) {
    if ((int64_t)blockIdx.x < p.conv_blocks) {
        const int64_t cblocks = (p.C + kThreads - 1) / kThreads;
        const int64_t c = (int64_t)(blockIdx.x % cblocks) * kThreads + threadIdx.x;
        const int64_t tile = blockIdx.x / cblocks;
        if (c >= p.C) return;
        const int64_t t0 = tile * kTile, t1 = t0 + kTile < p.T ? t0 + kTile : p.T;
        float w[kW];
#pragma unroll
        for (int j = 0; j < kW; ++j) w[j] = conv_w[c * kW + j];
        // x[t0-3 .. t0-1]: from the state before token 0, else from the input.
        float x[kW];
#pragma unroll
        for (int j = 0; j < kS; ++j) {
            const int64_t i = t0 - kS + j;
            x[j] = i >= 0 ? load(in, i * p.in_ts + c) : p.reset ? 0.0f : load(state_in, (kS + i) * p.C + c);
        }
        float old[kS];  // tile 0's view of the old state (x[-3..-1])
#pragma unroll
        for (int j = 0; j < kS; ++j) old[j] = x[j];
        for (int64_t t = t0; t < t1; ++t) {
            x[kS] = load(in, t * p.in_ts + c);
            float y = 0.0f;
#pragma unroll
            for (int j = 0; j < kW; ++j) y = fmaf(w[j], x[j], y);
            store(out, t * p.C + c, silu(y));
#pragma unroll
            for (int j = 0; j < kS; ++j) x[j] = x[j + 1];
        }
        if (tile == 0) {
#pragma unroll
            for (int r = 0; r < kS; ++r) {
                const int64_t i = p.T - kS + r;  // input index held in state row r
                // old[kS + i] with a runtime index would put old[] in scratch; select instead.
                const int64_t k = kS + i;
                const float from_old = k == 0 ? old[0] : k == 1 ? old[1] : old[2];
                store(state_out, r * p.C + c, i >= 0 ? load(in, i * p.in_ts + c) : from_old);
            }
        }
        return;
    }
    const int64_t i = (int64_t)(blockIdx.x - p.conv_blocks) * kThreads + threadIdx.x;
    if (i >= p.T * p.Hv) return;
    const int64_t t = i / p.Hv, h = i % p.Hv;
    const float bv = load(in, t * p.in_ts + p.b_col + h), av = load(in, t * p.in_ts + p.a_col + h);
    beta[i] = 1.0f / (1.0f + expf(-bv));
    const float x = av + dt_bias[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    g[i] = -expf(A_log[h]) * sp;
}

// Delta rule: one block per (value head, 32 state columns). Thread (ks, vl)
// holds S[ks*16 .. ks*16+15][v0 + vl] in registers for all T tokens, so the
// state is read once and written once per call. Per token: the block
// L2-normalizes q and k of its key head into LDS; partial dot products over
// each thread's 16 k rows are summed across the 8 k slices in LDS (kv = S^T k
// for the delta, then S^T q for the output). Wave-size agnostic: the norms
// use warpSize shuffles + per-wave partials, everything else goes via LDS.
constexpr int kDrCols = 32, kDrSlices = kThreads / kDrCols, kDrRows = (int)kGdnHeadDim / kDrSlices;
static_assert(kDrSlices * kDrRows == kGdnHeadDim && kDrCols * kDrSlices == kThreads, "delta rule tiling");
static_assert(2 * kGdnHeadDim == kThreads, "one thread per q and k element for the L2 norms");
constexpr int kMaxWaves = kThreads / 32;  // warpSize >= 32

struct DrParams {
    int64_t T, Hk, Hv, ts;
    bool reset;
};

template <typename T>
__global__ void __launch_bounds__(kThreads)
    delta_rule_kernel(const T *qkv, const float *beta, const float *g, const float *state_in, float *state_out, T *out,
                      DrParams p) {
    constexpr int64_t D = kGdnHeadDim;
    const int64_t col_blocks = D / kDrCols;
    const int64_t h = blockIdx.x / col_blocks, v0 = (blockIdx.x % col_blocks) * kDrCols;
    const int64_t kh = h / (p.Hv / p.Hk);
    const int tid = threadIdx.x, vl = tid % kDrCols, ks = tid / kDrCols;
    const int lane = tid % warpSize, wave = tid / warpSize, waves_per_vec = (int)D / warpSize;

    __shared__ float qn[D], kn[D];                 // normalized q (scaled) and k of this token
    __shared__ float red_kv[kDrSlices][kDrCols];   // partial S^T k
    __shared__ float red_o[kDrSlices][kDrCols];    // partial S^T q (separate: token t+1's kv overlaps t's reads)
    __shared__ float delta[kDrCols];
    __shared__ float ss[kMaxWaves];                // per-wave sums of squares

    float S[kDrRows];
    const int64_t sbase = h * D * D + (int64_t)ks * kDrRows * D + v0 + vl;
#pragma unroll
    for (int r = 0; r < kDrRows; ++r) S[r] = p.reset ? 0.0f : state_in[sbase + r * D];

    const float q_scale = rsqrtf((float)D);
    // This thread's inputs for token t: its q or k element (threads 0..127 / 128..255), and for
    // threads < 32 the v element, beta; g for all. Token t+1's are loaded at the top of token t's
    // step, so their global-memory latency hides behind t's barriers instead of adding to them.
    const int64_t x_off = tid < D ? kh * D + tid : p.Hk * D + kh * D + (tid - D);
    const int64_t v_off = 2 * p.Hk * D + h * D + v0 + (tid % kDrCols);
    auto fetch = [&](int64_t t, float &x, float &v, float &b, float &gg) {
        x = load(qkv, t * p.ts + x_off);
        gg = g[t * p.Hv + h];
        if (tid < kDrCols) {
            v = load(qkv, t * p.ts + v_off);
            b = beta[t * p.Hv + h];
        }
    };
    float x_n = 0.0f, v_n = 0.0f, b_n = 0.0f, g_n = 0.0f;
    fetch(0, x_n, v_n, b_n, g_n);
    for (int64_t t = 0; t < p.T; ++t) {
        const float x = x_n, v = v_n, b = b_n, gt = g_n;
        if (t + 1 < p.T) fetch(t + 1, x_n, v_n, b_n, g_n);
        // q and k of key head kh, L2-normalized in FP32.
        float sq = x * x;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down(sq, off, 32);
        if (lane == 0) ss[wave] = sq;
        __syncthreads();
        float tot = 0.0f;
        const int w0 = tid < D ? 0 : waves_per_vec;
        for (int w = 0; w < waves_per_vec; ++w) tot += ss[w0 + w];
        if (tid < D) qn[tid] = x * (rsqrtf(tot + 1e-6f) * q_scale);
        else kn[tid - D] = x * rsqrtf(tot + 1e-6f);
        __syncthreads();

        // Decay, then this thread's share of kv = S^T k.
        const float decay = expf(gt);
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < kDrRows; ++r) {
            S[r] *= decay;
            acc = fmaf(S[r], kn[ks * kDrRows + r], acc);
        }
        red_kv[ks][vl] = acc;
        __syncthreads();
        if (tid < kDrCols) {
            float kv = 0.0f;
            for (int j = 0; j < kDrSlices; ++j) kv += red_kv[j][tid];
            delta[tid] = b * (v - kv);
        }
        __syncthreads();

        // S += k delta^T, then this thread's share of out = S^T q.
        const float d = delta[vl];
        acc = 0.0f;
#pragma unroll
        for (int r = 0; r < kDrRows; ++r) {
            S[r] = fmaf(kn[ks * kDrRows + r], d, S[r]);
            acc = fmaf(S[r], qn[ks * kDrRows + r], acc);
        }
        red_o[ks][vl] = acc;
        __syncthreads();
        if (tid < kDrCols) {
            float o = 0.0f;
            for (int j = 0; j < kDrSlices; ++j) o += red_o[j][tid];
            store(out, (t * p.Hv + h) * D + v0 + tid, o);
        }
        // No barrier needed here: the next token writes ss/qn/kn/red_kv only after its first
        // barrier, and red_o (read just above) is next written after three more barriers.
    }
#pragma unroll
    for (int r = 0; r < kDrRows; ++r) state_out[sbase + r * D] = S[r];
}

// Chunked delta rule (the matrix form in kernels/gdn.hpp), one block per
// (value head, 32 state columns) like the per-token kernel - every term of the
// chunk form splits by state column. The block's state slice S [128][32]
// lives in LDS for the whole call; per 32-token chunk it stages normalized K
// and Q (rows padded to 129 floats: the dot products read rows at a lane-
// varying stride), A, P = e^(G_t - G_s) q_t.k_s (s <= t) and U. Missing
// tokens of a last partial chunk are zero rows with beta = 0, so their
// corrections are zero; the state decays by the last real token's G.
// Stages per chunk: load + L2-normalize Q, K; G = cumsum(g); R = beta (V -
// e^G K S) into U; A, P; forward substitution (I + A) U = R, one thread per
// column, U column in registers; O = e^G Q S + P U; S = e^(G_last) S +
// (e^(G_last - G_s) k_s)^T U.
constexpr int kCk = 32;                    // chunk length
constexpr int kCkCols = 32;                // state columns per block
constexpr int kPad = (int)kGdnHeadDim + 1; // padded Q/K row
static_assert(kCk * 8 == kThreads, "Q/K load: 8 threads per row");
static_assert(kCk * kCkCols == 4 * kThreads, "4 (t, j) outputs per thread");

template <typename T>
__global__ void __launch_bounds__(kThreads)
    delta_rule_chunked_kernel(const T *qkv, const float *beta, const float *g, const float *state_in,
                              float *state_out, T *out, DrParams p) {
    constexpr int D = (int)kGdnHeadDim;
    const int64_t col_blocks = D / kCkCols;
    const int64_t h = blockIdx.x / col_blocks, v0 = (blockIdx.x % col_blocks) * kCkCols;
    const int64_t kh = h / (p.Hv / p.Hk);
    const int tid = threadIdx.x, j = tid % kCkCols, t_base = tid / kCkCols;  // (t, j) outputs: t = t_base + 8 i

    __shared__ float S[D][kCkCols];
    __shared__ float K[kCk][kPad], Q[kCk][kPad];
    __shared__ float A[kCk][kCk + 1], P[kCk][kCk + 1], U[kCk][kCkCols];
    __shared__ float G[kCk], eG[kCk], B[kCk], eL[kCk];  // eL[s] = e^(G_last - G_s)

    for (int i = tid; i < D * kCkCols; i += kThreads) {
        const int k = i / kCkCols, c = i % kCkCols;
        S[k][c] = p.reset ? 0.0f : state_in[h * D * D + (int64_t)k * D + v0 + c];
    }
    const float q_scale = rsqrtf((float)D);
    for (int64_t c0 = 0; c0 < p.T; c0 += kCk) {
        const int n = p.T - c0 < kCk ? (int)(p.T - c0) : kCk;
        // Q, K rows: thread (row, part) loads 16 elements of q and of k, 8 threads per row.
        {
            const int r = tid / 8, part = tid % 8;
            float qv[16], kv[16], sq = 0.0f, sk = 0.0f;
            const int64_t row = (c0 + r) * p.ts;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int d = part * 16 + e;
                qv[e] = r < n ? load(qkv, row + kh * D + d) : 0.0f;
                kv[e] = r < n ? load(qkv, row + p.Hk * D + kh * D + d) : 0.0f;
                sq = fmaf(qv[e], qv[e], sq);
                sk = fmaf(kv[e], kv[e], sk);
            }
            for (int off = 1; off < 8; off <<= 1) {  // the 8 threads of a row are adjacent lanes
                sq += __shfl_xor(sq, off);
                sk += __shfl_xor(sk, off);
            }
            const float iq = rsqrtf(sq + 1e-6f) * q_scale, ik = rsqrtf(sk + 1e-6f);
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                Q[r][part * 16 + e] = qv[e] * iq;
                K[r][part * 16 + e] = kv[e] * ik;
            }
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < kCk; ++t) {
                if (t < n) acc += g[(c0 + t) * p.Hv + h];  // padding rows keep the last real G
                G[t] = acc;
                eG[t] = expf(acc);
                B[t] = t < n ? beta[(c0 + t) * p.Hv + h] : 0.0f;
            }
            for (int t = 0; t < kCk; ++t) eL[t] = expf(acc - G[t]);
        }
        __syncthreads();

        // R = beta (V - e^G K S) into U; A and P.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t_base + 8 * i;
            float w = 0.0f, a = 0.0f, pq = 0.0f;
            for (int k = 0; k < D; ++k) {
                w = fmaf(K[t][k], S[k][j], w);
                a = fmaf(K[t][k], K[j][k], a);   // j doubles as s for the [32][32] matrices
                pq = fmaf(Q[t][k], K[j][k], pq);
            }
            const float v = t < n ? load(qkv, (c0 + t) * p.ts + 2 * p.Hk * D + h * D + v0 + j) : 0.0f;
            U[t][j] = B[t] * (v - eG[t] * w);
            const float decay = j <= t ? expf(G[t] - G[j]) : 0.0f;
            A[t][j] = j < t ? B[t] * decay * a : 0.0f;
            P[t][j] = decay * pq;
        }
        __syncthreads();

        // (I + A) U = R, forward substitution: thread j owns column j.
        if (tid < kCkCols) {
            float u[kCk];
#pragma unroll
            for (int t = 0; t < kCk; ++t) {
                float x = U[t][tid];
#pragma unroll
                for (int s2 = 0; s2 < t; ++s2) x = fmaf(-A[t][s2], u[s2], x);
                u[t] = x;
            }
#pragma unroll
            for (int t = 0; t < kCk; ++t) U[t][tid] = u[t];
        }
        __syncthreads();

        // O = e^G Q S + P U.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t_base + 8 * i;
            if (t >= n) continue;
            float qs = 0.0f, pu = 0.0f;
            for (int k = 0; k < D; ++k) qs = fmaf(Q[t][k], S[k][j], qs);
            for (int s2 = 0; s2 <= t; ++s2) pu = fmaf(P[t][s2], U[s2][j], pu);
            store(out, ((c0 + t) * p.Hv + h) * D + v0 + j, eG[t] * qs + pu);
        }
        __syncthreads();  // everyone has read S

        // S = e^(G_last) S + (e^(G_last - G_s) k_s)^T U; thread owns rows k = t_base + 8 i (16 of them), column j.
        const float e_last = eG[kCk - 1];
        for (int i = 0; i < D / 8; ++i) {
            const int k = t_base + 8 * i;
            float acc = e_last * S[k][j];
            for (int s2 = 0; s2 < kCk; ++s2) acc = fmaf(eL[s2] * K[s2][k], U[s2][j], acc);
            S[k][j] = acc;
        }
        __syncthreads();  // S complete before the next chunk reads it; Q/K/U free to overwrite
    }
    for (int i = tid; i < D * kCkCols; i += kThreads) {
        const int k = i / kCkCols, c = i % kCkCols;
        state_out[h * D * D + (int64_t)k * D + v0 + c] = S[k][c];
    }
}

// ---- chunked delta rule on the matrix units ----------------------------------------------------------------
//
// The chunk form above, per value head in one block of kWgWaves waves; wave w owns state columns [16w, 16w + 16).
// Its slice of S [128 k][16 v] (FP32) stays in WMMA accumulator registers for the whole call: 8 tiles of 16 k rows.
// Per 32-token chunk:
//   stage   Q, K rows (L2-normalized, FP32) as BF16 hi + lo in LDS; G = cumsum(g), e^G, beta, e^(G_last - G);
//   A, P    = K K^T, Q K^T (one 16 x 16 tile per wave), masked / decayed as the scalar kernel; A to LDS (FP32), P as
//           BF16 hi / lo;
//   T       = (I + A)^-1 (unit lower triangular, one column per thread of wave 0) as BF16 hi / lo;
//   W, QS   = K S, Q S from the accumulators (S tiles turned into B fragments with a cross-half lane exchange);
//   U       = T (beta (V - e^G W));   O = e^G QS + P U  (stored);
//   S       = e^(G_last) S + (e^(G_last - G) K)^T U  (the transposed, scaled K staged over Q's LDS).
// Every product is a hi / lo split on both operands (hi hi + hi lo + lo hi: ~2^-16 of each product) with FP32
// accumulation, so the state, which carries over the whole sequence, stays close to the FP32 recurrence.
constexpr int kWgWaves = 8;        // 8 x 16 state columns = 128
constexpr int kWgC = 32;           // chunk length
constexpr int kWgLd = 128 + 8;     // [t][k] rows (BF16), 16-byte aligned
constexpr int kWgLdT = kWgC + 8;   // [t][s] rows (BF16), 16-byte aligned
constexpr int kWgLdKT = kWgC + 2;  // K~^T [k][t] rows (BF16): 128 x 34 fills a Q buffer (32 x 136) exactly; even (32-bit
                                   // fragment reads), 17 words (the 16 rows of a fragment in distinct banks)
static_assert(128 * kWgLdKT <= kWgC * kWgLd, "K~^T fits the Q buffer it reuses");
static_assert(kWgWaves * 16 == (int)kGdnHeadDim, "8 waves x 16 columns = the head dim");

// The value of the lane 16 away (lanes 0-15 <-> 16-31) with v_permlanex16 - a VALU op, where __shfl_xor(x, 16) is a
// ds_bpermute through the LDS crossbar.
__device__ inline float wg_xor16(float x) {
    const int i = __float_as_int(x);
    return __int_as_float(__builtin_amdgcn_permlanex16(i, i, 0x76543210u, 0xfedcba98u, true, false));
}
__device__ inline uint16_t wg_bf16(float f) {  // round to nearest even (finite values)
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ inline float wg_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
// One = false (gdn-precision exact): f as BF16 hi + lo. One = true (fast): hi only - the lo halves are neither
// written nor read, and their LDS buffers are not allocated.
template <bool One, bool F16> __device__ inline void wg_split(float f, uint16_t &hi, uint16_t &lo) {
    // F16 (half, half-u): F16 instead of BF16 - One = true one F16 term, One = false an F16 hi + lo pair (11 + 11 bits,
    // half-u's U-path products).
    if constexpr (F16) {
        const _Float16 h = (_Float16)f;
        hi = __builtin_bit_cast(uint16_t, h);
        if constexpr (!One) lo = __builtin_bit_cast(uint16_t, (_Float16)(f - (float)h));
        else (void)lo;
        return;
    }
    hi = wg_bf16(f);
    if constexpr (!One) lo = wg_bf16(f - wg_f32(hi));
    else (void)lo;
}
// C tile (16 x 16 FP32, element v = row 2v + lane / 16, column lane % 16) -> the B fragment of its transpose view:
// column lane % 16 with rows 0..15 as the K run (lanes 16..31 the same), as hi and lo BF16.
// sc: a factor applied first (half: the operand's power-of-two scale, so F16 holds it; 1 otherwise).
template <bool One, bool F16>
__device__ inline void wg_c_to_b(const wmma_f32x8 &c, wmma_bf16x16 &hi, wmma_bf16x16 &lo, float sc = 1.0f) {
    const bool upper = (threadIdx.x % 32) >= 16;
    wmma_u32x8 h, l;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float own = F16 ? c[v] * sc : c[v], other = wg_xor16(own);
        const float even = upper ? other : own, odd = upper ? own : other;  // rows 2v, 2v + 1
        uint16_t eh, el, oh, ol;
        wg_split<One, F16>(even, eh, el);
        wg_split<One, F16>(odd, oh, ol);
        h[v] = (uint32_t)eh | ((uint32_t)oh << 16);
        l[v] = (uint32_t)el | ((uint32_t)ol << 16);
    }
    hi = __builtin_bit_cast(wmma_bf16x16, h);
    if constexpr (!One) lo = __builtin_bit_cast(wmma_bf16x16, l);
    else (void)l;
}
// acc += a b: One = false with hi / lo terms (the lo x lo term dropped: ~2^-18); One = true hi x hi only.
template <bool One, bool F16>
__device__ inline wmma_f32x8 wg_mma3(const wmma_bf16x16 &ah, const wmma_bf16x16 &al, const wmma_bf16x16 &bh,
                                     const wmma_bf16x16 &bl, wmma_f32x8 acc) {
    if constexpr (F16) {
        acc = wmma_f16(__builtin_bit_cast(wmma_f16x16, ah), __builtin_bit_cast(wmma_f16x16, bh), acc);
        if constexpr (One) {
            (void)al, (void)bl;
            return acc;
        } else {
            acc = wmma_f16(__builtin_bit_cast(wmma_f16x16, ah), __builtin_bit_cast(wmma_f16x16, bl), acc);
            return wmma_f16(__builtin_bit_cast(wmma_f16x16, al), __builtin_bit_cast(wmma_f16x16, bh), acc);
        }
    }
    acc = wmma_bf16(ah, bh, acc);
    if constexpr (One) {
        (void)al, (void)bl;
        return acc;
    } else {
        acc = wmma_bf16(ah, bl, acc);
        return wmma_bf16(al, bh, acc);
    }
}
// The lo half of a fragment from LDS; One = true never reads it.
template <bool One, bool F16> __device__ inline wmma_bf16x16 wg_load_lo(const uint16_t *p, int ld) {
    if constexpr (One) return wmma_bf16x16{};
    else return wmma_load_rows(p, ld);
}

// half: an operand's power-of-two exponent e so that its largest |value| x 2^-e lies in [2^14, 2^15) - well inside
// F16's range (65504) with its full precision; the product is scaled back by 2^e (exact: powers of two). m: the wave's
// largest |value| (wg_wave_max); 0 for an all-zero operand.
__device__ inline float wg_wave_max(float m) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 32));
    return m;
}
__device__ inline int wg_exp(float m) { return m > 0.0f ? ilogbf(m) - 14 : 0; }
__device__ inline float wg_tiles_max(const wmma_f32x8 *t, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i)
#pragma unroll
        for (int v = 0; v < 8; ++v) m = fmaxf(m, fabsf(t[i][v]));
    return wg_wave_max(m);
}

// A workgroup barrier that orders LDS only (as attention.hip's lds_barrier): __syncthreads' workgroup fence also
// waits for every global load in flight (s_waitcnt vmcnt(0) - a workgroup spans two CUs' L0 caches in WGP mode), so
// with 5-6 barriers a chunk a load issued ahead was drained at the next barrier. The waves share only LDS.
__device__ __forceinline__ void wg_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <typename T, bool One, bool F16, bool U2 = false>
__global__ void __launch_bounds__(kWgWaves * 32)
    delta_rule_wmma_kernel(const T *qkv, const float *beta, const float *g, const float *state_in, float *state_out,
                           T *out, DrParams p) {
    constexpr int D = (int)kGdnHeadDim;
    const int64_t h = blockIdx.x, kh = h / (p.Hv / p.Hk);
    const int tid = threadIdx.x, lane = tid % 32, w = tid / 32, col = lane % 16, half = lane / 16, v_w = 16 * w;

    // One (gdn-precision fast): the lo buffers are a token 8 elements, never written or read.
    // U2 (gdn-precision half-u) = half with the two U-path products (U = T R, S += K~^T U) as F16 hi + lo pairs - K, T,
    // K~^T keep their lo halves; everything else one F16 product.
    // kRaw (U2, BF16 activations): Q, K staged raw - BF16 x a power of two is exact in F16 - with 1/|q|, 1/|k| (and the
    // scale) applied to the FP32 products, so no product sees Q / K rounding; K~^T's lo half lives in Vs (free after R).
    // P as an F16 hi + lo pair too (P U is the chunk's smallest product).
    static_assert(!U2 || (One && F16), "U2 is a half variant");
    constexpr bool OneU = One && !U2;  // the U-path products' form (and P U's)
    constexpr bool kRaw = U2 && sizeof(T) == 2;
    constexpr bool kQKlo = !One || (U2 && !kRaw);  // Kl / Ql hold Q / K lo halves (or K~^T's lo, F32 half-u)
    __shared__ __attribute__((aligned(16))) uint16_t Kh[kWgC * kWgLd], Kl[kQKlo ? kWgC * kWgLd : 8];
    __shared__ __attribute__((aligned(16))) uint16_t Qh[kWgC * kWgLd], Ql[kQKlo ? kWgC * kWgLd : 8];  // later: K~^T [k][t]
    __shared__ __attribute__((aligned(16))) uint16_t Th[kWgC * kWgLdT], Tl[OneU ? 8 : kWgC * kWgLdT];
    __shared__ __attribute__((aligned(16))) uint16_t Ph[kWgC * kWgLdT], Pl[OneU ? 8 : kWgC * kWgLdT];
    __shared__ float Qsc[kRaw ? kWgC : 1], Ksc[kRaw ? kWgC : 1];  // kRaw: a row's factor back to normalized
    __shared__ float Am[kWgC][kWgC + 1];
    // BF16 activations: the chunk's V rows (this value head), loaded with Q and K (16-byte loads, one latency) instead of
    // 16 scattered 2-byte loads per lane after W and QS. F32 activations (tests) keep the direct loads.
    __shared__ __attribute__((aligned(16))) uint16_t Vs[sizeof(T) == 2 ? kWgC * kWgLd : 8];
    __shared__ float G[kWgC], eG[kWgC], B[kWgC], eL[kWgC];
    __shared__ float t_max[kWgWaves];  // half: each wave's max |T| of the chunk (written every chunk: nothing to reset)
    uint16_t *KTh = Qh, *KTl = kRaw ? Vs : Ql;  // K~^T [128][kWgLdKT], reusing the Q buffers (kRaw: its lo in Vs)

    // S slice: tile rt holds k rows [16 rt, 16 rt + 16), element v = row 16 rt + 2v + half, column v_w + col.
    wmma_f32x8 S[D / 16];
#pragma unroll
    for (int rt = 0; rt < D / 16; ++rt)
#pragma unroll
        for (int v = 0; v < 8; ++v)
            S[rt][v] = p.reset ? 0.0f : state_in[(h * D + 16 * rt + 2 * v + half) * D + v_w + col];

    const float q_scale = rsqrtf((float)D);
    // BF16 activations: a chunk's Q, K, V (this thread's 16 elements of row tid / 8 each) and wave 0's g / beta are
    // loaded into registers one chunk ahead - issued right after the previous chunk's staging, so their latency runs
    // under that chunk's compute. The loads are unconditional at clamped addresses (rows past the end re-read the last
    // row and are zeroed where staged): a branch around them makes the compiler drain them (vmcnt(0)) before the
    // compute - the prefetch with guarded loads spilled and lost on the half path.
    // Named registers, not an array: an indexed uint4 array went to scratch (each prefetch load stored there right
    // after it returned - its latency exposed at the issue point), though the resource report showed no spills.
    uint4 pq0, pq1, pk0, pk1, pv0, pv1;
    float g_pf = 0.0f, b_pf = 0.0f;
    const auto prefetch = [&](int64_t c) {
        if constexpr (sizeof(T) == 2) {
            const int r = tid / 8, part = tid % 8;
            const int64_t row = c + r < p.T ? c + r : p.T - 1;
            const T *qrow = qkv + row * p.ts + kh * D + part * 16, *krow = qrow + p.Hk * D,
                    *vrow = qkv + row * p.ts + 2 * p.Hk * D + h * D + part * 16;
            pq0 = reinterpret_cast<const uint4 *>(qrow)[0], pq1 = reinterpret_cast<const uint4 *>(qrow)[1];
            pk0 = reinterpret_cast<const uint4 *>(krow)[0], pk1 = reinterpret_cast<const uint4 *>(krow)[1];
            pv0 = reinterpret_cast<const uint4 *>(vrow)[0], pv1 = reinterpret_cast<const uint4 *>(vrow)[1];
            const int64_t tr = c + lane < p.T ? c + lane : p.T - 1;
            g_pf = g[tr * p.Hv + h], b_pf = beta[tr * p.Hv + h];
        } else
            (void)c;
    };
    // Ahead only for half, where it measured faster; the one-pass BF16 and exact variants load at the chunk start.
    // Not for half-u: the prefetch's registers spill it at 256 VGPRs.
    constexpr bool kAhead = F16 && !U2;
    if constexpr (kAhead) prefetch(0);
    for (int64_t c0 = 0; c0 < p.T; c0 += kWgC) {
        const int n = p.T - c0 < kWgC ? (int)(p.T - c0) : kWgC;
        if constexpr (!kAhead) prefetch(c0);
        // Q, K rows: 8 threads per row, 16 contiguous elements each (vector loads); L2-normalized in FP32 (as the
        // scalar kernels), stored hi / lo.
        {
            const int r = tid / 8, part = tid % 8;
            float qv[16], kv[16], sq = 0.0f, sk = 0.0f;
            const T *qrow = qkv + (c0 + r) * p.ts + kh * D + part * 16, *krow = qrow + p.Hk * D;
            if constexpr (sizeof(T) == 2) {
                (void)qrow, (void)krow;
                // Rows past the end (the clamped re-read of the last row) zeroed - not needed for the result (their beta
                // is 0, so their T row is the identity, U row 0, no state change) but kept so they hold no stale values.
                const bool in = r < n;
                const auto zf = [in](uint4 u) { return in ? u : make_uint4(0, 0, 0, 0); };
                const uint4 qa = zf(pq0), qb = zf(pq1), ka = zf(pk0), kb = zf(pk1), va = zf(pv0), vb = zf(pv1);
                reinterpret_cast<uint4 *>(Vs + r * kWgLd + part * 16)[0] = va;
                reinterpret_cast<uint4 *>(Vs + r * kWgLd + part * 16)[1] = vb;
                const uint32_t qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                const uint32_t kw[8] = {ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    qv[2 * i] = __uint_as_float(qw[i] << 16), qv[2 * i + 1] = __uint_as_float(qw[i] & 0xffff0000u);
                    kv[2 * i] = __uint_as_float(kw[i] << 16), kv[2 * i + 1] = __uint_as_float(kw[i] & 0xffff0000u);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4 q4 = make_float4(0.f, 0.f, 0.f, 0.f), k4 = q4;
                    if (r < n) q4 = reinterpret_cast<const float4 *>(qrow)[i], k4 = reinterpret_cast<const float4 *>(krow)[i];
                    qv[4 * i] = q4.x, qv[4 * i + 1] = q4.y, qv[4 * i + 2] = q4.z, qv[4 * i + 3] = q4.w;
                    kv[4 * i] = k4.x, kv[4 * i + 1] = k4.y, kv[4 * i + 2] = k4.z, kv[4 * i + 3] = k4.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 16; ++e) sq = fmaf(qv[e], qv[e], sq), sk = fmaf(kv[e], kv[e], sk);
            for (int off = 1; off < 8; off <<= 1) {
                sq += __shfl_xor(sq, off);
                sk += __shfl_xor(sk, off);
            }
            const float iq = rsqrtf(sq + 1e-6f) * q_scale, ik = rsqrtf(sk + 1e-6f);
            if constexpr (kRaw) {
                // Raw rows x 2^-e (e: the row's largest |value| into [2^14, 2^15)): exact in F16; the factor back to the
                // normalized row (1/|x| x 2^e) applied to each product's FP32 result.
                float mq = 0.0f, mk = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) mq = fmaxf(mq, fabsf(qv[e])), mk = fmaxf(mk, fabsf(kv[e]));
                for (int off = 1; off < 8; off <<= 1) mq = fmaxf(mq, __shfl_xor(mq, off)), mk = fmaxf(mk, __shfl_xor(mk, off));
                const int eq = wg_exp(mq), ek = wg_exp(mk);
                const float sq2 = ldexpf(1.0f, -eq), sk2 = ldexpf(1.0f, -ek);
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int d = part * 16 + e;
                    wg_split<true, F16>(qv[e] * sq2, Qh[r * kWgLd + d], Ql[0]);
                    wg_split<true, F16>(kv[e] * sk2, Kh[r * kWgLd + d], Kl[0]);
                }
                if (part == 0) Qsc[r] = ldexpf(iq, eq), Ksc[r] = ldexpf(ik, ek);
            } else {
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int d = part * 16 + e;
                    wg_split<One, F16>(qv[e] * iq, Qh[r * kWgLd + d], Ql[r * kWgLd + d]);
                    wg_split<kQKlo ? false : true, F16>(kv[e] * ik, Kh[r * kWgLd + d], Kl[r * kWgLd + d]);
                }
            }
        }
        // G = cumsum(g) (padding rows keep the last real G), e^G, beta, e^(G_last - G): wave 0, one token per lane, the
        // prefix sum as a shuffle scan (the order of the adds differs from a sequential sum only in FP32 rounding).
        if (w == 0) {
            const int t = lane;
            float gv, bv;
            if constexpr (sizeof(T) == 2) gv = t < n ? g_pf : 0.0f, bv = t < n ? b_pf : 0.0f;
            else gv = t < n ? g[(c0 + t) * p.Hv + h] : 0.0f, bv = t < n ? beta[(c0 + t) * p.Hv + h] : 0.0f;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float up = __shfl_up(gv, off, 32);
                if (t >= off) gv += up;
            }
            const float last = __shfl(gv, 31, 32);
            G[t] = gv, eG[t] = expf(gv), B[t] = bv, eL[t] = expf(last - gv);
        }
        wg_lds_barrier();
        if constexpr (kAhead) prefetch(c0 + kWgC < p.T ? c0 + kWgC : c0);  // the next chunk (the last re-reads itself)

        // A (waves 0..3) and P (waves 4..7): one 16 x 16 tile each.
        {
            const bool isA = w < 4;
            const int ti = (w % 4) / 2, tj = (w % 4) % 2;
            const uint16_t *Xh = isA ? Kh : Qh, *Xl = isA ? Kl : Ql;
            wmma_f32x8 acc = {};
#pragma unroll
            for (int d0 = 0; d0 < D; d0 += 16) {
                const wmma_bf16x16 ah = wmma_load_rows(Xh + ti * 16 * kWgLd + d0, kWgLd),
                                   al = wg_load_lo<One, F16>(Xl + ti * 16 * kWgLd + d0, kWgLd);
                const wmma_bf16x16 bh = wmma_load_rows(Kh + tj * 16 * kWgLd + d0, kWgLd),
                                   bl = wg_load_lo<One, F16>(Kl + tj * 16 * kWgLd + d0, kWgLd);
                acc = wg_mma3<One, F16>(ah, al, bh, bl, acc);
                __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = ti * 16 + 2 * v + half, s2 = tj * 16 + col;
                const float decay = s2 <= t ? expf(G[t] - G[s2]) : 0.0f;
                float a = acc[v];
                if constexpr (kRaw) a *= (isA ? Ksc[t] : Qsc[t]) * Ksc[s2];  // raw rows back to normalized
                if (isA) Am[t][s2] = s2 < t ? B[t] * decay * a : 0.0f;
                else wg_split<OneU, F16>(decay * a, Ph[t * kWgLdT + s2], Pl[t * kWgLdT + s2]);
            }
        }
        wg_lds_barrier();

        // T = (I + A)^-1 over all 8 waves: lane r owns row r, wave w the columns [4w, 4w + 4). Column-oriented forward
        // substitution: at step q row q of every column is final, so each lane takes A[r][q] (one LDS read; 0 for
        // r <= q) and row q's value by readlane, and updates its 4 columns - no barrier, no per-FMA LDS read. (Was:
        // wave 0 alone, 32 dependent steps with an LDS read per FMA, the other waves waiting at the next barrier -
        // 26.5% of the kernel's instruction latency.)
        int e_T = 0;
        {
            constexpr int kCpw = kWgC / kWgWaves;
            static_assert(kCpw * kWgWaves == kWgC && kWgC == 32, "4 columns a wave, a row a lane");
            const int j0 = w * kCpw;
            float x[kCpw];
#pragma unroll
            for (int c = 0; c < kCpw; ++c) x[c] = lane == j0 + c ? 1.0f : 0.0f;
#pragma unroll
            for (int q = 0; q < kWgC - 1; ++q) {
                if (q < j0) continue;  // rows above the wave's first column are 0 in all its columns (wave-uniform)
                const float a = Am[lane][q];
#pragma unroll
                for (int c = 0; c < kCpw; ++c)
                    x[c] = fmaf(-a, __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[c]), q)), x[c]);
            }
            float sT = 1.0f;
            if constexpr (F16) {
                float m = 0.0f;
#pragma unroll
                for (int c = 0; c < kCpw; ++c) m = fmaxf(m, fabsf(x[c]));
                m = wg_wave_max(m);
                if (lane == 0) t_max[w] = m;
                wg_lds_barrier();  // every wave's max in
                float mt = 0.0f;
#pragma unroll
                for (int i = 0; i < kWgWaves; ++i) mt = fmaxf(mt, t_max[i]);
                e_T = wg_exp(mt);
                sT = ldexpf(1.0f, -e_T);
            }
#pragma unroll
            for (int c = 0; c < kCpw; ++c) wg_split<OneU, F16>(x[c] * sT, Th[lane * kWgLdT + j0 + c], Tl[lane * kWgLdT + j0 + c]);
        }
        // W = K S and QS = Q S (two 16-row tiles of tokens each), S's tiles as B fragments one k tile at a time.
        wmma_f32x8 W[2] = {}, QS[2] = {};
        const int e_S = F16 ? wg_exp(wg_tiles_max(S, D / 16)) : 0;
        const float sS = ldexpf(1.0f, -e_S);
#pragma unroll
        for (int rt = 0; rt < D / 16; ++rt) {
            wmma_bf16x16 bh, bl;
            wg_c_to_b<One, F16>(S[rt], bh, bl, sS);
#pragma unroll
            for (int ti = 0; ti < 2; ++ti) {
                const int off = ti * 16 * kWgLd + rt * 16;
                W[ti] = wg_mma3<One, F16>(wmma_load_rows(Kh + off, kWgLd), wg_load_lo<One, F16>(Kl + off, kWgLd), bh, bl, W[ti]);
                QS[ti] = wg_mma3<One, F16>(wmma_load_rows(Qh + off, kWgLd), wg_load_lo<One, F16>(Ql + off, kWgLd), bh, bl, QS[ti]);
            }
            __builtin_amdgcn_sched_barrier(0);  // keep each tile's loads with its tile: hoisting them all spilled
        }
        if constexpr (F16)
#pragma unroll
            for (int ti = 0; ti < 2; ++ti)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    W[ti][v] = ldexpf(W[ti][v], e_S), QS[ti][v] = ldexpf(QS[ti][v], e_S);
                    if constexpr (kRaw) {
                        const int t = ti * 16 + 2 * v + half;
                        W[ti][v] *= Ksc[t], QS[ti][v] *= Qsc[t];
                    }
                }
        // R = beta (V - e^G W) in C layout (rows t, columns v).
        wmma_f32x8 R[2];
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = ti * 16 + 2 * v + half;
                float vv;
                if constexpr (sizeof(T) == 2) vv = wg_f32(Vs[t * kWgLd + v_w + col]);  // zero past n (staged so)
                else vv = t < n ? load(qkv, (c0 + t) * p.ts + 2 * p.Hk * D + h * D + v_w + col) : 0.0f;
                R[ti][v] = B[t] * (vv - eG[t] * W[ti][v]);
            }
        wg_lds_barrier();  // T complete; every wave done reading Q (its buffer becomes K~^T below)

        // K~^T [k][t] = (e^(G_last - G_t) K[t][k]) into Q's buffer, hi / lo: 256 threads x 16 elements.
        for (int i = tid; i < D * kWgC; i += kWgWaves * 32) {
            const int k = i / kWgC, t = i % kWgC;
            float kv;
            if constexpr (kRaw) kv = (float)__builtin_bit_cast(_Float16, Kh[t * kWgLd + k]) * Ksc[t];
            else if constexpr (F16)
                kv = (float)__builtin_bit_cast(_Float16, Kh[t * kWgLd + k]) +
                     (OneU ? 0.0f : (float)__builtin_bit_cast(_Float16, Kl[t * kWgLd + k]));
            else kv = wg_f32(Kh[t * kWgLd + k]) + (One ? 0.0f : wg_f32(Kl[t * kWgLd + k]));
            wg_split<OneU, F16>(eL[t] * kv, KTh[k * kWgLdKT + t], KTl[k * kWgLdKT + t]);
        }
        // U = T R: A = T rows (t), K = s (2 steps); B = R's tiles.
        wmma_bf16x16 rbh[2], rbl[2];
        const int e_R = F16 ? wg_exp(wg_tiles_max(R, 2)) : 0;
        wg_c_to_b<OneU, F16>(R[0], rbh[0], rbl[0], ldexpf(1.0f, -e_R));
        wg_c_to_b<OneU, F16>(R[1], rbh[1], rbl[1], ldexpf(1.0f, -e_R));
        wmma_f32x8 U[2] = {};
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int off = ti * 16 * kWgLdT + ks * 16;
                U[ti] = wg_mma3<OneU, F16>(wmma_load_rows(Th + off, kWgLdT), wg_load_lo<OneU, F16>(Tl + off, kWgLdT), rbh[ks], rbl[ks], U[ti]);
            }
        int e_U = 0;
        if constexpr (F16) {
            const int eTR = e_T + e_R;
#pragma unroll
            for (int ti = 0; ti < 2; ++ti)
#pragma unroll
                for (int v = 0; v < 8; ++v) U[ti][v] = ldexpf(U[ti][v], eTR);
            e_U = wg_exp(wg_tiles_max(U, 2));
        }
        // O = e^G QS + P U, stored.
        wmma_bf16x16 ubh[2], ubl[2];
        wg_c_to_b<OneU, F16>(U[0], ubh[0], ubl[0], ldexpf(1.0f, -e_U));
        wg_c_to_b<OneU, F16>(U[1], ubh[1], ubl[1], ldexpf(1.0f, -e_U));
#pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            wmma_f32x8 PU = {};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int off = ti * 16 * kWgLdT + ks * 16;
                PU = wg_mma3<OneU, F16>(wmma_load_rows(Ph + off, kWgLdT), wg_load_lo<OneU, F16>(Pl + off, kWgLdT), ubh[ks], ubl[ks], PU);
            }
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = ti * 16 + 2 * v + half;
                if (t < n) store(out, ((c0 + t) * p.Hv + h) * D + v_w + col, eG[t] * QS[ti][v] + (F16 ? ldexpf(PU[v], e_U) : PU[v]));
            }
        }
        wg_lds_barrier();  // K~^T staged
        // S = e^(G_last) S + K~^T U: A = K~^T rows (k), K = t (2 steps); B = U's tiles.
        const float e_last = eG[kWgC - 1];
#pragma unroll
        for (int rt = 0; rt < D / 16; ++rt) {
            // half: S x 2^-e_U, the scaled product added, then x 2^e_U - powers of two, so the same as adding the
            // product x 2^e_U.
            const float pre = F16 ? ldexpf(e_last, -e_U) : e_last;
#pragma unroll
            for (int v = 0; v < 8; ++v) S[rt][v] *= pre;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int off = rt * 16 * kWgLdKT + ks * 16;
                S[rt] = wg_mma3<OneU, F16>(wmma_load_rows(KTh + off, kWgLdKT), wg_load_lo<OneU, F16>(KTl + off, kWgLdKT), ubh[ks], ubl[ks], S[rt]);
            }
            if constexpr (F16)
#pragma unroll
                for (int v = 0; v < 8; ++v) S[rt][v] = ldexpf(S[rt][v], e_U);
            __builtin_amdgcn_sched_barrier(0);
        }
        wg_lds_barrier();  // everything of this chunk read before the next chunk's staging
    }
#pragma unroll
    for (int rt = 0; rt < D / 16; ++rt)
#pragma unroll
        for (int v = 0; v < 8; ++v) state_out[(h * D + 16 * rt + 2 * v + half) * D + v_w + col] = S[rt][v];
}

bool overlaps(const void *a, int64_t a_bytes, const void *b, int64_t b_bytes) {
    const auto ab = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return ab < bb + (uintptr_t)b_bytes && bb < ab + (uintptr_t)a_bytes;
}

}  // namespace

void gdn_front(const void *in_proj, const GdnFrontShape &s, const float *conv_w, const void *conv_state_in, void *conv_state_out,
               bool reset_state,
               const float *A_log, const float *dt_bias, void *qkv_out, float *beta, float *g, Act act,
               hipStream_t stream) {
    STRIX_CHECK(in_proj && conv_w && (conv_state_in || reset_state) && conv_state_out && A_log && dt_bias && qkv_out &&
                    beta && g,
                "gdn_front: null pointer (in_proj=", in_proj, ", conv_w=", conv_w, ", conv_state_in=", conv_state_in,
                " - may be null only with reset_state, conv_state_out=", conv_state_out,
                ", A_log=", A_log, ", dt_bias=", dt_bias, ", qkv_out=", qkv_out, ", beta=", beta, ", g=", g, ")");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "gdn_front: unsupported activation dtype ", (int)act);
    STRIX_CHECK(s.T >= 1 && s.T <= kMaxT, "gdn_front: T = ", s.T, ", expected 1..", kMaxT);
    STRIX_CHECK(s.Hk >= 1 && s.Dk >= 1 && s.Hv >= 1 && s.Dv >= 1 && s.Hv % s.Hk == 0, "gdn_front: heads Hk = ", s.Hk,
                " x Dk = ", s.Dk, ", Hv = ", s.Hv, " x Dv = ", s.Dv,
                " - expected all >= 1 and Hv a multiple of Hk (value head h pairs with key head h / (Hv/Hk))");
    STRIX_CHECK(s.Hk * s.Dk <= kMaxChannels && s.Hv * s.Dv <= kMaxChannels, "gdn_front: Hk*Dk = ", s.Hk * s.Dk,
                ", Hv*Dv = ", s.Hv * s.Dv, " exceed ", kMaxChannels, " (shape mix-up?)");
    const int64_t C = s.channels();
    STRIX_CHECK(s.in_ts >= C, "gdn_front: in_proj row stride ", s.in_ts, " < qkv channels ", C);
    STRIX_CHECK(s.b_col >= C && s.b_col + s.Hv <= s.in_ts && s.a_col >= C && s.a_col + s.Hv <= s.in_ts,
                "gdn_front: b columns [", s.b_col, ", ", s.b_col + s.Hv, ") and a columns [", s.a_col, ", ",
                s.a_col + s.Hv, ") must lie in [", C, ", ", s.in_ts, ") - after qkv, within the row");
    STRIX_CHECK(s.b_col + s.Hv <= s.a_col || s.a_col + s.Hv <= s.b_col, "gdn_front: b columns [", s.b_col, ", ",
                s.b_col + s.Hv, ") overlap a columns [", s.a_col, ", ", s.a_col + s.Hv, ")");
    const int64_t es = act == Act::F32 ? 4 : 2;
    for (auto [p, name] : {std::pair<const void *, const char *>{in_proj, "in_proj"},
                           {conv_state_in, "conv_state_in"}, {conv_state_out, "conv_state_out"}, {qkv_out, "qkv_out"}})
        STRIX_CHECK(reinterpret_cast<uintptr_t>(p) % es == 0, "gdn_front: ", name, " = ", p, " not aligned to ", es,
                    " bytes for ", act_name(act));
    for (auto [p, name] : {std::pair<const void *, const char *>{conv_w, "conv_w"}, {A_log, "A_log"},
                           {dt_bias, "dt_bias"}, {beta, "beta"}, {g, "g"}})
        STRIX_CHECK(reinterpret_cast<uintptr_t>(p) % 4 == 0, "gdn_front: ", name, " = ", p, " not 4-byte aligned");

    const int64_t in_bytes = s.T * s.in_ts * es, out_bytes = s.T * C * es,
                  state_bytes = kS * C * es, gate_bytes = s.T * s.Hv * 4;
    const struct {
        const void *p;
        int64_t bytes;
        const char *name;
    } bufs[] = {{in_proj, in_bytes, "in_proj"},
                {conv_state_out, state_bytes, "conv_state_out"},
                {qkv_out, out_bytes, "qkv_out"},
                {beta, gate_bytes, "beta"},
                {g, gate_bytes, "g"},
                // In place is fine (see the kernel); otherwise the old state must be separate from everything.
                {conv_state_in == conv_state_out ? nullptr : conv_state_in, state_bytes, "conv_state_in"}};
    for (size_t i = 0; i < std::size(bufs); ++i)
        for (size_t j = i + 1; j < std::size(bufs); ++j)
            if (bufs[i].p && bufs[j].p)
                STRIX_CHECK(!overlaps(bufs[i].p, bufs[i].bytes, bufs[j].p, bufs[j].bytes), "gdn_front: ",
                            bufs[i].name, " [", bufs[i].p, ", +", bufs[i].bytes, " B) overlaps ", bufs[j].name, " [",
                            bufs[j].p, ", +", bufs[j].bytes, " B)");

    const int64_t tiles = (s.T + kTile - 1) / kTile, cblocks = (C + kThreads - 1) / kThreads;
    const int64_t conv_blocks = cblocks * tiles, gate_blocks = (s.T * s.Hv + kThreads - 1) / kThreads;
    STRIX_CHECK(conv_blocks + gate_blocks <= 0x7fffffff, "gdn_front: grid of ", conv_blocks + gate_blocks,
                " blocks exceeds the launch limit (T = ", s.T, ", C = ", C, ")");
    const Params p{s.T, C, s.in_ts, s.b_col, s.a_col, s.Hv, tiles, conv_blocks, reset_state};
    dim3 grid((unsigned)(conv_blocks + gate_blocks)), block(kThreads);
    if (act == Act::F32)
        gdn_front_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float *>(in_proj), conv_w,
                                                            static_cast<const float *>(conv_state_in),
                                                            static_cast<float *>(conv_state_out), A_log, dt_bias,
                                                            static_cast<float *>(qkv_out), beta, g, p);
    else
        gdn_front_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(in_proj), conv_w,
                                                               static_cast<const uint16_t *>(conv_state_in),
                                                               static_cast<uint16_t *>(conv_state_out), A_log, dt_bias,
                                                               static_cast<uint16_t *>(qkv_out), beta, g, p);
    STRIX_HIP_CHECK_LAUNCH("gdn_front ", act_name(act), " T=", s.T, " C=", C, " Hv=", s.Hv);
}

void delta_rule(const void *qkv, const float *beta, const float *g, const DeltaRuleShape &s, const float *state_in,
                float *state_out, bool reset_state, void *out, Act act, hipStream_t stream, DeltaRulePath path,
                DrPrecision precision) {
    STRIX_CHECK(path == DeltaRulePath::Auto || path == DeltaRulePath::PerToken || path == DeltaRulePath::Chunked ||
                    path == DeltaRulePath::ChunkedWmma,
                "delta_rule: unknown path ", (int)path);
    STRIX_CHECK(precision == DrPrecision::Exact || precision == DrPrecision::Fast || precision == DrPrecision::Half || precision == DrPrecision::HalfU,
                "delta_rule: unknown precision ",
                (int)precision);
    STRIX_CHECK(qkv && beta && g && (state_in || reset_state) && state_out && out, "delta_rule: null pointer (qkv=", qkv,
                ", beta=", beta, ", g=", g, ", state_in=", state_in, " - may be null only with reset_state, state_out=",
                state_out, ", out=", out, ")");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "delta_rule: unsupported activation dtype ", (int)act);
    STRIX_CHECK(s.T >= 1 && s.T <= kMaxT, "delta_rule: T = ", s.T, ", expected 1..", kMaxT);
    STRIX_CHECK(s.Hk >= 1 && s.Hv >= s.Hk && s.Hv % s.Hk == 0 && s.Hv <= 4096, "delta_rule: Hk = ", s.Hk, ", Hv = ",
                s.Hv, " - expected 1 <= Hk <= Hv <= 4096, Hv a multiple of Hk");
    constexpr int64_t D = kGdnHeadDim;
    const int64_t C = (2 * s.Hk + s.Hv) * D;
    STRIX_CHECK(s.qkv_ts >= C, "delta_rule: qkv row stride ", s.qkv_ts, " < q|k|v width ", C, " (Hk = ", s.Hk,
                ", Hv = ", s.Hv, ", head dim ", D, ")");
    int wsize = 0;
    STRIX_HIP_CHECK(hipDeviceGetAttribute(&wsize, hipDeviceAttributeWarpSize, 0), "delta_rule: querying the wave size");
    STRIX_CHECK(wsize >= 32 && D % wsize == 0, "delta_rule: wave size ", wsize, " - the L2-norm reduction needs a wave "
                "size >= 32 that divides ", D);
    const int64_t es = act == Act::F32 ? 4 : 2;
    STRIX_CHECK(reinterpret_cast<uintptr_t>(qkv) % es == 0 && reinterpret_cast<uintptr_t>(out) % es == 0,
                "delta_rule: qkv = ", qkv, ", out = ", out, " not aligned to ", es, " bytes for ", act_name(act));
    for (auto [p, name] : {std::pair<const void *, const char *>{beta, "beta"}, {g, "g"}, {state_in, "state_in"},
                           {state_out, "state_out"}})
        STRIX_CHECK(reinterpret_cast<uintptr_t>(p) % 4 == 0, "delta_rule: ", name, " = ", p, " not 4-byte aligned");
    const int64_t qkv_bytes = s.T * s.qkv_ts * es, gate_bytes = s.T * s.Hv * 4, state_bytes = s.Hv * D * D * 4,
                  out_bytes = s.T * s.Hv * D * es;
    const struct {
        const void *p;
        int64_t bytes;
        const char *name;
    } bufs[] = {{qkv, qkv_bytes, "qkv"},
                {beta, gate_bytes, "beta"},
                {g, gate_bytes, "g"},
                {state_out, state_bytes, "state_out"},
                {out, out_bytes, "out"},
                // In place is fine (see the kernel); otherwise the old state must be separate from everything.
                {state_in == state_out ? nullptr : state_in, state_bytes, "state_in"}};
    for (size_t i = 0; i < std::size(bufs); ++i)
        for (size_t j = i + 1; j < std::size(bufs); ++j)
            if (bufs[i].p && bufs[j].p)
                STRIX_CHECK(!overlaps(bufs[i].p, bufs[i].bytes, bufs[j].p, bufs[j].bytes), "delta_rule: ",
                            bufs[i].name, " [", bufs[i].p, ", +", bufs[i].bytes, " B) overlaps ", bufs[j].name, " [",
                            bufs[j].p, ", +", bufs[j].bytes, " B)");

    const DrParams p{s.T, s.Hk, s.Hv, s.qkv_ts, reset_state};
    const bool chunked = path == DeltaRulePath::Chunked || (path == DeltaRulePath::Auto && s.T >= kDeltaRuleChunkMinT);
    if (path == DeltaRulePath::ChunkedWmma) {
        STRIX_CHECK(wsize == 32, "delta_rule (WMMA): needs wave32 (gfx11 WMMA), wave size ", wsize);
        dim3 grid((unsigned)s.Hv), block(kWgWaves * 32);
        auto launch = [&](auto one) {
            constexpr bool One = decltype(one)::value != 0, F16 = decltype(one)::value >= 2, U2 = decltype(one)::value == 3;
            if (act == Act::F32)
                delta_rule_wmma_kernel<float, One, F16, U2><<<grid, block, 0, stream>>>(static_cast<const float *>(qkv), beta, g,
                                                                               state_in, state_out,
                                                                               static_cast<float *>(out), p);
            else
                delta_rule_wmma_kernel<uint16_t, One, F16, U2><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(qkv), beta,
                                                                                  g, state_in, state_out,
                                                                                  static_cast<uint16_t *>(out), p);
        };
        if (precision == DrPrecision::Fast) launch(std::integral_constant<int, 1>{});
        else if (precision == DrPrecision::Half) launch(std::integral_constant<int, 2>{});
        else if (precision == DrPrecision::HalfU) launch(std::integral_constant<int, 3>{});
        else launch(std::integral_constant<int, 0>{});
    } else if (chunked) {
        dim3 grid((unsigned)(s.Hv * (D / kCkCols))), block(kThreads);
        if (act == Act::F32)
            delta_rule_chunked_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float *>(qkv), beta, g,
                                                                         state_in, state_out,
                                                                         static_cast<float *>(out), p);
        else
            delta_rule_chunked_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(qkv), beta,
                                                                            g, state_in, state_out,
                                                                            static_cast<uint16_t *>(out), p);
    } else {
        dim3 grid((unsigned)(s.Hv * (D / kDrCols))), block(kThreads);
        if (act == Act::F32)
            delta_rule_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float *>(qkv), beta, g, state_in,
                                                                 state_out, static_cast<float *>(out), p);
        else
            delta_rule_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(qkv), beta, g,
                                                                    state_in, state_out,
                                                                    static_cast<uint16_t *>(out), p);
    }
    STRIX_HIP_CHECK_LAUNCH("delta_rule ", path == DeltaRulePath::ChunkedWmma ? "chunked-wmma " : chunked ? "chunked " : "per-token ", act_name(act), " T=", s.T, " Hk=", s.Hk,
                           " Hv=", s.Hv);
}

}  // namespace strix::kernels
