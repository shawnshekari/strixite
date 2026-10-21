#include "kernels/attention.hpp"

#include "common/hip_check.hpp"
#include "kernels/kv8_math.hpp"
#include "kernels/rope_math.hpp"
#include "kernels/wmma.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace strix::kernels {

namespace {

using rope_math::bf16_bits_to_f32;
using rope_math::f32_to_bf16_bits;

constexpr int kNormThreads = 256;
constexpr int kD = 256;       // head dim the attention kernel is built for
constexpr int kTile = 16;     // keys staged in LDS per step
constexpr int64_t kQsaTokens = 4;  // tokens per QSA indexer block (kernels/qsa.hpp kQsaBlock)
constexpr int kMaxGroup = 16; // query heads per KV head (one wave each; q for all of them in LDS)
constexpr int64_t kMaxPos = 1ll << 24;

__device__ inline float load(const float *p, int64_t i) { return p[i]; }
__device__ inline float load(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }
template <typename T> __device__ inline float round_act(float v) { return v; }
template <> __device__ inline float round_act<uint16_t>(float v) { return rope_math::rb(v); }

// ---- q/k norm + RoPE: one block per (token, head) ------------------------------------------------

// One head: zero-centred RMSNorm (1 + w), then RoPE on the first 2 * half dims, written to o. Shared by the
// single-call and the multi-job kernel, so both compute exactly the same values.
// triple: this row's interleaved-mRoPE (t, h, w) section positions (kernels/rope.hpp MropePos), or null for
// the scalar pos (text-only: all sections equal pos, and pair j's angle is pos * inv_freq[j] as ever -
// bit-identical, the branch is outside the rotation math).
template <typename T>
__device__ inline void norm_rope_head(const T *x, T *o, int64_t D, const float *w, float eps, int64_t half, int64_t pos,
                                      const int64_t *triple, const float *inv_freq, float cos_sin_scale, float *row,
                                      float *red) {
    constexpr bool kBF16 = sizeof(T) == 2;
    // Same reduction shape as rmsnorm_zc (strided partials, tree), same per-element formula.
    float ss = 0.f;
    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
        const float v = load(x, j);
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down(ss, off, 32);
    }
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int num_waves = blockDim.x >> 5;
    if (lane == 0) {
        red[wave] = ss;
    }
    __syncthreads();
    if (wave == 0) {
        float v = (lane < num_waves) ? red[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off, 32);
        }
        if (lane == 0) {
            red[0] = v;
        }
    }
    __syncthreads();
    const float inv = 1.0f / sqrtf(red[0] / (float)D + eps);
    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) row[j] = round_act<T>((load(x, j) * inv) * (1.0f + w[j]));
    __syncthreads();  // all of x read and row complete before any write (in place is safe)
    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
        if (j < half) {
            float c, s, y0, y1;
            rope_math::cos_sin<kBF16>(triple ? triple[j % 3] : pos, inv_freq[j], cos_sin_scale, c, s);
            rope_math::rotate<kBF16>(row[j], row[j + half], c, s, y0, y1);
            store(o, j, y0);
            store(o, j + half, y1);
        } else if (j >= 2 * half) {
            store(o, j, row[j]);
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(kNormThreads)
    qk_norm_rope_kernel(const T *in, int64_t in_ts, int64_t in_hs, T *out, int64_t out_ts, int64_t out_hs,
                        int64_t heads, int64_t D, const float *w, float eps, int64_t half, int64_t pos0,
                        const int64_t *pos3, const float *inv_freq, float cos_sin_scale) {
    extern __shared__ float row[];  // D normalized values, then kNormThreads partial sums
    const int64_t t = blockIdx.x / heads, h = blockIdx.x % heads;
    norm_rope_head(in + t * in_ts + h * in_hs, out + t * out_ts + h * out_hs, D, w, eps, half, pos0 + t,
                   pos3 ? pos3 + 3 * t : nullptr, inv_freq, cos_sin_scale, row, row + D);
}

// Multi-job: blocks job-major, T * heads of each; a job without w copies its heads verbatim.
struct NormRopeJobsArg {
    NormRopeJob j[kMaxNormRopeJobs];
    int n;
    int64_t max_d;
    const int64_t *pos3 = nullptr;  // interleaved-mRoPE triples for rows 0..T-1 (null: scalar pos0 + t)
};

template <typename T>
__global__ void __launch_bounds__(kNormThreads)
    qk_norm_rope_jobs_kernel(NormRopeJobsArg a, int64_t T_, float eps, int64_t half, int64_t pos0,
                             const float *inv_freq, float cos_sin_scale) {
    extern __shared__ float row[];  // max D normalized values, then kNormThreads partial sums
    int64_t b = blockIdx.x;
    int k = 0;
    while (k + 1 < a.n && b >= T_ * a.j[k].heads) b -= T_ * a.j[k].heads, ++k;  // uniform across the block
    const NormRopeJob &jb = a.j[k];
    const int64_t t = b / jb.heads, h = b % jb.heads;
    const T *x = static_cast<const T *>(jb.in) + t * jb.in_ts + h * jb.in_hs;
    T *o = static_cast<T *>(jb.out) + t * jb.out_ts + h * jb.out_hs;
    if (jb.w == nullptr) {
        for (int64_t j = threadIdx.x; j < jb.D; j += blockDim.x) o[j] = x[j];
        return;
    }
    norm_rope_head(x, o, jb.D, jb.w, eps, half, pos0 + t, a.pos3 ? a.pos3 + 3 * t : nullptr, inv_freq, cos_sin_scale,
                   row, row + a.max_d);
}

// ---- attention: partial softmax per (key split, KV head, token), one wave per query head --------

// Block = G waves (the query heads sharing KV head kh). K/V tiles of kTile keys are staged in LDS in
// the activation dtype, loaded with 16-byte vector loads and shared by the G waves. LDS rows are
// unpadded; their 16-byte chunks are XOR-swizzled by key (chunk c of key j lives at c ^ (j % 16)) so
// both the per-key score reads and the per-dim P*V reads are conflict-free 16-byte LDS reads.
// Scores: lane l takes key l % kTile over half of D; the halves are summed with a shuffle.
// Output: lane owns D / warpSize consecutive dims. Writes (m, l, acc[D]) unnormalized per
// (token, head, split).
template <typename T>
__device__ inline void unpack(const uint4 &u, float (&f)[16 / sizeof(T)]);
template <>
__device__ inline void unpack<uint16_t>(const uint4 &u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) f[2 * i] = __uint_as_float(w[i] << 16), f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
}
template <>
__device__ inline void unpack<float>(const uint4 &u, float (&f)[4]) {
    f[0] = __uint_as_float(u.x), f[1] = __uint_as_float(u.y), f[2] = __uint_as_float(u.z), f[3] = __uint_as_float(u.w);
}

// GATHER (QSA): query t attends to the tokens of its nsel[t] selected blocks sel[t * sel_ld + i] (4 each, in
// list order) and then its (pos + 1) % 4 tail tokens - key index j of the query maps to a position; otherwise keys
// 0..pos (dense, causal).
// R: the K/V reader (kernels/kv8_math.hpp) - the BF16 / FP32 cache or the 8-bit one.
template <typename T, bool GATHER, typename R>
__global__ void attention_partial_kernel(const T *q, int64_t q_ts, int64_t q_hs, R kv,
                                         int64_t Hq, int64_t Hk, int64_t pos0, int64_t kps, float scale,
                                         float *ws, int64_t splits, const int32_t *sel, const int32_t *nsel,
                                         int64_t sel_ld, uint32_t *err) {
    constexpr int kVec = 16 / sizeof(T);        // elements per 16-byte chunk
    constexpr int kChunks = kD / kVec;          // chunks per row
    __shared__ uint4 ks[kTile][kChunks], vs[kTile][kChunks];
    extern __shared__ float qs_flat[];  // [G][kD], sized at launch
    float(*qs)[kD] = reinterpret_cast<float(*)[kD]>(qs_flat);
    const int G = (int)(Hq / Hk);
    const int lane = threadIdx.x % warpSize, wv = threadIdx.x / warpSize;
    const int64_t sp = blockIdx.x, kh = blockIdx.y, t = blockIdx.z;
    const int64_t h = kh * G + wv;
    const int64_t pos = pos0 + t;
    const int64_t tail_n = (pos + 1) % kQsaTokens, tail0 = pos + 1 - tail_n;
    const int64_t g0q = kv8_math::exact_from(pos);  // 8-bit K/V: the rows this query sees exact (kernels/kv8.hpp)
    // A selection outside the query's complete blocks (a bug upstream) sets err (err[1] = the token, err[2] = the
    // bad value; the first flag wins) and is clamped to a visible key, so nothing out of bounds is read and the
    // caller fails the request (check_attention_error).
    const int64_t complete = (pos + 1) / kQsaTokens;
    auto flag = [&](int64_t value) {
        if (atomicCAS(err, 0u, 1u) == 0u) err[1] = (uint32_t)t, err[2] = (uint32_t)value;
    };
    int64_t ns = GATHER ? nsel[t] : 0;
    if (GATHER && (ns < 0 || ns > sel_ld || ns > complete)) {
        if (threadIdx.x == 0 && sp == 0 && kh == 0) flag(ns);
        ns = ns < 0 ? 0 : min(ns, min(sel_ld, complete));
    }
    const int64_t n_sel_keys = ns * kQsaTokens;
    const int64_t n_keys = GATHER ? n_sel_keys + tail_n : pos + 1;  // causal either way
    const int32_t *sel_row = GATHER ? sel + t * sel_ld : nullptr;
    auto key_pos = [&](int64_t j) -> int64_t {
        if (!GATHER) return j;
        if (j >= n_sel_keys) return tail0 + (j - n_sel_keys);
        const int64_t b = sel_row[j / kQsaTokens];
        if (b < 0 || b >= complete) {
            flag(b);
            return pos;
        }
        return b * kQsaTokens + j % kQsaTokens;
    };
    const int64_t k_begin = sp * kps, k_end = min(n_keys, k_begin + kps);
    constexpr int kPerLane = kD / 32;           // output dims per lane (wave32)
    constexpr int kLaneChunks = kPerLane / kVec; // 1 (BF16) or 2 (F32)

    for (int i = threadIdx.x; i < G * kD; i += blockDim.x)
        qs[i / kD][i % kD] = load(q, t * q_ts + (kh * G + i / kD) * q_hs + i % kD);

    float m = -INFINITY, l = 0.f, acc[kPerLane];
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) acc[e] = 0.f;

    const int key_lane = lane % kTile, half_sel = lane / kTile;  // wave32: 2 halves of D per key
    for (int64_t k0 = k_begin; k0 < k_end; k0 += kTile) {
        __syncthreads();  // previous tile consumed (and q staged, first time)
        const int64_t nk = min((int64_t)kTile, k_end - k0);
        for (int i = threadIdx.x; i < kTile * kChunks; i += blockDim.x) {
            const int j = i / kChunks, c = i % kChunks;
            uint4 kv4 = {0, 0, 0, 0}, vv4 = {0, 0, 0, 0};
            if (j < nk) {
                const int64_t row = key_pos(k0 + j), col = kh * kD + (int64_t)c * kVec;
                kv4 = kv.k(row, col, g0q);
                vv4 = kv.v(row, col, g0q);
            }
            ks[j][c ^ (j % 16)] = kv4;
            vs[j][c ^ (j % 16)] = vv4;
        }
        __syncthreads();
        float dot = 0.f;
#pragma unroll 4
        for (int c = half_sel * (kChunks / 2); c < (half_sel + 1) * (kChunks / 2); ++c) {
            float kf[kVec];
            unpack<T>(ks[key_lane][c ^ (key_lane % 16)], kf);
            const float *qr = &qs[wv][c * kVec];
#pragma unroll
            for (int e = 0; e < kVec; ++e) dot = fmaf(qr[e], kf[e], dot);
        }
#pragma unroll
        for (int off = kTile; off < 32; off <<= 1) dot += __shfl_xor(dot, off, 32);
        const float s = key_lane < nk ? dot * scale : -INFINITY;
        float tile_max = s;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 32));
        const float m_new = fmaxf(m, tile_max);
        const float alpha = expf(m - m_new);  // m = -inf on the first tile: alpha = 0
        const float p = key_lane < nk ? expf(s - m_new) : 0.f;
        float psum = lane < kTile ? p : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor(psum, off, 32);
        l = l * alpha + psum;
#pragma unroll
        for (int e = 0; e < kPerLane; ++e) acc[e] *= alpha;
        for (int j = 0; j < nk; ++j) {
            const float pj = __shfl(p, j);
#pragma unroll
            for (int cc = 0; cc < kLaneChunks; ++cc) {
                float vf[kVec];
                unpack<T>(vs[j][(lane * kLaneChunks + cc) ^ (j % 16)], vf);
#pragma unroll
                for (int e = 0; e < kVec; ++e) acc[cc * kVec + e] = fmaf(pj, vf[e], acc[cc * kVec + e]);
            }
        }
        m = m_new;
    }
    float *o = ws + ((t * Hq + h) * splits + sp) * (kD + 2);
    if (lane == 0) o[0] = m, o[1] = l;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) o[2 + lane * kPerLane + e] = acc[e];
}

// Merge the splits of one (token, head), divide by the softmax sum, times sigmoid(gate). The per-split
// maxima and weights are computed cooperatively (split counts reach the hundreds), then each thread
// sums its dim over the splits with independent loads.
constexpr int kMaxSplits = 1024;
template <typename T>
__global__ void __launch_bounds__(kD) attention_combine_kernel(const float *ws, int64_t splits, const T *gate,
                                                               int64_t q_ts, int64_t q_hs, int64_t Hq, T *out) {
    __shared__ float wgt[kMaxSplits], red[kD / 32];
    const int64_t h = blockIdx.x, t = blockIdx.y;
    const float *p = ws + (t * Hq + h) * splits * (kD + 2);
    const int d = threadIdx.x;
    float mx = -INFINITY;
    for (int64_t i = d; i < splits; i += kD) mx = fmaxf(mx, p[i * (kD + 2)]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down(mx, off, 32));
    const int lane = d & 31;
    const int wave = d >> 5;
    if (lane == 0) red[wave] = mx;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < kD / 32) ? red[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float M = red[0];
    __syncthreads();
    float lsum = 0.f;
    for (int64_t i = d; i < splits; i += kD) {
        const float mi = p[i * (kD + 2)];
        const float f = mi == -INFINITY ? 0.f : expf(mi - M);  // empty split: weight 0
        wgt[i] = f;
        lsum = fmaf(p[i * (kD + 2) + 1], f, lsum);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_down(lsum, off, 32);
    if (lane == 0) red[wave] = lsum;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < kD / 32) ? red[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float L = red[0];
    float a = 0.f;
#pragma unroll 8
    for (int64_t i = 0; i < splits; ++i) a = fmaf(p[i * (kD + 2) + 2 + d], wgt[i], a);
    const float g = load(gate, t * q_ts + h * q_hs + d);
    store(out, t * Hq * kD + h * kD + d, (a / L) * (1.0f / (1.0f + expf(-g))));
}

// ---- attention on the matrix units (BF16, multi-token forwards) ----------------------------------

// Block = kWWaves waves for one (split, KV head, token); all waves walk the split's keys together, 16 per tile, and
// split the head dim D: wave w owns dims [kDW w, kDW (w + 1)). Per tile:
//   S [16 heads x 16 keys] = Q K^T - each wave its kDW dims (kDW / 16 WMMA steps, K rows read straight from the
//     cache through the selection), the partials summed through LDS;
//   the online softmax in FP32, computed identically by every wave (a row's 16 keys are one lane half's 16 lanes:
//     shuffle reductions), so all waves hold the same m, l, P;
//   O [16 x kDW] += P V_w - P as BF16 hi + lo from LDS, V_w (the wave's dims of the 16 keys) transposed in LDS.
// The C layout's row for element v (2v + lane / 16) is the same in S and O, so the softmax rescale of O is per
// element. Query rows past G are zero and never written. The softmax is split by rows across the waves (shared P
// and rescale factors in LDS); Q fragments stay in registers; the next tile's K / V rows load during this tile's
// softmax and P * V; gathered key positions come from a per-chunk LDS table. Measured (gathered
// at 8k, T = 512): 139.7 ms per 512 tokens (the first version: all of D in one wave, 240 VGPRs, occupancy 2, a
// 32-way conflicted V transpose) -> 33.6 (208 VGPRs, 16.7 KB LDS, occupancy 7); the vector kernel: 142.9.
constexpr int kWWaves = 4;  // 8 waves x 32 dims measured slower (36.8 vs 33.6 ms per 512 tokens at 8k, occupancy 10 vs 7)
constexpr int kDW = kD / kWWaves;  // head dims per wave
constexpr int kEPW = 8 / kWWaves;  // softmax elements (of a lane's 8 S rows) per wave
static_assert(kDW % 32 == 0 && 8 % kWWaves == 0, "wave split: whole 32-dim steps, whole S elements per wave");
// Row stride (BF16 elements) of the transposed V tile [kDW][16 keys] and of P [16][16] in LDS: 10 words, so the 16
// rows a WMMA fragment load reads fall in distinct banks (a stride of 16 elements put rows l and l + 4 in one bank).
constexpr int kTLd = 20;
constexpr int kKeyChunk = 256;  // keys per position-table fill (a multiple of the 16-key tile)
// A workgroup barrier that orders LDS only. __syncthreads' workgroup fence also waits for every global load the
// wave has in flight (s_waitcnt vmcnt(0): in WGP mode a workgroup spans two CUs' L0 caches), so in the tile loop it
// drained the next tile's K / V prefetch at every tile. The loop's waves share only LDS, so LDS ordering suffices.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// One (attention-precision fast): P as one BF16 term - no lo half written, loaded or multiplied.
template <bool GATHER, bool One, typename R>
__global__ void __launch_bounds__(kWWaves * 32)
    attention_partial_wmma_kernel(const uint16_t *q, int64_t q_ts, int64_t q_hs, R kv, int64_t Hq, int64_t Hk, int64_t pos0, int64_t kps, float scale,
                                  float *ws, int64_t splits, const int32_t *sel, const int32_t *nsel, int64_t sel_ld,
                                  uint32_t *err) {
    __shared__ __attribute__((aligned(16))) float sp[kWWaves][8][32];  // S partials, [wave][element v][lane]
    __shared__ __attribute__((aligned(16))) uint16_t vt[kWWaves][kDW * kTLd];
    __shared__ __attribute__((aligned(16))) uint16_t ps[2][16 * kTLd];  // P hi, lo [row][key] (shared)
    __shared__ float s_alpha[16];                                        // this tile's rescale factor per row
    __shared__ int32_t kpos[kKeyChunk];  // positions of the current chunk's keys (GATHER), resolved once
    const int G = (int)(Hq / Hk), lane = (int)threadIdx.x % 32, w = (int)threadIdx.x / 32, col = lane % 16,
              half = lane / 16, d_w = w * kDW;
    const int64_t sp_i = blockIdx.x, kh = blockIdx.y, t = blockIdx.z, pos = pos0 + t;
    const int64_t tail_n = (pos + 1) % kQsaTokens, tail0 = pos + 1 - tail_n, complete = (pos + 1) / kQsaTokens;
    const int64_t g0q = kv8_math::exact_from(pos);  // 8-bit K/V: the rows this query sees exact (kernels/kv8.hpp)
    auto flag = [&](int64_t value) {
        if (atomicCAS(err, 0u, 1u) == 0u) err[1] = (uint32_t)t, err[2] = (uint32_t)value;
    };
    int64_t ns = GATHER ? nsel[t] : 0;
    if (GATHER && (ns < 0 || ns > sel_ld || ns > complete)) {
        if (threadIdx.x == 0 && sp_i == 0 && kh == 0) flag(ns);
        ns = ns < 0 ? 0 : min(ns, min(sel_ld, complete));
    }
    const int64_t n_sel_keys = ns * kQsaTokens, n_keys = GATHER ? n_sel_keys + tail_n : pos + 1;
    const int32_t *sel_row = GATHER ? sel + t * sel_ld : nullptr;
    auto key_pos = [&](int64_t j) -> int64_t {
        if (!GATHER) return j;
        if (j >= n_sel_keys) return tail0 + (j - n_sel_keys);
        const int64_t b = sel_row[j / kQsaTokens];
        if (b < 0 || b >= complete) {
            flag(b);
            return pos;
        }
        return b * kQsaTokens + j % kQsaTokens;
    };
    // This wave's Q fragments (its kDW dims; row col = query head kh G + col, zero past G), kept in registers.
    wmma_bf16x16 qf[kDW / 16];
#pragma unroll
    for (int i = 0; i < kDW / 16; ++i) {
        uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0;
        if (col < G) {
            const uint16_t *qr = q + t * q_ts + (kh * G + col) * q_hs + d_w + 16 * i;
            a0 = *reinterpret_cast<const uint4 *>(qr), a1 = *reinterpret_cast<const uint4 *>(qr + 8);
        }
        const wmma_u32x8 u = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        qf[i] = __builtin_bit_cast(wmma_bf16x16, u);
    }
    const int64_t kb = sp_i * kps, ke = min(n_keys, kb + kps);
    // The softmax is split by rows: wave w owns elements v = kEPW w + e (rows 2v, 2v + 1) - their m, l and P.
    float m[kEPW], l[kEPW];
    wmma_f32x8 o[kDW / 16];
#pragma unroll
    for (int e = 0; e < kEPW; ++e) m[e] = -INFINITY, l[e] = 0.f;
#pragma unroll
    for (int j = 0; j < kDW / 16; ++j) o[j] = wmma_f32x8{};
    uint16_t *vtw = vt[w], *ph = ps[0], *pl = ps[1];
    // A key's position: through the chunk table (GATHER) or the key index itself.
    auto kpos_of = [&](int64_t j) -> int64_t { return GATHER ? (int64_t)kpos[(j - kb) % kKeyChunk] : j; };
    auto fill_table = [&](int64_t j0) {  // positions of keys [j0, j0 + kKeyChunk) - all threads, caller syncs
        for (int64_t j = j0 + threadIdx.x; j < min(ke, j0 + kKeyChunk); j += kWWaves * 32)
            kpos[(j - kb) % kKeyChunk] = (int32_t)key_pos(j);
    };
    // Software-pipelined: tile k0 + 16's K rows (this lane's key column, the wave's dims) and V chunks (keys 2 kp2,
    // 2 kp2 + 1 of the lane's dim group) are loaded into registers while tile k0 computes.
    const int kp2 = lane % 8, grp = lane / 8;
    constexpr int kKLoads = kDW / 8, kVLoads = kDW / 32;  // 16-byte loads per lane per tile
    uint4 kr[kKLoads], va_r[kVLoads], vb_r[kVLoads];
    auto load_tile = [&](int64_t t0) {
        const int64_t krow = t0 + col < ke ? kpos_of(t0 + col) : pos, c0 = kh * kD + d_w;
#pragma unroll
        for (int i = 0; i < kKLoads; ++i) kr[i] = kv.k(krow, c0 + 8 * i, g0q);
        const int64_t ka = t0 + 2 * kp2, kb2 = ka + 1;
        const int64_t ra = ka < ke ? kpos_of(ka) : -1, rb2 = kb2 < ke ? kpos_of(kb2) : -1;
#pragma unroll
        for (int i = 0; i < kVLoads; ++i) {
            const int c = grp * 8 + 32 * i;
            va_r[i] = ra >= 0 ? kv.v(ra, c0 + c, g0q) : make_uint4(0, 0, 0, 0);
            vb_r[i] = rb2 >= 0 ? kv.v(rb2, c0 + c, g0q) : make_uint4(0, 0, 0, 0);
        }
    };
    if (GATHER && kb < ke) fill_table(kb);
    __syncthreads();  // the first key table staged
    if (kb < ke) load_tile(kb);
    for (int64_t k0 = kb; k0 < ke; k0 += 16) {
        const bool valid = k0 + col < ke;
        // This wave's part of S from the loaded K rows: its kDW dims, as two independent accumulation chains.
        wmma_f32x8 s0 = {}, s1 = {};
#pragma unroll
        for (int d0 = 0; d0 < kDW; d0 += 32) {
            const int i0 = d0 / 8;
            const wmma_u32x8 bu0 = {kr[i0].x, kr[i0].y, kr[i0].z, kr[i0].w, kr[i0 + 1].x, kr[i0 + 1].y, kr[i0 + 1].z, kr[i0 + 1].w};
            const wmma_u32x8 bu1 = {kr[i0 + 2].x, kr[i0 + 2].y, kr[i0 + 2].z, kr[i0 + 2].w,
                                    kr[i0 + 3].x, kr[i0 + 3].y, kr[i0 + 3].z, kr[i0 + 3].w};
            s0 = wmma_bf16(qf[d0 / 16], __builtin_bit_cast(wmma_bf16x16, bu0), s0);
            s1 = wmma_bf16(qf[d0 / 16 + 1], __builtin_bit_cast(wmma_bf16x16, bu1), s1);
        }
#pragma unroll
        for (int v = 0; v < 8; ++v) sp[w][v][lane] = s0[v] + s1[v];
        // V transposed, the wave's dims: each dim's pair of keys (2 kp2, 2 kp2 + 1) as one 32-bit word.
        {
            uint32_t *vt32 = reinterpret_cast<uint32_t *>(vtw);
#pragma unroll
            for (int ci = 0; ci < kVLoads; ++ci) {
                const int c = grp * 8 + 32 * ci;
                const uint32_t aw[4] = {va_r[ci].x, va_r[ci].y, va_r[ci].z, va_r[ci].w};
                const uint32_t bw[4] = {vb_r[ci].x, vb_r[ci].y, vb_r[ci].z, vb_r[ci].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    vt32[((c + 2 * i) * kTLd + 2 * kp2) / 2] = (aw[i] & 0xffffu) | (bw[i] << 16);
                    vt32[((c + 2 * i + 1) * kTLd + 2 * kp2) / 2] = (aw[i] >> 16) | (bw[i] & 0xffff0000u);
                }
            }
        }
        // The next tile's rows into the same registers: in flight during this tile's softmax and P * V. (No barrier
        // before a table refill: the previous tile's table reads happened before its second barrier.)
        const int64_t k1 = k0 + 16;
        if (k1 < ke) {
            if (GATHER && (k1 - kb) % kKeyChunk == 0) {
                fill_table(k1);
                lds_barrier();
            }
            load_tile(k1);
        }
        lds_barrier();  // all S partials written; every wave done with the previous tile's P
        // This wave's rows: full S (the partials summed in wave order), online softmax, P hi / lo and the rescale
        // factors into LDS for every wave.
#pragma unroll
        for (int e = 0; e < kEPW; ++e) {
            const int v = kEPW * w + e, r = 2 * v + half;
            float sfull = sp[0][v][lane];
#pragma unroll
            for (int u = 1; u < kWWaves; ++u) sfull += sp[u][v][lane];
            const float sv = valid ? sfull * scale : -INFINITY;
            float mx = sv;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float m_new = fmaxf(m[e], mx);
            const float alpha = expf(m[e] - m_new);  // m = -inf on the first tile: 0
            const float p = valid ? expf(sv - m_new) : 0.f;
            float sum = p;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor(sum, off, 32);
            l[e] = l[e] * alpha + sum;
            m[e] = m_new;
            const uint32_t hb = __float_as_uint(p);
            const uint16_t hi = (uint16_t)((hb + 0x7fffu + ((hb >> 16) & 1u)) >> 16);  // p >= 0, finite: RNE
            const uint32_t lb = __float_as_uint(p - __uint_as_float((uint32_t)hi << 16));
            ph[r * kTLd + col] = hi;
            if constexpr (!One) pl[r * kTLd + col] = (uint16_t)((lb + 0x7fffu + ((lb >> 16) & 1u)) >> 16);
            else (void)lb;
            if (col == 0) s_alpha[r] = alpha;
        }
        lds_barrier();  // P and the rescale factors complete
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const float alpha = s_alpha[2 * v + half];
#pragma unroll
            for (int j = 0; j < kDW / 16; ++j) o[j][v] *= alpha;
        }
        const wmma_bf16x16 a_hi = wmma_load_rows(ph, kTLd);
        wmma_bf16x16 a_lo{};
        if constexpr (!One) a_lo = wmma_load_rows(pl, kTLd);
#pragma unroll
        for (int j = 0; j < kDW / 16; ++j) {
            const wmma_bf16x16 b = wmma_load_rows(vtw + j * 16 * kTLd, kTLd);
            o[j] = wmma_bf16(a_hi, b, o[j]);
            if constexpr (!One) o[j] = wmma_bf16(a_lo, b, o[j]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // V reads done before this wave's next V writes
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    }
    // The block's partial for rows < G: m, l from the wave owning the row, O from each wave's dims.
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int r = 2 * v + half;
        if (r >= G) continue;
        float *out = ws + ((t * Hq + kh * G + r) * splits + sp_i) * (kD + 2);
        if (v / kEPW == w && col == 0) out[0] = m[v % kEPW], out[1] = l[v % kEPW];
#pragma unroll
        for (int j = 0; j < kDW / 16; ++j) out[2 + d_w + j * 16 + col] = o[j][v];
    }
}

// Keys per split. Measured on gfx1151: 80 blocks - 2 per
// CU of its 40 - is best; fewer, longer contiguous key ranges beat more splits (0.8B at 32K context: 92%
// of read peak at 80 blocks, 65% at 160, 52% at 320, 78% at 40); at least 2 tiles per split.
// HIP's multiprocessor count is in WGPs on this device (20 = 40 CUs / 2), hence 4 per reported unit.
int device_units() {
    static const int cus = [] {
        int dev = 0, n = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "attention: current device");
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev), "attention: multiprocessor count");
        STRIX_CHECK(n >= 1, "attention: device reports ", n, " multiprocessors");
        return n;
    }();
    return cus;
}

// n: the most keys any query of the call attends to.
int64_t keys_per_split(const AttentionShape &s, int64_t n) {
    const int64_t want_blocks = 4 * (int64_t)device_units();
    int64_t splits = want_blocks / (s.Hk * s.T);
    if (splits < 1) splits = 1;
    int64_t kps = (n + splits - 1) / splits;
    kps = (kps + kTile - 1) / kTile * kTile;
    if (kps < 2 * kTile) kps = 2 * kTile;
    if ((n + kps - 1) / kps > kMaxSplits) kps = ((n + kMaxSplits - 1) / kMaxSplits + kTile - 1) / kTile * kTile;
    return kps;
}

void check_shape(const AttentionShape &s) {
    STRIX_CHECK(s.T >= 1 && s.T <= 65535, "attention: T = ", s.T, ", expected 1..65535");
    STRIX_CHECK(s.pos0 >= 0 && s.pos0 + s.T <= kMaxPos, "attention: positions [", s.pos0, ", ", s.pos0 + s.T,
                ") outside [0, ", kMaxPos, ")");
    STRIX_CHECK(s.D == kD, "attention: head dim D = ", s.D, ", this kernel is built for ", kD);
    STRIX_CHECK(s.Hk >= 1 && s.Hq >= s.Hk && s.Hq % s.Hk == 0 && s.Hq / s.Hk <= kMaxGroup, "attention: Hq = ", s.Hq,
                ", Hk = ", s.Hk, " - need Hq a multiple of Hk with Hq/Hk <= ", kMaxGroup);
    STRIX_CHECK(s.q_hs >= s.D && s.q_ts >= (s.Hq - 1) * s.q_hs + s.D, "attention: q strides (token ", s.q_ts,
                ", head ", s.q_hs, ") overlap for ", s.Hq, " heads of ", s.D);
}

}  // namespace

void qk_norm_rope(const void *in, int64_t in_ts, int64_t in_hs, void *out, int64_t out_ts, int64_t out_hs, int64_t T,
                  int64_t heads, int64_t D, const float *w, float eps, int64_t rot_dim, int64_t pos0,
                  const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale, const MropePos &pos3) {
    const char *fn = "qk_norm_rope";
    STRIX_CHECK(in != nullptr && out != nullptr && w != nullptr && inv_freq != nullptr,
                "qk_norm_rope: null pointer (in=", in, ", out=", out, ", w=", (const void *)w,
                ", inv_freq=", (const void *)inv_freq, ")");
    STRIX_CHECK(T >= 1 && heads >= 1 && T * heads <= 0x7fffffff, "qk_norm_rope: T = ", T, ", heads = ", heads);
    STRIX_CHECK(D >= 2 && D <= 4096, "qk_norm_rope: D = ", D, ", expected 2..4096");
    STRIX_CHECK(rot_dim >= 2 && rot_dim % 2 == 0 && rot_dim <= D, "qk_norm_rope: rot_dim = ", rot_dim,
                ", expected even, 2..D = ", D);
    STRIX_CHECK(std::isfinite(eps) && eps > 0, "qk_norm_rope: eps = ", eps);
    STRIX_CHECK(std::isfinite(cos_sin_scale) && cos_sin_scale >= 1.0f && cos_sin_scale <= 2.0f,
                "qk_norm_rope: cos_sin_scale = ", cos_sin_scale, ", expected finite, 1..2 (YaRN's attention factor)");
    STRIX_CHECK(pos0 >= 0 && pos0 + T <= kMaxPos, "qk_norm_rope: positions [", pos0, ", ", pos0 + T, ") outside [0, ",
                kMaxPos, ")");
    if (pos3.pos3 != nullptr)
        STRIX_CHECK(pos3.rows >= T && pos3.base == pos0 && reinterpret_cast<uintptr_t>(pos3.pos3) % 8 == 0, fn,
                    ": mRoPE positions cover rows [", pos3.base, ", ", pos3.base + pos3.rows, "), expected to start at ",
                    pos0, " and cover ", T, " (aligned to ", 8, " B)");
    for (auto [ts, hs, what] : {std::tuple{in_ts, in_hs, "in"}, std::tuple{out_ts, out_hs, "out"}})
        STRIX_CHECK((heads == 1 || hs >= D) && hs >= 0 && ts >= (heads - 1) * hs + D, "qk_norm_rope: ", what,
                    " strides (token ", ts, ", head ", hs, ") overlap for ", heads, " heads of ", D);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "qk_norm_rope: unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2;
    const bool in_place = in == out && in_ts == out_ts && in_hs == out_hs;
    if (!in_place) {
        auto ib = reinterpret_cast<uintptr_t>(in), ob = reinterpret_cast<uintptr_t>(out);
        const uintptr_t ie = ib + (uintptr_t)(((T - 1) * in_ts + (heads - 1) * in_hs + D) * es),
                        oe = ob + (uintptr_t)(((T - 1) * out_ts + (heads - 1) * out_hs + D) * es);
        STRIX_CHECK(oe <= ib || ie <= ob, "qk_norm_rope: out [", out, ", ", (const void *)oe, ") overlaps in [", in,
                    ", ", (const void *)ie, ") without being exactly in place");
    }
    const size_t lds = (size_t)(D + kNormThreads) * 4;
    dim3 grid((unsigned)(T * heads)), block(kNormThreads);
    if (act == Act::F32)
        qk_norm_rope_kernel<float><<<grid, block, lds, stream>>>(static_cast<const float *>(in), in_ts, in_hs,
                                                                 static_cast<float *>(out), out_ts, out_hs, heads, D, w,
                                                                 eps, rot_dim / 2, pos0, pos3.pos3, inv_freq,
                                                                 cos_sin_scale);
    else
        qk_norm_rope_kernel<uint16_t><<<grid, block, lds, stream>>>(static_cast<const uint16_t *>(in), in_ts, in_hs,
                                                                    static_cast<uint16_t *>(out), out_ts, out_hs, heads,
                                                                    D, w, eps, rot_dim / 2, pos0, pos3.pos3, inv_freq,
                                                                    cos_sin_scale);
    STRIX_HIP_CHECK_LAUNCH("qk_norm_rope ", act_name(act), " T=", T, " heads=", heads, " D=", D);
}

void qk_norm_rope_jobs(const NormRopeJob *jobs, int n, int64_t T, float eps, int64_t rot_dim, int64_t pos0,
                       const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale, const MropePos &pos3) {
    const char *fn = "qk_norm_rope_jobs";
    STRIX_CHECK(jobs != nullptr && n >= 1 && n <= kMaxNormRopeJobs, fn, ": ", n, " jobs (", (const void *)jobs,
                "), expected 1..", kMaxNormRopeJobs);
    STRIX_CHECK(inv_freq != nullptr, fn, ": null inv_freq");
    STRIX_CHECK(T >= 1 && pos0 >= 0 && pos0 + T <= kMaxPos, fn, ": T = ", T, ", positions [", pos0, ", ", pos0 + T,
                ") outside [0, ", kMaxPos, ")");
    STRIX_CHECK(std::isfinite(eps) && eps > 0, fn, ": eps = ", eps);
    STRIX_CHECK(std::isfinite(cos_sin_scale) && cos_sin_scale >= 1.0f && cos_sin_scale <= 2.0f, fn,
                ": cos_sin_scale = ", cos_sin_scale, ", expected finite, 1..2 (YaRN's attention factor)");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    if (pos3.pos3 != nullptr)
        STRIX_CHECK(pos3.rows >= T && pos3.base == pos0 && reinterpret_cast<uintptr_t>(pos3.pos3) % 8 == 0, fn,
                    ": mRoPE positions cover rows [", pos3.base, ", ", pos3.base + pos3.rows, "), expected to start at ",
                    pos0, " and cover ", T, " (aligned to ", 8, " B)");
    const int64_t es = act == Act::F32 ? 4 : 2;
    NormRopeJobsArg a{};
    a.n = n;
    int64_t blocks = 0;
    // A side of a job: T rows of `width` bytes, `ts` bytes apart, from `b`.
    struct Side {
        uintptr_t b;
        int64_t ts, width;
    };
    std::vector<Side> ins, outs;
    for (int k = 0; k < n; ++k) {
        const NormRopeJob &j = jobs[k];
        STRIX_CHECK(j.in != nullptr && j.out != nullptr, fn, ": job ", k, ": null pointer (in=", j.in, ", out=", j.out, ")");
        STRIX_CHECK(j.heads >= 1 && j.D >= 2 && j.D <= 4096, fn, ": job ", k, ": heads = ", j.heads, ", D = ", j.D,
                    " (expected heads >= 1, D 2..4096)");
        STRIX_CHECK(j.w == nullptr || (rot_dim >= 2 && rot_dim % 2 == 0 && rot_dim <= j.D), fn, ": job ", k,
                    ": rot_dim = ", rot_dim, ", expected even, 2..D = ", j.D);
        for (auto [ts, hs, what] : {std::tuple{j.in_ts, j.in_hs, "in"}, std::tuple{j.out_ts, j.out_hs, "out"}})
            STRIX_CHECK((j.heads == 1 || hs >= j.D) && hs >= 0 && ts >= (j.heads - 1) * hs + j.D, fn, ": job ", k, " ",
                        what, " strides (token ", ts, ", head ", hs, ") overlap for ", j.heads, " heads of ", j.D);
        ins.push_back({reinterpret_cast<uintptr_t>(j.in), j.in_ts * es, ((j.heads - 1) * j.in_hs + j.D) * es});
        outs.push_back({reinterpret_cast<uintptr_t>(j.out), j.out_ts * es, ((j.heads - 1) * j.out_hs + j.D) * es});
        a.j[k] = j;
        a.max_d = std::max(a.max_d, j.D);
        blocks += T * j.heads;
    }
    // Disjoint if the bounding ranges are; else (jobs interleaved in the same token rows) if the token strides are
    // equal and the column ranges, taken modulo the stride, don't meet. Anything else is refused as unprovable.
    auto disjoint = [&](const Side &x, const Side &y) {
        const uintptr_t xe = x.b + (uintptr_t)((T - 1) * x.ts + x.width), ye = y.b + (uintptr_t)((T - 1) * y.ts + y.width);
        if (xe <= y.b || ye <= x.b) return true;
        if (x.ts != y.ts || x.ts <= 0) return false;
        const int64_t d = (int64_t)(x.b - y.b), c = ((d % x.ts) + x.ts) % x.ts;  // x's columns start at c in y's rows
        return c >= y.width && c + x.width <= x.ts;
    };
    for (int k = 0; k < n; ++k)
        for (int m = 0; m < n; ++m) {
            const bool self_in_place = k == m && jobs[k].in == jobs[k].out && jobs[k].in_ts == jobs[k].out_ts &&
                                       jobs[k].in_hs == jobs[k].out_hs;
            STRIX_CHECK(self_in_place || disjoint(outs[k], ins[m]), fn, ": job ", k, "'s out overlaps job ", m, "'s in");
            STRIX_CHECK(k == m || disjoint(outs[k], outs[m]), fn, ": jobs ", k, " and ", m, " write overlapping outputs");
        }
    STRIX_CHECK(blocks <= 0x7fffffff, fn, ": ", blocks, " blocks");
    a.pos3 = pos3.pos3;  // validated above; the rows start at pos0, as the kernel indexes them
    const size_t lds = (size_t)(a.max_d + kNormThreads) * 4;
    if (act == Act::F32)
        qk_norm_rope_jobs_kernel<float><<<(unsigned)blocks, kNormThreads, lds, stream>>>(a, T, eps, rot_dim / 2, pos0,
                                                                                          inv_freq, cos_sin_scale);
    else
        qk_norm_rope_jobs_kernel<uint16_t><<<(unsigned)blocks, kNormThreads, lds, stream>>>(a, T, eps, rot_dim / 2,
                                                                                             pos0, inv_freq, cos_sin_scale);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", T, " jobs=", n);
}

size_t attention_workspace_bytes(const AttentionShape &s) {
    check_shape(s);
    const int64_t kps = keys_per_split(s, s.pos0 + s.T), splits = (s.pos0 + s.T + kps - 1) / kps;
    return (size_t)(s.T * s.Hq * splits * (kD + 2)) * 4;
}

namespace {

// The K/V a call reads: the BF16 / FP32 cache (k, v: [capacity, Hk, D]) or the 8-bit one (q8).
struct KvArg {
    const void *k = nullptr, *v = nullptr;
    const Kv8View *q8 = nullptr;
};

// Pointers, alignment and - for the 8-bit view - that it holds every row [0, pos0 + T): storage below g0, the tail
// from g0 on (at most kKv8Group rows).
void check_kv(const char *fn, const KvArg &kv, int64_t capacity, const AttentionShape &s) {
    if (kv.q8 == nullptr) {
        STRIX_CHECK(kv.k != nullptr && kv.v != nullptr, fn, ": null K/V cache (k=", kv.k, ", v=", kv.v, ")");
        STRIX_CHECK(reinterpret_cast<uintptr_t>(kv.k) % 16 == 0 && reinterpret_cast<uintptr_t>(kv.v) % 16 == 0, fn,
                    ": K/V caches (", kv.k, ", ", kv.v, ") must be 16-byte aligned for vector loads");
        STRIX_CHECK(capacity >= s.pos0 + s.T, fn, ": KV cache capacity ", capacity, " < pos0 + T = ", s.pos0 + s.T);
        return;
    }
    const Kv8View &x = *kv.q8;
    const bool need_tail = s.pos0 % kKv8Group != 0;
    STRIX_CHECK(x.store != nullptr && x.k_new != nullptr && x.v_new != nullptr && (x.tail_in != nullptr || !need_tail),
                fn, ": 8-bit K/V view with a null pointer (store=", (const void *)x.store, ", tail_in=", x.tail_in,
                ", k_new=", x.k_new, ", v_new=", x.v_new, ")");
    const auto a16 = [](const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    STRIX_CHECK(a16(x.store) && a16(x.tail_in) && a16(x.k_new) && a16(x.v_new), fn,
                ": 8-bit K/V store / tail / new rows must be 16-byte aligned (", (const void *)x.store, ", ", x.tail_in,
                ", ", x.k_new, ", ", x.v_new, ")");
    STRIX_CHECK(x.W == s.Hk * s.D, fn, ": 8-bit K/V view has rows of ", x.W, " values, the shape Hk * D = ", s.Hk * s.D);
    STRIX_CHECK(x.capacity % kKv8Group == 0 && x.capacity >= s.pos0 + s.T, fn, ": 8-bit K/V capacity ", x.capacity,
                ", expected a multiple of ", kKv8Group, " >= pos0 + T = ", s.pos0 + s.T);
    STRIX_CHECK(x.pos0 == s.pos0, fn, ": 8-bit K/V view is for a call at ", x.pos0, ", the shape's pos0 is ", s.pos0);
    STRIX_CHECK(x.new_ts >= x.W && x.new_ts % 8 == 0, fn, ": 8-bit K/V new-row stride ", x.new_ts,
                ", expected >= W = ", x.W, " and a multiple of 8 (16-byte rows)");
    if (x.staged_k || x.staged_v || x.staged_rows) {
        const int64_t need = kv8_math::exact_from(s.pos0 + s.T - 1);
        STRIX_CHECK(x.staged_k != nullptr && x.staged_v != nullptr, fn, ": 8-bit K/V view staged with a null pointer "
                    "(staged_k=", x.staged_k, ", staged_v=", x.staged_v, ")");
        STRIX_CHECK(a16(x.staged_k) && a16(x.staged_v), fn, ": staged K/V (", x.staged_k, ", ", x.staged_v,
                    ") must be 16-byte aligned");
        STRIX_CHECK(x.staged_rows >= need && x.staged_rows <= x.capacity, fn, ": 8-bit K/V view stages ",
                    x.staged_rows, " rows, the call's last query needs ", need, " (capacity ", x.capacity, ")");
    }
}

// Calls f(reader) with the reader type for kv and the element type T.
template <typename T, typename F>
void with_reader(const KvArg &kv, int64_t W, F &&f) {
    if (kv.q8 && kv.q8->staged_k) f(kv8_math::StagedReader<T>(*kv.q8));
    else if (kv.q8) f(kv8_math::Kv8Reader<T>(*kv.q8));
    else f(kv8_math::PlainReader<T>{static_cast<const T *>(kv.k), static_cast<const T *>(kv.v), W});
}

int wave_size(const char *fn) {
    static const int warp = [] {
        int dev = 0, w = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "attention: current device");
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&w, hipDeviceAttributeWarpSize, dev), "attention: warp size");
        return w;
    }();
    STRIX_CHECK(warp == 32, fn, ": built for wave32 (2 halves of D per key), device wave size ", warp);
    return warp;
}

void attention_impl(const void *q, const void *gate, const KvArg &kv, int64_t capacity, const AttentionShape &s,
                    float scale, void *out, float *workspace, size_t workspace_bytes, Act act, hipStream_t stream) {
    const char *fn = "attention";
    STRIX_CHECK(q != nullptr && gate != nullptr && out != nullptr && workspace != nullptr, fn, ": null pointer (q=", q,
                ", gate=", gate, ", out=", out, ", ws=", (const void *)workspace, ")");
    check_shape(s);
    check_kv(fn, kv, capacity, s);
    STRIX_CHECK(std::isfinite(scale) && scale > 0, fn, ": scale = ", scale);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    const size_t need = attention_workspace_bytes(s);
    STRIX_CHECK(workspace_bytes >= need, fn, ": workspace ", workspace_bytes, " bytes, need ", need);
    const int warp = wave_size(fn);
    const int64_t kps = keys_per_split(s, s.pos0 + s.T), splits = (s.pos0 + s.T + kps - 1) / kps;
    dim3 grid((unsigned)splits, (unsigned)s.Hk, (unsigned)s.T), block((unsigned)(warp * (s.Hq / s.Hk)));
    dim3 cgrid((unsigned)s.Hq, (unsigned)s.T), cblock(kD);
    const size_t q_lds = (size_t)(s.Hq / s.Hk) * kD * 4;
    if (act == Act::F32) {
        with_reader<float>(kv, s.Hk * kD, [&](auto rd) {
            attention_partial_kernel<float, false><<<grid, block, q_lds, stream>>>(
                static_cast<const float *>(q), s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace, splits,
                nullptr, nullptr, 0, nullptr);
        });
        attention_combine_kernel<float><<<cgrid, cblock, 0, stream>>>(workspace, splits, static_cast<const float *>(gate),
                                                                      s.q_ts, s.q_hs, s.Hq, static_cast<float *>(out));
    } else {
        with_reader<uint16_t>(kv, s.Hk * kD, [&](auto rd) {
            attention_partial_kernel<uint16_t, false><<<grid, block, q_lds, stream>>>(
                static_cast<const uint16_t *>(q), s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace,
                splits, nullptr, nullptr, 0, nullptr);
        });
        attention_combine_kernel<uint16_t><<<cgrid, cblock, 0, stream>>>(workspace, splits,
                                                                         static_cast<const uint16_t *>(gate), s.q_ts,
                                                                         s.q_hs, s.Hq, static_cast<uint16_t *>(out));
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), kv.q8 ? " kv8" : "", " T=", s.T, " pos0=", s.pos0, " Hq=", s.Hq,
                           " Hk=", s.Hk, " splits=", splits);
}

}  // namespace

void attention(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
               const AttentionShape &s, float scale, void *out, float *workspace, size_t workspace_bytes, Act act,
               hipStream_t stream) {
    attention_impl(q, gate, KvArg{k_cache, v_cache, nullptr}, capacity, s, scale, out, workspace, workspace_bytes, act,
                   stream);
}

void attention(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s, float scale, void *out,
               float *workspace, size_t workspace_bytes, Act act, hipStream_t stream) {
    attention_impl(q, gate, KvArg{nullptr, nullptr, &kv}, kv.capacity, s, scale, out, workspace, workspace_bytes, act,
                   stream);
}

namespace {
int64_t gathered_max_keys(const AttentionShape &s, int64_t sel_ld) {
    const int64_t budget = sel_ld * kQsaTokens + (kQsaTokens - 1);  // selected blocks + a tail
    return s.pos0 + s.T < budget ? s.pos0 + s.T : budget;
}
}  // namespace

size_t attention_gathered_workspace_bytes(const AttentionShape &s, int64_t sel_ld) {
    check_shape(s);
    STRIX_CHECK(sel_ld >= 1 && sel_ld <= 65536, "attention_gathered_workspace_bytes: sel_ld = ", sel_ld,
                ", expected 1..65536");
    const int64_t n = gathered_max_keys(s, sel_ld), kps = keys_per_split(s, n), splits = (n + kps - 1) / kps;
    return (size_t)(s.T * s.Hq * splits * (kD + 2)) * 4;
}

namespace {
void attention_gathered_impl(const void *q, const void *gate, const KvArg &kv, int64_t capacity,
                             const AttentionShape &s, const int32_t *sel, const int32_t *nsel, int64_t sel_ld,
                             float scale, void *out, float *workspace, size_t workspace_bytes, uint32_t *err, Act act,
                             hipStream_t stream) {
    const char *fn = "attention_gathered";
    STRIX_CHECK(q != nullptr && gate != nullptr && out != nullptr && workspace != nullptr && sel != nullptr &&
                    nsel != nullptr && err != nullptr,
                fn, ": null pointer (q=", q, ", gate=", gate, ", out=", out, ", ws=", (const void *)workspace, ", sel=",
                (const void *)sel, ", nsel=", (const void *)nsel, ", err=", (const void *)err, ")");
    check_shape(s);
    check_kv(fn, kv, capacity, s);
    STRIX_CHECK(std::isfinite(scale) && scale > 0, fn, ": scale = ", scale);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    const size_t need = attention_gathered_workspace_bytes(s, sel_ld);
    STRIX_CHECK(workspace_bytes >= need, fn, ": workspace ", workspace_bytes, " bytes, need ", need);
    const int warp = wave_size(fn);
    const int64_t n = gathered_max_keys(s, sel_ld), kps = keys_per_split(s, n), splits = (n + kps - 1) / kps;
    dim3 grid((unsigned)splits, (unsigned)s.Hk, (unsigned)s.T), block((unsigned)(warp * (s.Hq / s.Hk)));
    dim3 cgrid((unsigned)s.Hq, (unsigned)s.T), cblock(kD);
    const size_t q_lds = (size_t)(s.Hq / s.Hk) * kD * 4;
    if (act == Act::F32) {
        with_reader<float>(kv, s.Hk * kD, [&](auto rd) {
            attention_partial_kernel<float, true><<<grid, block, q_lds, stream>>>(
                static_cast<const float *>(q), s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace, splits,
                sel, nsel, sel_ld, err);
        });
        attention_combine_kernel<float><<<cgrid, cblock, 0, stream>>>(workspace, splits, static_cast<const float *>(gate),
                                                                      s.q_ts, s.q_hs, s.Hq, static_cast<float *>(out));
    } else {
        with_reader<uint16_t>(kv, s.Hk * kD, [&](auto rd) {
            attention_partial_kernel<uint16_t, true><<<grid, block, q_lds, stream>>>(
                static_cast<const uint16_t *>(q), s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace,
                splits, sel, nsel, sel_ld, err);
        });
        attention_combine_kernel<uint16_t><<<cgrid, cblock, 0, stream>>>(workspace, splits,
                                                                         static_cast<const uint16_t *>(gate), s.q_ts,
                                                                         s.q_hs, s.Hq, static_cast<uint16_t *>(out));
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), kv.q8 ? " kv8" : "", " T=", s.T, " pos0=", s.pos0, " Hq=", s.Hq,
                           " Hk=", s.Hk, " splits=", splits);
}
}  // namespace

void attention_gathered(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
                        const AttentionShape &s, const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale,
                        void *out, float *workspace, size_t workspace_bytes, uint32_t *err, Act act,
                        hipStream_t stream) {
    attention_gathered_impl(q, gate, KvArg{k_cache, v_cache, nullptr}, capacity, s, sel, nsel, sel_ld, scale, out,
                            workspace, workspace_bytes, err, act, stream);
}

void attention_gathered(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s,
                        const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale, void *out,
                        float *workspace, size_t workspace_bytes, uint32_t *err, Act act, hipStream_t stream) {
    attention_gathered_impl(q, gate, KvArg{nullptr, nullptr, &kv}, kv.capacity, s, sel, nsel, sel_ld, scale, out,
                            workspace, workspace_bytes, err, act, stream);
}

void check_attention_error(const uint32_t *err, const char *what) {
    STRIX_CHECK(err != nullptr, "check_attention_error(", what, "): null err");
    uint32_t e[3];
    STRIX_HIP_CHECK(hipMemcpy(e, err, sizeof(e), hipMemcpyDeviceToHost), "reading attention error word for ", what);
    report_attention_error(e, what);
}

void report_attention_error(const uint32_t e[3], const char *what) {
    STRIX_CHECK(e[0] == 0, what, ": QSA selection out of range for token ", e[1], " (value ", (int32_t)e[2],
                " - a block id outside the query's complete blocks, or a count above them); outputs are wrong");
}


namespace {
void wmma_attention_launch(bool gather, const void *q, const void *gate, const KvArg &kv, const AttentionShape &s,
                           int64_t n, const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale, void *out,
                           float *workspace, uint32_t *err, hipStream_t stream, AttnPrecision precision) {
    const int64_t kps = keys_per_split(s, n), splits = (n + kps - 1) / kps;
    dim3 grid((unsigned)splits, (unsigned)s.Hk, (unsigned)s.T), block(kWWaves * 32);
    dim3 cgrid((unsigned)s.Hq, (unsigned)s.T), cblock(kD);
    const auto *qb = static_cast<const uint16_t *>(q);
    STRIX_CHECK(precision == AttnPrecision::Exact || precision == AttnPrecision::Fast,
                "attention (WMMA): unknown precision ", (int)precision);
    with_reader<uint16_t>(kv, s.Hk * kD, [&](auto rd) {
        using R = decltype(rd);
        auto partial = [&](auto one) {
            constexpr bool One = decltype(one)::value;
            if (gather)
                attention_partial_wmma_kernel<true, One, R><<<grid, block, 0, stream>>>(
                    qb, s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace, splits, sel, nsel, sel_ld, err);
            else
                attention_partial_wmma_kernel<false, One, R><<<grid, block, 0, stream>>>(
                    qb, s.q_ts, s.q_hs, rd, s.Hq, s.Hk, s.pos0, kps, scale, workspace, splits, nullptr, nullptr, 0,
                    nullptr);
        };
        if (precision == AttnPrecision::Fast) partial(std::true_type{});
        else partial(std::false_type{});
    });
    attention_combine_kernel<uint16_t><<<cgrid, cblock, 0, stream>>>(workspace, splits,
                                                                     static_cast<const uint16_t *>(gate), s.q_ts,
                                                                     s.q_hs, s.Hq, static_cast<uint16_t *>(out));
}
void wmma_attention_checks(const char *fn, const void *q, const void *gate, const KvArg &kv, const void *out,
                           const float *workspace, int64_t capacity, const AttentionShape &s, float scale, Act act) {
    STRIX_CHECK(q != nullptr && gate != nullptr && out != nullptr && workspace != nullptr, fn, ": null pointer (q=", q,
                ", gate=", gate, ", out=", out, ", ws=", (const void *)workspace, ")");
    check_shape(s);
    STRIX_CHECK(act == Act::BF16, fn, ": BF16 activations only (got ", act_name(act), ")");
    check_kv(fn, kv, capacity, s);
    STRIX_CHECK(std::isfinite(scale) && scale > 0, fn, ": scale = ", scale);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(q) % 16 == 0 && s.q_ts % 8 == 0 && s.q_hs % 8 == 0, fn, ": q ", q,
                " and its strides (token ", s.q_ts, ", head ", s.q_hs, ") must be 16-byte aligned");
    int dev = 0, warp = 0;
    STRIX_HIP_CHECK(hipGetDevice(&dev), fn, ": current device");
    STRIX_HIP_CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev), fn, ": warp size");
    STRIX_CHECK(warp == 32, fn, ": needs wave32 (gfx11 WMMA), device wave size ", warp);
}
size_t wmma_workspace(const AttentionShape &s, int64_t n) {
    const int64_t kps = keys_per_split(s, n), splits = (n + kps - 1) / kps;
    return (size_t)(s.T * s.Hq * splits * (kD + 2)) * 4;
}

void attention_wmma_impl(const void *q, const void *gate, const KvArg &kv, int64_t capacity, const AttentionShape &s,
                         float scale, void *out, float *workspace, size_t workspace_bytes, Act act, hipStream_t stream,
                         AttnPrecision precision) {
    const char *fn = "attention_wmma";
    wmma_attention_checks(fn, q, gate, kv, out, workspace, capacity, s, scale, act);
    const size_t need = attention_wmma_workspace_bytes(s);
    STRIX_CHECK(workspace_bytes >= need, fn, ": workspace ", workspace_bytes, " bytes, need ", need);
    wmma_attention_launch(false, q, gate, kv, s, s.pos0 + s.T, nullptr, nullptr, 0, scale, out, workspace, nullptr,
                          stream, precision);
    STRIX_HIP_CHECK_LAUNCH(fn, kv.q8 ? " kv8" : "", " T=", s.T, " pos0=", s.pos0, " Hq=", s.Hq, " Hk=", s.Hk);
}

void attention_gathered_wmma_impl(const void *q, const void *gate, const KvArg &kv, int64_t capacity,
                                  const AttentionShape &s, const int32_t *sel, const int32_t *nsel, int64_t sel_ld,
                                  float scale, void *out, float *workspace, size_t workspace_bytes, uint32_t *err,
                                  Act act, hipStream_t stream, AttnPrecision precision) {
    const char *fn = "attention_gathered_wmma";
    wmma_attention_checks(fn, q, gate, kv, out, workspace, capacity, s, scale, act);
    STRIX_CHECK(sel != nullptr && nsel != nullptr && err != nullptr, fn, ": null pointer (sel=", (const void *)sel,
                ", nsel=", (const void *)nsel, ", err=", (const void *)err, ")");
    const size_t need = attention_gathered_wmma_workspace_bytes(s, sel_ld);
    STRIX_CHECK(workspace_bytes >= need, fn, ": workspace ", workspace_bytes, " bytes, need ", need);
    wmma_attention_launch(true, q, gate, kv, s, gathered_max_keys(s, sel_ld), sel, nsel, sel_ld, scale, out, workspace,
                          err, stream, precision);
    STRIX_HIP_CHECK_LAUNCH(fn, kv.q8 ? " kv8" : "", " T=", s.T, " pos0=", s.pos0, " Hq=", s.Hq, " Hk=", s.Hk);
}
}  // namespace

size_t attention_wmma_workspace_bytes(const AttentionShape &s) {
    check_shape(s);
    return wmma_workspace(s, s.pos0 + s.T);
}

void attention_wmma(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
                    const AttentionShape &s, float scale, void *out, float *workspace, size_t workspace_bytes, Act act,
                    hipStream_t stream, AttnPrecision precision) {
    attention_wmma_impl(q, gate, KvArg{k_cache, v_cache, nullptr}, capacity, s, scale, out, workspace, workspace_bytes,
                        act, stream, precision);
}

void attention_wmma(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s, float scale,
                    void *out, float *workspace, size_t workspace_bytes, Act act, hipStream_t stream,
                    AttnPrecision precision) {
    attention_wmma_impl(q, gate, KvArg{nullptr, nullptr, &kv}, kv.capacity, s, scale, out, workspace, workspace_bytes,
                        act, stream, precision);
}

size_t attention_gathered_wmma_workspace_bytes(const AttentionShape &s, int64_t sel_ld) {
    check_shape(s);
    STRIX_CHECK(sel_ld >= 1 && sel_ld <= 65536, "attention_gathered_wmma_workspace_bytes: sel_ld = ", sel_ld,
                ", expected 1..65536");
    return wmma_workspace(s, gathered_max_keys(s, sel_ld));
}

void attention_gathered_wmma(const void *q, const void *gate, const void *k_cache, const void *v_cache,
                             int64_t capacity, const AttentionShape &s, const int32_t *sel, const int32_t *nsel,
                             int64_t sel_ld, float scale, void *out, float *workspace, size_t workspace_bytes,
                             uint32_t *err, Act act, hipStream_t stream, AttnPrecision precision) {
    attention_gathered_wmma_impl(q, gate, KvArg{k_cache, v_cache, nullptr}, capacity, s, sel, nsel, sel_ld, scale, out,
                                 workspace, workspace_bytes, err, act, stream, precision);
}

void attention_gathered_wmma(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s,
                             const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale, void *out,
                             float *workspace, size_t workspace_bytes, uint32_t *err, Act act, hipStream_t stream,
                             AttnPrecision precision) {
    attention_gathered_wmma_impl(q, gate, KvArg{nullptr, nullptr, &kv}, kv.capacity, s, sel, nsel, sel_ld, scale, out,
                                 workspace, workspace_bytes, err, act, stream, precision);
}

}  // namespace strix::kernels
