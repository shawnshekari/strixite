#include "kernels/qsa.hpp"

#include "common/hip_check.hpp"
#include "kernels/rope_math.hpp"
#include "kernels/wmma.hpp"

#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

namespace rm = rope_math;

constexpr int kThreads = 256;
constexpr int64_t kMaxD = 256, kMaxH = 8, kMaxK = 1024;
constexpr int64_t kMaxTokens = 65535;       // grid.y of qsa_scores
constexpr int64_t kMaxPos = 1ll << 24;      // RoPE positions (as kernels/rope)

__device__ inline float load(const float *p, int64_t i) { return p[i]; }
__device__ inline float load(const uint16_t *p, int64_t i) { return rm::bf16_bits_to_f32(p[i]); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = rm::f32_to_bf16_bits(v); }
template <typename T> __device__ inline float round_act(float v);
template <> __device__ inline float round_act<float>(float v) { return v; }
template <> __device__ inline float round_act<uint16_t>(float v) { return rm::rb(v); }

// Workgroup b0 + blockIdx.x (< nblocks): one completed block's key. Workgroup nblocks: the tail copy.
// pos3 (kernels/rope.hpp MropePos, may be null): interleaved-mRoPE section triples for the sequence positions
// the block keys rope at - the block's first key, which for the first completed block can be a tail position.
// base = tail0 = 4*(pos0/4); rows >= (pos0 + T) - base. Null = scalar 4b (text-only, today's values bit for bit).
template <typename T>
__global__ void __launch_bounds__(kThreads)
    block_keys_kernel(const T *raw, int64_t raw_ts, int64_t T_, int64_t pos0, const T *tail_in, T *tail_out, int D,
                      const float *w, float eps, const float *inv_freq, float cos_sin_scale, int rot, T *bk,
                      int64_t cap, int64_t b0, int64_t nblocks, const int64_t *pos3, int64_t p3base) {
    __shared__ float xs[kMaxD];
    __shared__ float red[kThreads];
    const int64_t tail0 = (pos0 / kQsaBlock) * kQsaBlock;  // first position held by tail_in
    auto raw_at = [&](int64_t p, int d) {
        return p >= pos0 ? load(raw, (p - pos0) * raw_ts + d) : load(tail_in, (p - tail0) * D + d);
    };
    if ((int64_t)blockIdx.x == nblocks) {
        const int64_t start = ((pos0 + T_) / kQsaBlock) * kQsaBlock;
        for (int64_t i = threadIdx.x; i < (pos0 + T_ - start) * D; i += blockDim.x) {
            const int64_t p = start + i / D;
            const int d = (int)(i % D);
            store(tail_out, (p - start) * D + d, raw_at(p, d));
        }
        return;
    }
    const int64_t b = b0 + blockIdx.x;
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = 0.f;
        for (int r = 0; r < kQsaBlock; ++r) s += raw_at(b * kQsaBlock + r, d);  // FP32 mean, then the dtype's cast
        const float m = round_act<T>(s / (float)kQsaBlock);
        xs[d] = m;
        ss = fmaf(m, m, ss);
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)D + eps);
    float nv[kMaxD / kThreads + 1];
    int cnt = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) nv[cnt++] = round_act<T>((xs[d] * inv) * (1.0f + w[d]));
    __syncthreads();
    cnt = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) xs[d] = nv[cnt++];
    __syncthreads();
    const int half = rot / 2;
    constexpr bool kBF16 = sizeof(T) == 2;
    const int64_t *tr = pos3 ? pos3 + 3 * (b * kQsaBlock - p3base) : nullptr;  // the block's first key's triple
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float y = xs[d];
        if (d < rot) {
            const int i = d < half ? d : d - half;
            float c, s, y0, y1;
            rm::cos_sin<kBF16>(tr ? tr[i % 3] : b * kQsaBlock, inv_freq[i], cos_sin_scale, c, s);
            rm::rotate<kBF16>(xs[i], xs[i + half], c, s, y0, y1);
            y = d < half ? y0 : y1;
        }
        store(bk, ((int64_t)(d / 8) * cap + b) * 8 + d % 8, y);
    }
}

__device__ inline void load8(const float *p, int64_t i, float (&v)[8]) {
    const float4 a = reinterpret_cast<const float4 *>(p + i)[0], b = reinterpret_cast<const float4 *>(p + i)[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ inline void load8(const uint16_t *p, int64_t i, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p + i);
    const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w4[j] << 16), v[2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
}

// Thread = one block key b, for up to kScQ queries [kScQ y, kScQ y + kScQ) (grid.y); their H heads in LDS (dynamic,
// kScQ x H x D floats). Each key is read once per kScQ queries: an MTP verify (T = 2..8) used to re-read every block
// key per query (grid.y = T) - at 491k context T = 8 cost 2.6x T = 1, which runs at the read peak. Per (query, key)
// the products and their order are as before (bit-identical scores).
constexpr int kScQ = 8;
template <typename T>
__global__ void __launch_bounds__(kThreads)
    scores_kernel(const T *q, int64_t q_ts, int64_t q_hs, int64_t T_rows, int64_t pos0, int H, int D, const T *bk,
                  int64_t cap, float *scores, int64_t ld) {
    extern __shared__ float qs[];  // [kScQ][H][D]
    const int64_t t0 = (int64_t)blockIdx.y * kScQ;
    const int nq = (int)(T_rows - t0 < kScQ ? T_rows - t0 : kScQ);
    for (int i = threadIdx.x; i < nq * H * D; i += blockDim.x) {
        const int tq = i / (H * D), r = i % (H * D);
        qs[i] = load(q, (t0 + tq) * q_ts + (r / D) * q_hs + r % D);
    }
    __syncthreads();
    // The last query sees the most blocks; a key past it is past every query of the group.
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (pos0 + t0 + nq) / kQsaBlock) return;
    float acc[kScQ][kMaxH];
#pragma unroll
    for (int tq = 0; tq < kScQ; ++tq)
#pragma unroll
        for (int h = 0; h < kMaxH; ++h) acc[tq][h] = 0.f;
    for (int c = 0; c < D / 8; ++c) {
        float v[8];
        load8(bk, ((int64_t)c * cap + b) * 8, v);
#pragma unroll
        for (int tq = 0; tq < kScQ; ++tq)
            if (tq < nq)
#pragma unroll
                for (int h = 0; h < kMaxH; ++h)
                    if (h < H)
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[tq][h] = fmaf(qs[(tq * H + h) * D + c * 8 + e], v[e], acc[tq][h]);
    }
#pragma unroll
    for (int tq = 0; tq < kScQ; ++tq) {
        if (tq >= nq || b >= (pos0 + t0 + tq + 1) / kQsaBlock) continue;  // query t0 + tq doesn't see block b
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < kMaxH; ++h)
            if (h < H) s += fmaxf(acc[tq][h], 0.f);
        scores[(t0 + tq) * ld + b] = s / sqrtf((float)D);
    }
}

// Prefill scores on the matrix units (BF16 activations, D % 16 == 0): a workgroup of kSwWaves waves keeps 64 block
// keys in LDS and walks every 64-query tile of the call past them (keys read once); wave w takes queries
// [16w, 16w + 16) of a tile against the 64 keys - per head, D / 16 steps of v_wmma_f32_16x16x16_bf16 (the query rows
// as A, read straight from memory; the keys as B from LDS), then ReLU and the sum over heads in registers, the
// 1 / sqrt(D) scale and the causal visibility (row t keeps its (pos0 + t + 1) / 4 complete blocks) at the store. The
// products are exact in FP32 as in scores_kernel; only the summation order and the matrix units' accumulation differ
// (FP32 noise, wmma.hpp). The scalar kernel above stays for decode / verify (small T) and F32 activations.
constexpr int kSwKeys = 64, kSwQ = 64, kSwWaves = kSwQ / 16, kSwThreads = kSwWaves * 32;
constexpr int64_t kQsaScoresWmmaMinT = 64;  // below: the scalar kernel (decode, MTP verifies, short prefill tails)

__global__ void __launch_bounds__(kSwThreads)
    scores_wmma_kernel(const uint16_t *q, int64_t q_ts, int64_t q_hs, int64_t T, int64_t pos0, int H, int D,
                       const uint16_t *bk, int64_t cap, int64_t nb, float *scores, int64_t ld) {
    __shared__ __attribute__((aligned(16))) uint16_t ks[kSwKeys * (kMaxD + 8)];
    const int kld = D + 8;  // padded row (16-byte multiple)
    const int64_t b0 = (int64_t)blockIdx.x * kSwKeys;
    // The 64 keys' rows: element (b, d) of the block-key cache sits at ((d / 8) * cap + b) * 8 + d % 8.
    for (int i = threadIdx.x; i < kSwKeys * (D / 8); i += kSwThreads) {
        const int j = i % kSwKeys, c = i / kSwKeys;
        const int64_t b = b0 + j;
        uint4 v = {0u, 0u, 0u, 0u};
        if (b < nb) v = *reinterpret_cast<const uint4 *>(bk + ((int64_t)c * cap + b) * 8);
        *reinterpret_cast<uint4 *>(ks + j * kld + c * 8) = v;
    }
    __syncthreads();
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float inv_sqrt_d = 1.0f / sqrtf((float)D);
    // The first query that sees key b0: (pos0 + t + 1) / 4 > b0  <=>  t >= 4 * b0 + 3 - pos0.
    const int64_t t_first = std::max<int64_t>(0, 4 * b0 + 3 - pos0);
    for (int64_t t0 = (t_first / kSwQ) * kSwQ; t0 < T; t0 += kSwQ) {
        const int64_t tq = t0 + wave * 16;  // this wave's 16 queries
        if (tq >= T) continue;              // no barrier below: waves are independent past the key load
        wmma_f32x8 sum[kSwKeys / 16];
#pragma unroll
        for (int n = 0; n < kSwKeys / 16; ++n) sum[n] = wmma_f32x8{};
        const int64_t trow = std::min<int64_t>(tq + lane % 16, T - 1);  // A rows past T: clamped, never stored
        for (int h = 0; h < H; ++h) {
            wmma_f32x8 acc[kSwKeys / 16];
#pragma unroll
            for (int n = 0; n < kSwKeys / 16; ++n) acc[n] = wmma_f32x8{};
            const uint16_t *qa = q + trow * q_ts + (int64_t)h * q_hs;
            for (int k0 = 0; k0 < D; k0 += 16) {
                const uint4 *r = reinterpret_cast<const uint4 *>(qa + k0);
                const uint4 lo = r[0], hi = r[1];
                const wmma_u32x8 av = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                const wmma_bf16x16 a = __builtin_bit_cast(wmma_bf16x16, av);
#pragma unroll
                for (int n = 0; n < kSwKeys / 16; ++n)
                    acc[n] = wmma_bf16(a, wmma_load_rows_a16(ks + n * 16 * kld + k0, kld), acc[n]);
            }
#pragma unroll
            for (int n = 0; n < kSwKeys / 16; ++n)
#pragma unroll
                for (int v = 0; v < 8; ++v) sum[n][v] += fmaxf(acc[n][v], 0.f);
        }
        // Element v of lane l: query tq + wmma_c_row(v), key b0 + 16 n + l % 16.
#pragma unroll
        for (int n = 0; n < kSwKeys / 16; ++n)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t t = tq + wmma_c_row(v), b = b0 + n * 16 + lane % 16;
                if (t < T && b < (pos0 + t + 1) / kQsaBlock) scores[t * ld + b] = sum[n][v] * inv_sqrt_d;
            }
    }
}

// Order-preserving map of FP32 to uint32 (larger float -> larger key). -0 is folded into +0 first: they
// compare equal, so they must tie (a score is a sum of relus, and all-negative heads give exactly 0).
__device__ inline uint32_t order_key(float f) {
    const uint32_t u = f == 0.0f ? 0u : __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Top-k, split across workgroups: each query's row is cut into kTopkChunk-block chunks, one workgroup
// each. Three histogram passes radix-select the k-th largest key, 11 + 11 + 10 bits: every workgroup
// histograms its chunk's keys that match the prefix so far in LDS and adds the nonzero bins into the
// query's global histogram; the last workgroup to arrive (arrival counter, fence) picks the digit that holds
// the remaining-th largest, stores prefix + remaining for the next pass, and re-zeroes the histogram and
// counter - so the workspace is all zeros between calls (it must start that way: qsa_topk_workspace_bytes).
// After the last pass the key is the threshold; if more blocks equal it than still fit, the last workgroup
// also finds the index of the last equal one that fits (index order: ties to the lower block). The select
// pass then takes every block above the threshold, and the equal ones up to that index.
constexpr int64_t kTopkChunk = 1024;  // blocks per workgroup (4 per thread)
constexpr int kBins = 2048;           // 11-bit digits (the last pass uses 1024)
// Per-query workspace, uint32 words.
constexpr int64_t kWsHist = 0, kWsCount = kBins, kWsPrefix = kBins + 1, kWsRemaining = kBins + 2,
                  kWsIdxThr = kBins + 3, kWsOut = kBins + 4, kWsStride = kBins + 8;
constexpr int kPassShift[3] = {21, 10, 0};
constexpr uint32_t kPassMask[3] = {0u, 0xffe00000u, 0xfffffc00u};  // key bits fixed before the pass

__device__ inline uint32_t atomic_load(const uint32_t *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// True in every thread of the workgroup that arrives last; the counter is reset for the next pass.
__device__ inline bool arrive_last(uint32_t *count, uint32_t nwg) {
    __shared__ bool s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        s_last = atomicAdd(count, 1u) == nwg - 1;
        if (s_last) *count = 0u;
    }
    __syncthreads();
    if (s_last) __threadfence();
    return s_last;
}

__global__ void __launch_bounds__(kThreads)
    topk_hist_kernel(const float *scores, int64_t ld, int64_t pos0, int k, int pass, uint32_t *ws) {
    __shared__ uint32_t hist[kBins];
    __shared__ int scan[kThreads];
    __shared__ uint32_t s_digit, s_above;
    __shared__ int s_idx;
    const int64_t t = blockIdx.y, n = (pos0 + t + 1) / kQsaBlock;
    const uint32_t nwg = (uint32_t)((n + kTopkChunk - 1) / kTopkChunk);
    if (n <= k || blockIdx.x >= nwg) return;
    uint32_t *w = ws + t * kWsStride;
    const float *row = scores + t * ld;
    const int shift = kPassShift[pass];
    const uint32_t mask = kPassMask[pass], prefix = pass == 0 ? 0u : w[kWsPrefix],
                   remaining = pass == 0 ? (uint32_t)k : w[kWsRemaining], dmask = pass == 2 ? 1023u : 2047u;
    for (int i = threadIdx.x; i < kBins; i += kThreads) hist[i] = 0u;
    __syncthreads();
    const int64_t i0 = (int64_t)blockIdx.x * kTopkChunk, i1 = n < i0 + kTopkChunk ? n : i0 + kTopkChunk;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) {
        const uint32_t key = order_key(row[i]);
        if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & dmask], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kBins; i += kThreads)
        if (hist[i]) atomicAdd(&w[kWsHist + i], hist[i]);
    if (!arrive_last(&w[kWsCount], nwg)) return;

    // Last workgroup: the whole histogram, re-zeroed behind it.
    for (int i = threadIdx.x; i < kBins; i += kThreads) {
        hist[i] = atomic_load(&w[kWsHist + i]);
        w[kWsHist + i] = 0u;
    }
    __syncthreads();
    // Thread j owns bins [8j, 8j + 8); counts above a thread's bins = a suffix sum over the higher threads.
    constexpr int kPer = kBins / kThreads;
    const int j = threadIdx.x;
    uint32_t own = 0;
    for (int b = 0; b < kPer; ++b) own += hist[j * kPer + b];
    scan[kThreads - 1 - j] = (int)own;  // reversed: an inclusive scan then sums the bins from the top down
    __syncthreads();
    for (int o = 1; o < kThreads; o <<= 1) {
        const int v = (int)threadIdx.x >= o ? scan[threadIdx.x - o] : 0;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    uint32_t above = (uint32_t)scan[kThreads - 1 - j] - own;
    if (above < remaining && above + own >= remaining) {  // exactly one thread (the matching keys >= remaining)
        for (int b = kPer - 1; b >= 0; --b) {
            const uint32_t c = hist[j * kPer + b];
            if (above + c >= remaining) {
                s_digit = (uint32_t)(j * kPer + b), s_above = above;
                break;
            }
            above += c;
        }
    }
    __syncthreads();
    const uint32_t key = prefix | (s_digit << shift), need = remaining - s_above;
    if (pass < 2) {
        if (threadIdx.x == 0) w[kWsPrefix] = key, w[kWsRemaining] = need;
        return;
    }
    // Threshold known. Ties at it: find the need-th equal block in index order (rare - one workgroup).
    const uint32_t equal = hist[s_digit];
    if (threadIdx.x == 0) w[kWsPrefix] = key, w[kWsIdxThr] = 0xffffffffu, s_idx = -1;
    if (equal <= need) return;
    __syncthreads();
    int taken = 0;
    for (int64_t start = 0; start < n && taken < (int)need; start += kThreads) {
        const int64_t i = start + threadIdx.x;
        const int f = (i < n && order_key(row[i]) == key) ? 1 : 0;
        scan[threadIdx.x] = f;
        __syncthreads();
        for (int o = 1; o < kThreads; o <<= 1) {  // inclusive scan (Hillis-Steele)
            const int v = (int)threadIdx.x >= o ? scan[threadIdx.x - o] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        if (f && taken + scan[threadIdx.x] == (int)need) s_idx = (int)i;
        taken += scan[kThreads - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0) w[kWsIdxThr] = (uint32_t)s_idx;
}

__global__ void __launch_bounds__(kThreads)
    topk_select_kernel(const float *scores, int64_t ld, int64_t pos0, int k, int32_t *sel, int32_t *nsel,
                       uint32_t *ws) {
    const int64_t t = blockIdx.y, n = (pos0 + t + 1) / kQsaBlock;
    const uint32_t nwg = (uint32_t)((n + kTopkChunk - 1) / kTopkChunk);
    const int64_t i0 = (int64_t)blockIdx.x * kTopkChunk, i1 = n < i0 + kTopkChunk ? n : i0 + kTopkChunk;
    int32_t *out = sel + t * k;
    if (n <= k) {  // everything
        for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) out[i] = (int32_t)i;
        if (blockIdx.x == 0 && threadIdx.x == 0) nsel[t] = (int32_t)n;
        return;
    }
    if (blockIdx.x >= nwg) return;
    uint32_t *w = ws + t * kWsStride;
    const float *row = scores + t * ld;
    const uint32_t thr = w[kWsPrefix], idx_thr = w[kWsIdxThr];
    for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) {
        const uint32_t key = order_key(row[i]);
        if (key > thr || (key == thr && (uint32_t)i <= idx_thr)) {
            const uint32_t at = atomicAdd(&w[kWsOut], 1u);
            if (at < (uint32_t)k) out[at] = (int32_t)i;  // more than k would be a bug: nsel reports it
        }
    }
    if (!arrive_last(&w[kWsCount], nwg)) return;
    if (threadIdx.x == 0) {
        nsel[t] = (int32_t)atomic_load(&w[kWsOut]);  // == k unless the selection is broken
        w[kWsOut] = 0u;
    }
}

// One workgroup per query row with more than k blocks: its k selected block ids (written in atomic order by
// topk_select_kernel) sorted ascending in LDS (bitonic, padded to a power of two with INT_MAX). Gathered
// attention accumulates over the blocks in this order, so a fixed order makes its result reproducible.
__global__ void __launch_bounds__(kThreads) topk_sort_kernel(int32_t *sel, int64_t pos0, int k, int p2) {
    __shared__ int32_t v[kMaxK];
    const int64_t t = blockIdx.x, n = (pos0 + t + 1) / kQsaBlock;
    if (n <= k) return;  // selected 0..n-1 in order already
    int32_t *row = sel + t * k;
    for (int i = threadIdx.x; i < p2; i += kThreads) v[i] = i < k ? row[i] : INT_MAX;
    __syncthreads();
    for (int size = 2; size <= p2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < p2; i += kThreads) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    const int32_t a = v[i], b = v[j];
                    if ((a > b) == up) v[i] = b, v[j] = a;
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < k; i += kThreads) row[i] = v[i];
}

// Row top-k: one workgroup of kRowThreads per query, one launch. The same radix select as the split kernels
// (three passes, 11 + 11 + 10 bits, over the whole row - it stays in L2), then one select pass in index order:
// a block is taken when its key is above the threshold, or equal to it and among the first `need` equal ones
// (ties to the lower block index). Each thread selects from its own slice of the row, placed by one workgroup
// prefix sum of the slices' counts, so the output is ascending without a sort.
constexpr int kRowThreads = 1024;
constexpr int kRowEqBits = 21;  // the select's packed counts: equal-key blocks in the low bits, the rest above
constexpr uint32_t kRowEqMask = (1u << kRowEqBits) - 1u;
constexpr int kRowWaves = kRowThreads / 32;  // warpSize >= 32

// Exclusive sum of v over the threads before this one (thread order) and the workgroup's total; wave shuffles, then
// the wave totals. Every thread calls. Sums must not overflow 32 bits (the caller packs two 16-bit counts).
__device__ inline uint32_t row_scan_u32(uint32_t v, int *s_wave, uint32_t &total) {
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5, nw = kRowThreads >> 5;
    uint32_t inc = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const uint32_t o = (uint32_t)__shfl_up((int)inc, off, 32);
        if (lane >= off) inc += o;
    }
    if (lane == 31) s_wave[wave] = (int)inc;
    __syncthreads();
    uint32_t wbefore = 0, tot = 0;
    for (int w = 0; w < nw; ++w) {
        const uint32_t c = (uint32_t)s_wave[w];
        wbefore += w < wave ? c : 0u;
        tot += c;
    }
    __syncthreads();  // s_wave is reused
    total = tot;
    return wbefore + inc - v;
}

// The row algorithm over any source of n keys in index order: key(i) its order key, id(i) the block index written out
// (ascending in i). Shared by the row path (the scores row itself) and the filter path's exact select (its candidates,
// sorted by block index, in LDS). Three radix passes (11 + 11 + 10 bits) find the k-th largest key; the select then
// takes keys above it and the first `remaining` equal ones in index order - ties to the lower block. Each thread
// selects from its own slice, placed by one exclusive scan of the slices' packed counts: ascending without a sort.
// n > k; n < kRowEqMask (the packed counts). Every thread of the kRowThreads workgroup calls it.
template <typename Src>
__device__ inline void row_select(const Src &src, int64_t n, int k, int32_t *out, int32_t *nsel_t, uint32_t *hist,
                                  int *scan, int *s_wave, uint32_t &s_digit, uint32_t &s_above) {
    uint32_t prefix = 0u, remaining = (uint32_t)k;
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = kPassShift[pass];
        const uint32_t mask = kPassMask[pass], dmask = pass == 2 ? 1023u : 2047u;
        for (int i = threadIdx.x; i < kBins; i += kRowThreads) hist[i] = 0u;
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += kRowThreads) {
            const uint32_t key = src.key(i);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & dmask], 1u);
        }
        __syncthreads();
        // Thread j owns bins [kPer j, kPer j + kPer); counts above them = a suffix sum over the higher threads.
        constexpr int kPer = kBins / kRowThreads;
        const int j = threadIdx.x;
        uint32_t own = 0;
        for (int b = 0; b < kPer; ++b) own += hist[j * kPer + b];
        scan[kRowThreads - 1 - j] = (int)own;  // reversed: an inclusive scan then sums the bins from the top down
        __syncthreads();
        for (int o = 1; o < kRowThreads; o <<= 1) {
            const int v = j >= o ? scan[j - o] : 0;
            __syncthreads();
            scan[j] += v;
            __syncthreads();
        }
        uint32_t above = (uint32_t)scan[kRowThreads - 1 - j] - own;
        if (above < remaining && above + own >= remaining) {  // exactly one thread
            for (int b = kPer - 1; b >= 0; --b) {
                const uint32_t c = hist[j * kPer + b];
                if (above + c >= remaining) {
                    s_digit = (uint32_t)(j * kPer + b), s_above = above;
                    break;
                }
                above += c;
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        remaining -= s_above;
        __syncthreads();  // s_digit / s_above / hist are rewritten by the next pass
    }
    // prefix = the k-th largest key; remaining = how many keys equal to it fit.
    // The select: thread j owns keys [j * per, (j + 1) * per) in index order. It counts its keys above / equal to
    // the threshold (no barriers), one workgroup exclusive scan of the packed counts gives what comes before its
    // slice, then it writes its picks from there - ascending, ties to the lower block index. Replaced stepping
    // through the row 1024 blocks at a time with two flag scans (four barriers) a step: ~17 of 26.6 us at 12288
    // blocks (T = 1).
    const int64_t per = (n + kRowThreads - 1) / kRowThreads;
    const int64_t b0 = std::min<int64_t>(n, (int64_t)threadIdx.x * per), b1 = std::min<int64_t>(n, b0 + per);
    uint32_t gt = 0, eq = 0;
    for (int64_t i = b0; i < b1; ++i) {
        const uint32_t key = src.key(i);
        gt += key > prefix ? 1u : 0u;
        eq += key == prefix ? 1u : 0u;
    }
    // One packed scan: above the threshold there are fewer than k <= kMaxK keys (11 bits), equal ones at most n
    // (< 2^21, checked at launch).
    uint32_t total;
    const uint32_t before = row_scan_u32((gt << kRowEqBits) | eq, s_wave, total);
    const uint32_t gt_before = before >> kRowEqBits, eq_before = before & kRowEqMask;
    int64_t at = (int64_t)gt_before + std::min<uint32_t>(eq_before, remaining);
    uint32_t eq_rank = eq_before;
    for (int64_t i = b0; i < b1; ++i) {
        const uint32_t key = src.key(i);
        const bool take = key > prefix || (key == prefix && eq_rank++ < remaining);
        if (take && at < k) out[at] = src.id(i);  // more than k would be a bug: nsel reports it
        at += take ? 1 : 0;
    }
    if (threadIdx.x == 0)  // == k unless the selection is broken
        *nsel_t = (int32_t)((total >> kRowEqBits) + std::min<uint32_t>(total & kRowEqMask, remaining));
}

// A scores row as row_select's source.
struct RowSrc {
    const float *row;
    __device__ uint32_t key(int64_t i) const { return order_key(row[i]); }
    __device__ int32_t id(int64_t i) const { return (int32_t)i; }
};

__global__ void __launch_bounds__(kRowThreads)
    topk_row_kernel(const float *scores, int64_t ld, int64_t pos0, int k, int32_t *sel, int32_t *nsel) {
    __shared__ uint32_t hist[kBins];
    __shared__ int scan[kRowThreads];
    __shared__ int s_wave[kRowWaves];
    __shared__ uint32_t s_digit, s_above;
    const int64_t t = blockIdx.x, n = (pos0 + t + 1) / kQsaBlock;
    int32_t *out = sel + t * k;
    if (n <= k) {  // everything, in order
        for (int64_t i = threadIdx.x; i < n; i += kRowThreads) out[i] = (int32_t)i;
        if (threadIdx.x == 0) nsel[t] = (int32_t)n;
        return;
    }
    row_select(RowSrc{scores + t * ld}, n, k, out, nsel + t, hist, scan, s_wave, s_digit, s_above);
}

// Filter top-k (long rows of many queries - deep prefill chunks): the split path reads a row four times (three
// histogram passes and the select: 3.7 GB a layer per pass at 460k context, 8192 queries - 1,345 of a deep chunk's
// 7,437 ms). Here the row is read about once:
//  1. bound: contiguous samples (kFilterSeg blocks every kFilterStride: ~1/16 of the row), one workgroup per query,
//     to (at most) the sample ranked ~kFilterAim * k * samples / n - so about kFilterAim * k blocks of the whole row
//     are expected at or above it;
//  2. filter: each workgroup scans a chunk of its row and appends the (key, block) pairs at or above the bound to the
//     query's candidate list (one atomic a workgroup), counting them all;
//  3. exact: one workgroup per query sorts its candidates by block index in LDS and runs row_select on them - the
//     same result as on the whole row, because every key of the top k is at or above the bound when at least k
//     candidates are (the k-th largest key is then >= the bound). A row with fewer than k candidates (the bound
//     landed too high) or more than the list holds runs row_select on its whole row instead (fallback; counted).
constexpr int64_t kFilterSeg = 64, kFilterStride = 1024;  // sampled blocks, and every how many
// Candidates a query holds (the exact kernel's LDS, a power of two for its sort) and how many are aimed for, in
// units of k: 2.5 k = 1280 keeps both misses (fewer than k, more than the cap) past ~4 sigma of the sample's rank
// noise - at 1.5 k / 2048 a row (8192 queries, 131k) fell back ~6% of the time (counts 502-928).
constexpr int64_t kFilterCap = 4096;
constexpr double kFilterAim = 2.5;
// Per-query filter words, at the workspace's end: bound, count, then kFilterCap (key, block) pairs (the bound kernel
// uses the pairs' space as scratch before the filter fills it).
constexpr int64_t kFwBound = 0, kFwCount = 1, kFwCand = 2, kFwStride = 2 + 2 * kFilterCap;

// Sampled block j (of the row's samples) -> its block index.
__device__ inline int64_t filter_sample_block(int64_t j) { return (j / kFilterSeg) * kFilterStride + j % kFilterSeg; }
__device__ inline int64_t filter_samples(int64_t n) {
    return (n / kFilterStride) * kFilterSeg + std::min<int64_t>(n % kFilterStride, kFilterSeg);
}

// One workgroup of kThreads per query: the bound, a key at most the sampled key of rank r, and the count zeroed for the
// filter. Each thread keeps the best kBoundPer of its samples in registers; the kThreads x kBoundPer survivors go to
// LDS and the bound is the survivor of rank r - never above the true rank-r sample (the survivors are a subset of
// the samples), so at worst a few more candidates. Radix passes over the samples serialized on LDS atomics (a row's
// keys fall in a handful of bins): 1024 threads x 3 passes 40 ms, 256 x 2 passes 35 ms at 8192 queries x 117,760
// blocks (rocprofv3, 471k); this way 17.6 ms - nearly all the scattered sample reads (one 4 KB page a
// 64-block segment: the same reads from contiguous memory took 5.5 ms). Contiguous segments would be cheaper
// but RoPE gives the scores structure along the row: 1024 blocks every 16384 made the counts swing 206-2970.
constexpr int kBoundPer = 4;
constexpr int kBoundSurvivors = kThreads * kBoundPer;
__global__ void __launch_bounds__(kThreads) topk_filter_bound_kernel(const float *scores, int64_t ld, int64_t pos0, int k,
                                                                     uint32_t *fw) {
    __shared__ uint32_t surv[kBoundSurvivors];
    const int64_t t = blockIdx.x, n = (pos0 + t + 1) / kQsaBlock;
    uint32_t *w = fw + t * kFwStride;
    if (threadIdx.x == 0) w[kFwCount] = 0u;
    if (n <= k) return;  // the exact kernel takes everything
    const int64_t ns = filter_samples(n);
    // The rank of the bound among the samples: the aimed share of the row, at least 1.
    const int64_t r = std::max<int64_t>(1, (int64_t)(kFilterAim * (double)k * (double)ns / (double)n) + 1);
    if (r >= ns || r > kBoundSurvivors) {  // can't aim: a bound of 0 keeps every block (the exact kernel falls back)
        if (threadIdx.x == 0) w[kFwBound] = 0u;
        return;
    }
    const float *row = scores + t * ld;
    uint32_t best[kBoundPer];  // descending; 0 = none (no real key is 0: a key's top bit is set for non-negative scores,
                               // and a negative score's key is ~bits, nonzero)
#pragma unroll
    for (int b = 0; b < kBoundPer; ++b) best[b] = 0u;
    for (int64_t j = threadIdx.x; j < ns; j += kThreads) {
        uint32_t key = order_key(row[filter_sample_block(j)]);
#pragma unroll
        for (int b = 0; b < kBoundPer; ++b)  // insertion: carry the smaller one down
            if (key > best[b]) {
                const uint32_t x = best[b];
                best[b] = key, key = x;
            }
    }
#pragma unroll
    for (int b = 0; b < kBoundPer; ++b) surv[threadIdx.x * kBoundPer + b] = best[b];
    if (threadIdx.x == 0) w[kFwBound] = 0u;  // stays 0 if fewer than r real survivors: everything (fallback past the cap)
    __syncthreads();
    // The rank-r survivor by counting (as kernels/logits_topk places its candidates): a survivor with fewer than r
    // greater and at least r greater-or-equal is it - one barrier and broadcast LDS reads instead of a sort's 55
    // barrier stages (17.5 ms at 8192 x 117,760, rocprofv3).
#pragma unroll
    for (int b = 0; b < kBoundPer; ++b) {
        const uint32_t key = best[b];
        if (key == 0u) continue;
        int gt = 0, ge = 0;
        for (int c = 0; c < kBoundSurvivors; ++c) {
            const uint32_t o = surv[c];
            gt += o > key ? 1 : 0;
            ge += o >= key ? 1 : 0;
        }
        if (gt < r && ge >= r) w[kFwBound] = key;  // every thread that finds it writes the same key
    }
}

// Grid (chunks, T): a workgroup's chunk of kTopkChunk blocks; its picks compacted in LDS, then one atomic reserves
// their place in the query's list (entries past the cap are counted, not stored).
__global__ void __launch_bounds__(kThreads)
    topk_filter_kernel(const float *scores, int64_t ld, int64_t pos0, int k, uint32_t *fw, int64_t cap) {
    __shared__ uint32_t s_key[kTopkChunk];
    __shared__ int32_t s_id[kTopkChunk];
    __shared__ uint32_t s_n, s_at;
    const int64_t t = blockIdx.y, n = (pos0 + t + 1) / kQsaBlock;
    const int64_t i0 = (int64_t)blockIdx.x * kTopkChunk;
    if (n <= k || i0 >= n) return;
    uint32_t *w = fw + t * kFwStride;
    const uint32_t bound = w[kFwBound];
    if (threadIdx.x == 0) s_n = 0u;
    __syncthreads();
    const int64_t i1 = std::min<int64_t>(n, i0 + kTopkChunk);
    const float *row = scores + t * ld;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) {
        const uint32_t key = order_key(row[i]);
        if (key >= bound) {
            const uint32_t at = atomicAdd(&s_n, 1u);
            s_key[at] = key, s_id[at] = (int32_t)i;
        }
    }
    __syncthreads();
    const uint32_t m = s_n;
    if (m == 0) return;
    if (threadIdx.x == 0) s_at = atomicAdd(&w[kFwCount], m);
    __syncthreads();
    const uint32_t at = s_at;
    for (uint32_t c = threadIdx.x; c < m; c += kThreads)
        if ((int64_t)(at + c) < cap) {
            w[kFwCand + 2 * (at + c)] = s_key[c];
            w[kFwCand + 2 * (at + c) + 1] = (uint32_t)s_id[c];
        }
}

struct CandSrc {
    const uint32_t *key_, *id_;
    __device__ uint32_t key(int64_t i) const { return key_[i]; }
    __device__ int32_t id(int64_t i) const { return (int32_t)id_[i]; }
};

// One workgroup per query: the exact top k from the candidates (sorted by block index first), or from the whole row
// when the candidates can't hold it (fallback, counted in fallbacks[0]).
__global__ void __launch_bounds__(kRowThreads)
    topk_filter_exact_kernel(const float *scores, int64_t ld, int64_t pos0, int k, int32_t *sel, int32_t *nsel,
                             const uint32_t *fw, int64_t cap, uint32_t *fallbacks) {
    __shared__ uint32_t hist[kBins];
    __shared__ int scan[kRowThreads];
    __shared__ int s_wave[kRowWaves];
    __shared__ uint32_t s_digit, s_above;
    __shared__ uint32_t c_key[kFilterCap], c_id[kFilterCap];
    const int64_t t = blockIdx.x, n = (pos0 + t + 1) / kQsaBlock;
    int32_t *out = sel + t * k;
    if (n <= k) {  // everything, in order
        for (int64_t i = threadIdx.x; i < n; i += kRowThreads) out[i] = (int32_t)i;
        if (threadIdx.x == 0) nsel[t] = (int32_t)n;
        return;
    }
    const uint32_t *w = fw + t * kFwStride;
    const int64_t m = (int64_t)w[kFwCount];
    if (m < k || m > cap) {  // the bound missed: the whole row (correct, slower)
        if (threadIdx.x == 0) atomicAdd(fallbacks, 1u);
        row_select(RowSrc{scores + t * ld}, n, k, out, nsel + t, hist, scan, s_wave, s_digit, s_above);
        return;
    }
    // Candidates into LDS, padded to a power of two with block index INT_MAX, then a bitonic sort by block index.
    int64_t p2 = 1;
    while (p2 < m) p2 <<= 1;
    for (int64_t i = threadIdx.x; i < p2; i += kRowThreads) {
        c_key[i] = i < m ? w[kFwCand + 2 * i] : 0u;
        c_id[i] = i < m ? w[kFwCand + 2 * i + 1] : 0xffffffffu;
    }
    __syncthreads();
    for (int64_t size = 2; size <= p2; size <<= 1)
        for (int64_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (int64_t i = threadIdx.x; i < p2; i += kRowThreads) {
                const int64_t j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    const uint32_t a = c_id[i], b = c_id[j];
                    if ((a > b) == up) {
                        c_id[i] = b, c_id[j] = a;
                        const uint32_t ka = c_key[i];
                        c_key[i] = c_key[j], c_key[j] = ka;
                    }
                }
            }
            __syncthreads();
        }
    row_select(CandSrc{c_key, c_id}, m, k, out, nsel + t, hist, scan, s_wave, s_digit, s_above);
}

// Cooperative top-k: the split path's parallelism in one launch. Grid (W, T) of kThreads, all resident
// (hipLaunchCooperativeKernel); workgroup w of query t owns a contiguous chunk of its row. Each of the three radix
// passes: chunk histogram in LDS, nonzero bins added into the row's global histogram for that pass, grid sync, then
// every workgroup reads the whole histogram and picks the same digit (no hand-off). Select: each workgroup counts its
// chunk's keys above / equal to the threshold, grid sync, an exclusive sum over the lower workgroups gives where its
// chunk's selection starts and how many equal keys came before - so the output is in index order (no sort) with
// ties to the lower block. After that sync every histogram has been read: each workgroup zeroes its share of them,
// so the workspace is zeros again for the next call. Rows with n <= k take everything (they still join the syncs).
constexpr int kCoopMaxWgs = 64;  // workgroups per query at most
// Per-query coop workspace, uint32 words: three pass histograms, then (above, equal) per workgroup.
constexpr int64_t kCoopHist = 0, kCoopCounts = 3 * kBins, kCoopStride = 3 * kBins + 2 * kCoopMaxWgs,
                  kCoopWords = kQsaTopkRowMaxT * kCoopStride;  // at the workspace's start, before the split rows

__device__ inline int wg_scan_flag(bool f, int *s_wave, int &total) {  // as row_scan_flag, for kThreads
    const int lane = (int)threadIdx.x % warpSize, wave = (int)threadIdx.x / warpSize, nw = kThreads / warpSize;
    const unsigned long long b = __ballot(f);
    const int in_wave = __popcll(b & ((1ull << lane) - 1ull));
    if (lane == 0) s_wave[wave] = __popcll(b);
    __syncthreads();
    int before = 0, tot = 0;
    for (int w = 0; w < nw; ++w) {
        const int c = s_wave[w];
        before += w < wave ? c : 0;
        tot += c;
    }
    __syncthreads();
    total = tot;
    return before + in_wave;
}

__global__ void __launch_bounds__(kThreads)
    topk_coop_kernel(const float *scores, int64_t ld, int64_t pos0, int k, int32_t *sel, int32_t *nsel, uint32_t *ws) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ uint32_t hist[kBins];
    __shared__ int scan[kThreads];
    __shared__ int s_wave[kThreads / 32];
    __shared__ uint32_t s_digit, s_above;
    const int64_t t = blockIdx.y, n = (pos0 + t + 1) / kQsaBlock;
    const int W = (int)gridDim.x, w = (int)blockIdx.x;
    const int64_t chunk = (n + W - 1) / W, i0 = std::min<int64_t>(n, (int64_t)w * chunk), i1 = std::min<int64_t>(n, i0 + chunk);
    const bool all = n <= k;
    int32_t *out = sel + t * k;
    uint32_t *cw = ws + t * kCoopStride;
    const float *row = scores + t * ld;
    uint32_t prefix = 0u, remaining = (uint32_t)k;
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = kPassShift[pass];
        const uint32_t mask = kPassMask[pass], dmask = pass == 2 ? 1023u : 2047u;
        uint32_t *gh = cw + kCoopHist + pass * kBins;
        if (!all) {
            for (int i = threadIdx.x; i < kBins; i += kThreads) hist[i] = 0u;
            __syncthreads();
            for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) {
                const uint32_t key = order_key(row[i]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & dmask], 1u);
            }
            __syncthreads();
            for (int i = threadIdx.x; i < kBins; i += kThreads)
                if (hist[i]) atomicAdd(&gh[i], hist[i]);
        }
        grid.sync();
        if (all) continue;
        for (int i = threadIdx.x; i < kBins; i += kThreads) hist[i] = atomic_load(&gh[i]);
        __syncthreads();
        constexpr int kPer = kBins / kThreads;
        const int j = threadIdx.x;
        uint32_t own = 0;
        for (int b = 0; b < kPer; ++b) own += hist[j * kPer + b];
        scan[kThreads - 1 - j] = (int)own;
        __syncthreads();
        for (int o = 1; o < kThreads; o <<= 1) {
            const int v = j >= o ? scan[j - o] : 0;
            __syncthreads();
            scan[j] += v;
            __syncthreads();
        }
        uint32_t above = (uint32_t)scan[kThreads - 1 - j] - own;
        if (above < remaining && above + own >= remaining) {
            for (int b = kPer - 1; b >= 0; --b) {
                const uint32_t c = hist[j * kPer + b];
                if (above + c >= remaining) {
                    s_digit = (uint32_t)(j * kPer + b), s_above = above;
                    break;
                }
                above += c;
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        remaining -= s_above;
        __syncthreads();
    }
    // Counts per workgroup chunk: above the threshold, equal to it.
    if (!all) {
        int gt = 0, eq = 0;
        for (int64_t start = i0; start < i1; start += kThreads) {
            const int64_t i = start + threadIdx.x;
            const uint32_t key = i < i1 ? order_key(row[i]) : 0u;
            int tg, te;
            wg_scan_flag(i < i1 && key > prefix, s_wave, tg);
            wg_scan_flag(i < i1 && key == prefix, s_wave, te);
            gt += tg, eq += te;
        }
        if (threadIdx.x == 0) cw[kCoopCounts + 2 * w] = (uint32_t)gt, cw[kCoopCounts + 2 * w + 1] = (uint32_t)eq;
    }
    grid.sync();
    // Every histogram has been read: zero this workgroup's share for the next call.
    for (int64_t i = (int64_t)w * kThreads + threadIdx.x; i < 3 * kBins; i += (int64_t)W * kThreads) cw[kCoopHist + i] = 0u;
    if (all) {
        for (int64_t i = i0 + threadIdx.x; i < i1; i += kThreads) out[i] = (int32_t)i;
        if (w == 0 && threadIdx.x == 0) nsel[t] = (int32_t)n;
        return;
    }
    int gt_before = 0, eq_before = 0, gt_all = 0, eq_all = 0;
    for (int v = 0; v < W; ++v) {
        const int g = (int)atomic_load(&cw[kCoopCounts + 2 * v]), e = (int)atomic_load(&cw[kCoopCounts + 2 * v + 1]);
        if (v < w) gt_before += g, eq_before += e;
        gt_all += g, eq_all += e;
    }
    int taken = gt_before + std::min(eq_before, (int)remaining), eq_seen = eq_before;
    for (int64_t start = i0; start < i1; start += kThreads) {
        const int64_t i = start + threadIdx.x;
        const uint32_t key = i < i1 ? order_key(row[i]) : 0u;
        const bool eq = i < i1 && key == prefix;
        int eq_total;
        const int eq_rank = wg_scan_flag(eq, s_wave, eq_total);
        const bool take = (i < i1 && key > prefix) || (eq && eq_seen + eq_rank < (int)remaining);
        int total;
        const int at = wg_scan_flag(take, s_wave, total);
        if (take && taken + at < k) out[taken + at] = (int32_t)i;  // more than k would be a bug: nsel reports it
        taken += total, eq_seen += eq_total;
    }
    if (w == W - 1 && threadIdx.x == 0) nsel[t] = (int32_t)(gt_all + std::min(eq_all, (int)remaining));
}

// Workgroups the cooperative kernel may have resident at once on this device (queried once).
int64_t coop_max_resident() {
    static const int64_t r = [] {
        int dev = 0, per = 0;
        hipDeviceProp_t prop{};
        STRIX_HIP_CHECK(hipGetDevice(&dev), "qsa_topk coop: hipGetDevice");
        STRIX_HIP_CHECK(hipGetDeviceProperties(&prop, dev), "qsa_topk coop: hipGetDeviceProperties");
        STRIX_CHECK(prop.cooperativeLaunch, "qsa_topk coop: device ", dev, " has no cooperative launch");
        STRIX_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, reinterpret_cast<const void *>(topk_coop_kernel),
                                                                     kThreads, 0),
                        "qsa_topk coop: occupancy");
        STRIX_CHECK(per >= 1 && prop.multiProcessorCount >= 1, "qsa_topk coop: occupancy ", per, " x ",
                    prop.multiProcessorCount, " multiprocessors");
        return (int64_t)per * prop.multiProcessorCount;  // HIP counts WGPs on gfx1151: per WGP x WGPs, same units
    }();
    return r;
}

size_t act_bytes(const char *fn, Act act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    return act == Act::F32 ? 4 : 2;
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + b_bytes && y < x + a_bytes;
}

}  // namespace

void qsa_block_keys(const void *raw, int64_t raw_ts, int64_t T, int64_t pos0, const void *tail_in, void *tail_out,
                    int64_t D, const float *k_norm, float eps, const float *inv_freq, int64_t rot_dim,
                    void *block_keys, int64_t cap_blocks, Act act, hipStream_t stream, float cos_sin_scale,
                    const MropePos &pos3) {
    const char *fn = "qsa_block_keys";
    const size_t es = act_bytes(fn, act);
    STRIX_CHECK(raw && k_norm && inv_freq && block_keys && tail_out && (tail_in || pos0 % kQsaBlock == 0), fn,
                ": null pointer (raw=", raw, ", k_norm=", (const void *)k_norm, ", inv_freq=", (const void *)inv_freq,
                ", block_keys=", block_keys, ", tail_in=", tail_in, ", tail_out=", tail_out, ")");
    STRIX_CHECK(T >= 1 && pos0 >= 0 && pos0 + T <= kMaxPos, fn, ": T = ", T, ", pos0 = ", pos0, " (positions up to ",
                kMaxPos, ")");
    STRIX_CHECK(D >= 8 && D <= kMaxD && D % 8 == 0, fn, ": D = ", D, ", expected a multiple of 8 up to ", kMaxD);
    STRIX_CHECK(raw_ts >= D, fn, ": raw row stride ", raw_ts, " < D = ", D);
    STRIX_CHECK(rot_dim >= 2 && rot_dim % 2 == 0 && rot_dim <= D, fn, ": rot_dim = ", rot_dim, ", expected even, 2..", D);
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps must be finite and > 0, got ", eps);
    STRIX_CHECK(std::isfinite(cos_sin_scale) && cos_sin_scale >= 1.0f && cos_sin_scale <= 2.0f, fn,
                ": cos_sin_scale = ", cos_sin_scale, ", expected finite, 1..2 (YaRN's attention factor)");
    const int64_t b0 = pos0 / kQsaBlock, b1 = (pos0 + T) / kQsaBlock, nblocks = std::max<int64_t>(0, b1 - b0);
    STRIX_CHECK(b1 <= cap_blocks, fn, ": block ", b1 - 1, " beyond the cache's ", cap_blocks, " blocks");
    STRIX_CHECK(tail_in == nullptr || !overlaps(tail_in, (size_t)(3 * D) * es, tail_out, (size_t)(3 * D) * es), fn,
                ": tail_out overlaps tail_in (use two buffers: other workgroups may still read tail_in)");
    if (pos3.pos3 != nullptr) {
        const int64_t tail0 = b0 * kQsaBlock;  // the earliest position a completed block's key ropes at
        STRIX_CHECK(pos3.base == tail0 && pos3.rows >= pos0 + T - tail0 &&
                        reinterpret_cast<uintptr_t>(pos3.pos3) % 8 == 0,
                    fn, ": mRoPE positions cover rows [", pos3.base, ", ", pos3.base + pos3.rows,
                    "), expected to start at the block start ", tail0, " and reach pos0+T = ", pos0 + T);
    }
    const unsigned grid = (unsigned)(nblocks + 1);
    if (act == Act::F32)
        block_keys_kernel<float><<<grid, kThreads, 0, stream>>>(
            static_cast<const float *>(raw), raw_ts, T, pos0, static_cast<const float *>(tail_in),
            static_cast<float *>(tail_out), (int)D, k_norm, eps, inv_freq, cos_sin_scale, (int)rot_dim, static_cast<float *>(block_keys),
            cap_blocks, b0, nblocks, pos3.pos3, pos3.base);
    else
        block_keys_kernel<uint16_t><<<grid, kThreads, 0, stream>>>(
            static_cast<const uint16_t *>(raw), raw_ts, T, pos0, static_cast<const uint16_t *>(tail_in),
            static_cast<uint16_t *>(tail_out), (int)D, k_norm, eps, inv_freq, cos_sin_scale, (int)rot_dim,
            static_cast<uint16_t *>(block_keys), cap_blocks, b0, nblocks, pos3.pos3, pos3.base);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", T, " pos0=", pos0, " blocks=", nblocks);
}

void qsa_scores(const void *q, int64_t q_ts, int64_t q_hs, int64_t T, int64_t pos0, int64_t H, int64_t D,
                const void *block_keys, int64_t cap_blocks, float *scores, int64_t ld, Act act, hipStream_t stream,
                ScoresPath path) {
    const char *fn = "qsa_scores";
    act_bytes(fn, act);
    STRIX_CHECK(q && block_keys && scores, fn, ": null pointer (q=", q, ", block_keys=", block_keys, ", scores=",
                (void *)scores, ")");
    STRIX_CHECK(T >= 1 && T <= kMaxTokens && pos0 >= 0, fn, ": T = ", T, " (1..", kMaxTokens, "), pos0 = ", pos0);
    STRIX_CHECK(H >= 1 && H <= kMaxH, fn, ": H = ", H, " heads, expected 1..", kMaxH);
    STRIX_CHECK(D >= 8 && D <= kMaxD && D % 8 == 0, fn, ": D = ", D, ", expected a multiple of 8 up to ", kMaxD);
    const int64_t nb = (pos0 + T) / kQsaBlock;
    STRIX_CHECK(nb <= cap_blocks, fn, ": ", nb, " blocks beyond the cache's ", cap_blocks);
    STRIX_CHECK(ld >= std::max<int64_t>(nb, 1), fn, ": scores row stride ld = ", ld, " < ", nb, " blocks");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(block_keys) % 16 == 0, fn, ": block_keys must be 16-byte aligned");
    if (nb == 0) return;  // no complete block yet: nothing to score
    STRIX_CHECK(path == ScoresPath::Auto || path == ScoresPath::Scalar || path == ScoresPath::Wmma, fn, ": path ",
                (int)path);
    const bool wmma_ok = act == Act::BF16 && D % 16 == 0 && q_ts % 8 == 0 && q_hs % 8 == 0 &&
                         reinterpret_cast<uintptr_t>(q) % 16 == 0;
    if (path == ScoresPath::Wmma || (path == ScoresPath::Auto && wmma_ok && T >= kQsaScoresWmmaMinT)) {
        STRIX_CHECK(wmma_ok, fn, ": the matrix-unit path needs BF16 activations, D % 16 == 0 and 16-byte aligned query "
                    "rows (act ", act_name(act), ", D = ", D, ", q_ts = ", q_ts, ", q_hs = ", q_hs, ", q = ", q, ")");
        // The scale before the relu sum is applied after it, as in scores_kernel: s / sqrt(D).
        scores_wmma_kernel<<<(unsigned)((nb + kSwKeys - 1) / kSwKeys), kSwThreads, 0, stream>>>(
            static_cast<const uint16_t *>(q), q_ts, q_hs, T, pos0, (int)H, (int)D,
            static_cast<const uint16_t *>(block_keys), cap_blocks, nb, scores, ld);
        STRIX_HIP_CHECK_LAUNCH(fn, " wmma T=", T, " pos0=", pos0, " blocks=", nb);
        return;
    }
    dim3 grid((unsigned)((nb + kThreads - 1) / kThreads), (unsigned)((T + kScQ - 1) / kScQ)), block(kThreads);
    const size_t lds = (size_t)kScQ * (size_t)H * (size_t)D * sizeof(float);
    STRIX_CHECK(lds <= 64 * 1024, fn, ": ", kScQ, " queries x ", H, " heads x ", D, " dims = ", lds,
                " bytes of LDS, over 64 KiB");
    if (act == Act::F32)
        scores_kernel<float><<<grid, block, lds, stream>>>(static_cast<const float *>(q), q_ts, q_hs, T, pos0, (int)H,
                                                           (int)D, static_cast<const float *>(block_keys), cap_blocks,
                                                           scores, ld);
    else
        scores_kernel<uint16_t><<<grid, block, lds, stream>>>(static_cast<const uint16_t *>(q), q_ts, q_hs, T, pos0,
                                                              (int)H, (int)D, static_cast<const uint16_t *>(block_keys),
                                                              cap_blocks, scores, ld);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", T, " pos0=", pos0, " blocks=", nb);
}

// The filter path's words for T queries: per query kFwStride (bound, count, candidates), then one fallback counter
// (the workspace's last word).
static int64_t filter_words(int64_t T) { return T * kFwStride + 1; }

size_t qsa_topk_workspace_bytes(int64_t T) {
    STRIX_CHECK(T >= 1 && T <= kMaxTokens, "qsa_topk_workspace_bytes: T = ", T, ", expected 1..", kMaxTokens);
    // The coop path's words for kQsaTopkRowMaxT queries first, then the split path's per query - fixed offsets, so a
    // call with one T never meets words another T's call left behind (the coop words once sat at T * kWsStride,
    // and a decode call's counts landed inside a later prefill's split rows - NaNs in a real forward). The filter
    // path's words go at the very end (filter_words below), past every T's split rows: they aren't kept zero.
    return (size_t)(kCoopWords + T * kWsStride + filter_words(T)) * sizeof(uint32_t);
}

int64_t qsa_topk_filter_fallbacks(void *workspace, size_t workspace_bytes, hipStream_t stream) {
    STRIX_CHECK(workspace && workspace_bytes % 4 == 0 && workspace_bytes >= 4, "qsa_topk_filter_fallbacks: workspace ",
                workspace, " of ", workspace_bytes, " bytes");
    uint32_t *counter = static_cast<uint32_t *>(workspace) + workspace_bytes / 4 - 1;  // the last word
    uint32_t v = 0;
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "qsa_topk_filter_fallbacks");
    STRIX_HIP_CHECK(hipMemcpy(&v, counter, 4, hipMemcpyDeviceToHost), "qsa_topk_filter_fallbacks: read");
    STRIX_HIP_CHECK(hipMemset(counter, 0, 4), "qsa_topk_filter_fallbacks: reset");
    return (int64_t)v;
}

void qsa_topk(const float *scores, int64_t ld, int64_t T, int64_t pos0, int64_t k, int32_t *sel, int32_t *nsel,
              void *workspace, size_t workspace_bytes, hipStream_t stream, TopkPath path) {
    const char *fn = "qsa_topk";
    STRIX_CHECK(scores && sel && nsel && workspace, fn, ": null pointer (scores=", (const void *)scores, ", sel=",
                (void *)sel, ", nsel=", (void *)nsel, ", workspace=", workspace, ")");
    STRIX_CHECK(T >= 1 && T <= kMaxTokens && pos0 >= 0, fn, ": T = ", T, " (1..", kMaxTokens, "), pos0 = ", pos0);
    STRIX_CHECK(k >= 1 && k <= kMaxK, fn, ": k = ", k, ", expected 1..", kMaxK);
    const int64_t nb = (pos0 + T) / kQsaBlock;
    STRIX_CHECK(nb <= (int64_t)UINT32_MAX, fn, ": ", nb, " blocks overflow the 32-bit block index");
    STRIX_CHECK(ld >= std::max<int64_t>(nb, 1), fn, ": scores row stride ld = ", ld, " < ", nb, " blocks");
    STRIX_CHECK(workspace_bytes >= qsa_topk_workspace_bytes(T), fn, ": workspace ", workspace_bytes,
                " bytes, need ", qsa_topk_workspace_bytes(T), " for T = ", T);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(workspace) % 4 == 0, fn, ": workspace must be 4-byte aligned");
    STRIX_CHECK(path == TopkPath::Auto || path == TopkPath::Split || path == TopkPath::Row || path == TopkPath::Coop ||
                    path == TopkPath::Filter || path == TopkPath::FilterTight,
                fn, ": path ", (int)path);
    const bool small_t = T <= kQsaTopkRowMaxT;
    // Filter for many queries past the row path's limit (prefill chunks at depth); FilterTight: a candidate list of
    // exactly k, so most rows take the fallback (tests).
    const bool filter_auto = !small_t && nb > kQsaTopkRowMaxBlocks;
    if (path == TopkPath::Filter || path == TopkPath::FilterTight || (path == TopkPath::Auto && filter_auto)) {
        const int64_t cap = path == TopkPath::FilterTight ? k : kFilterCap;
        // The filter words at the end of the workspace (qsa_topk_workspace_bytes): past every T's split rows.
        uint32_t *end = static_cast<uint32_t *>(workspace) + workspace_bytes / 4;
        uint32_t *fallbacks = end - 1, *fw = end - filter_words(T);
        topk_filter_bound_kernel<<<(unsigned)T, kThreads, 0, stream>>>(scores, ld, pos0, (int)k, fw);
        STRIX_HIP_CHECK_LAUNCH(fn, " filter bound T=", T, " pos0=", pos0, " k=", k);
        const dim3 grid((unsigned)std::max<int64_t>(1, (nb + kTopkChunk - 1) / kTopkChunk), (unsigned)T);
        topk_filter_kernel<<<grid, kThreads, 0, stream>>>(scores, ld, pos0, (int)k, fw, cap);
        STRIX_HIP_CHECK_LAUNCH(fn, " filter T=", T, " pos0=", pos0, " k=", k);
        topk_filter_exact_kernel<<<(unsigned)T, kRowThreads, 0, stream>>>(scores, ld, pos0, (int)k, sel, nsel, fw, cap,
                                                                         fallbacks);
        STRIX_HIP_CHECK_LAUNCH(fn, " filter exact T=", T, " pos0=", pos0, " k=", k);
        return;
    }
    // Auto (gfx1151): row up to 16384 blocks; past that coop for verifies (T = 5: 27.8 /
    // 41.1 / 46.8 us at 16k / 32k / 64k blocks vs split 50.4 / 56.3 / 52.0) and for T = 1 up to 40960 blocks (23.7 /
    // 28.5 vs 31.2 / 31.8); T = 1 deeper keeps split (64k: coop 45.4 vs split 32.3).
    const bool coop_auto = small_t && nb > kQsaTopkRowMaxBlocks && (T > 1 || nb <= kQsaTopkCoopMaxBlocksT1);
    if (path == TopkPath::Coop || (path == TopkPath::Auto && coop_auto)) {
        STRIX_CHECK(small_t, fn, ": the coop path takes T <= ", kQsaTopkRowMaxT, ", got T = ", T);
        const int64_t W = std::clamp<int64_t>(std::min<int64_t>((nb + kTopkChunk - 1) / kTopkChunk,
                                                                  coop_max_resident() / T),
                                              1, kCoopMaxWgs);
        STRIX_CHECK(W * T <= coop_max_resident(), fn, ": coop grid ", W, " x ", T, " exceeds the ", coop_max_resident(),
                    " resident workgroups");
        uint32_t *cws = static_cast<uint32_t *>(workspace);
        int ki = (int)k;
        void *args[] = {(void *)&scores, (void *)&ld, (void *)&pos0, (void *)&ki, (void *)&sel, (void *)&nsel, (void *)&cws};
        STRIX_HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<const void *>(topk_coop_kernel), dim3((unsigned)W, (unsigned)T),
                                                   dim3(kThreads), args, 0, stream),
                        fn, " coop T=", T, " pos0=", pos0, " k=", k, " W=", W);
        return;
    }
    // Row also for prefill chunks (any T) up to kQsaTopkRowMaxBlocks: T = 2048 - row 0.33 / 0.47 /
    // 0.71 / 1.24 / 1.80 ms vs split 1.23 / 3.91 / 4.53 / 5.11 / 5.82 at 1k / 4k / 8k / 12k / 16k blocks; at 32k
    // blocks row 14.7 vs split 9.2, so split past the limit.
    if (path == TopkPath::Row || (path == TopkPath::Auto && nb <= kQsaTopkRowMaxBlocks)) {
        STRIX_CHECK(nb < (int64_t)kRowEqMask && kMaxK < (1 << (32 - kRowEqBits)), fn, ": the row path takes rows of fewer than ",
                    kRowEqMask, " blocks (its packed counts), got ", nb);
        topk_row_kernel<<<(unsigned)T, kRowThreads, 0, stream>>>(scores, ld, pos0, (int)k, sel, nsel);
        STRIX_HIP_CHECK_LAUNCH(fn, " row T=", T, " pos0=", pos0, " k=", k);
        return;
    }
    auto *ws = static_cast<uint32_t *>(workspace) + kCoopWords;  // the split rows follow the coop words
    const dim3 grid((unsigned)std::max<int64_t>(1, (nb + kTopkChunk - 1) / kTopkChunk), (unsigned)T);
    if (nb > k)  // otherwise every query takes all its blocks
        for (int pass = 0; pass < 3; ++pass) {
            topk_hist_kernel<<<grid, kThreads, 0, stream>>>(scores, ld, pos0, (int)k, pass, ws);
            STRIX_HIP_CHECK_LAUNCH(fn, " histogram pass ", pass, " T=", T, " pos0=", pos0, " k=", k);
        }
    topk_select_kernel<<<grid, kThreads, 0, stream>>>(scores, ld, pos0, (int)k, sel, nsel, ws);
    STRIX_HIP_CHECK_LAUNCH(fn, " select T=", T, " pos0=", pos0, " k=", k);
    if (nb > k) {
        int p2 = 1;
        while (p2 < k) p2 <<= 1;
        topk_sort_kernel<<<(unsigned)T, kThreads, 0, stream>>>(sel, pos0, (int)k, p2);
        STRIX_HIP_CHECK_LAUNCH(fn, " sort T=", T, " pos0=", pos0, " k=", k);
    }
}

}  // namespace strix::kernels
