#include "kernels/moe_grouped.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/qsplit.hpp"
#include "kernels/wmma_gemm.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRouteThreads = 1024;  // one per bucket: E experts + the invalid-id bucket
constexpr int kChunk = 32;           // Q4 codes per 16-byte load
constexpr int R = kMoeTileRows;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }
// The combine's partials (one per slot, summed by combine_reduce_kernel): FP32, or - the F16 kernel, MoeMath::WmmaF16,
// the served path for Q4 experts - rounded to BF16 (to nearest even) into the same buffer read as uint16_t, half its
// bytes. BF16 partials: the down kernel 88 -> 76 us a token at 8192, the same 400-needle long-context score as FP32
// partials (365 vs 366; F16 partials, as fast, scored 301), NLL inside the noise floor.
__device__ inline void put_partial(float *p, int64_t i, float v, bool bf16) {
    if (bf16) reinterpret_cast<uint16_t *>(p)[i] = f32_to_bf16_bits(v);
    else p[i] = v;
}
__device__ inline float get_partial(const float *p, int64_t i, bool bf16) {
    return bf16 ? bf16_bits_to_f32(reinterpret_cast<const uint16_t *>(p)[i]) : p[i];
}

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

// Workspace layout (int32): header [n_tiles, S, E, n_wtiles], offsets [E + 2] (bucket b's slots at slots[offsets[b]
// .. offsets[b+1]); bucket E = invalid ids), slots [S], tile_bucket [Tmax], tile_start [Tmax] (R-slot tiles, the
// FP32 kernels), wtile_bucket [WTmax], wtile_start [WTmax] (kMoeWmmaTileRows-slot tiles, the WMMA kernels).
constexpr int RW = kMoeWmmaTileRows;
struct RouteLayout {
    int64_t S, E, Tmax, WTmax;
    int64_t offsets() const { return 4; }
    int64_t slots() const { return offsets() + E + 2; }
    int64_t tile_bucket() const { return slots() + S; }
    int64_t tile_start() const { return tile_bucket() + Tmax; }
    int64_t wtile_bucket() const { return tile_start() + Tmax; }
    int64_t wtile_start() const { return wtile_bucket() + WTmax; }
    int64_t ints() const { return wtile_start() + WTmax; }
};
RouteLayout route_layout(int64_t M, int64_t A, int64_t E) {
    const int64_t S = M * A;
    // sum over buckets of ceil(count / R) <= S / R + (E + 1), likewise for RW
    return {S, E, S / R + E + 2, S / RW + E + 2};
}

// Exclusive scan of v over the block's kRouteThreads threads using a 2-stage hierarchical Wave32 scan;
// returns the prefix, *total the sum. Reduces barriers from 22 to 3.
__device__ int block_exclusive_scan(int v, int *buf, int *total) {
    static_assert(kRouteThreads == 1024, "hierarchical scan assumes 1024 threads (32 waves of 32)");
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int wid = t >> 5;

    // Stage 1: Intra-wave inclusive scan
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up(x, o, 32);
        if (lane >= o) x += y;
    }
    if (lane == 31) buf[wid] = x;
    __syncthreads();

    // Stage 2: Wave 0 scans the 32 wave totals
    if (wid == 0) {
        int ws = buf[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up(ws, o, 32);
            if (lane >= o) ws += y;
        }
        buf[lane] = ws - buf[lane];
        if (lane == 31) buf[32] = ws;
    }
    __syncthreads();

    // Stage 3: Add wave prefix to intra-wave sum
    const int incl = x + buf[wid];
    *total = buf[32];
    __syncthreads();
    return incl - v;
}

// One block. Counting sort by bucket (expert b, or b == E: invalid ids) with LDS atomics, each thread taking
// every kRouteThreads-th slot: counts, scans for the bucket and tile offsets, then placement through per-bucket
// cursors. The order of slots within a bucket depends on the atomics, and nothing else does: every output row of the
// grouped GEMMs is computed from its own input row only, never from its tile mates. The first invalid slot (in slot
// order, an atomic minimum) is what err reports.
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_kernel(const int32_t *ids, int64_t S, int64_t E, int64_t Tmax, int64_t WTmax, int32_t *ws, uint32_t *err) {
    __shared__ int cnt[kRouteThreads], cursor[kRouteThreads], buf[kRouteThreads];
    __shared__ int first_bad;
    const int t = threadIdx.x;
    const bool owner = t <= E;
    cnt[t] = 0;
    if (t == 0) first_bad = 0x7fffffff;
    __syncthreads();
    for (int64_t s = t; s < S; s += kRouteThreads) {
        const int32_t id = ids[s];
        const bool bad = id < 0 || id >= E;
        atomicAdd(&cnt[bad ? E : id], 1);
        if (bad) atomicMin(&first_bad, (int)s);
    }
    __syncthreads();
    if (t == 0 && first_bad != 0x7fffffff && atomicCAS(&err[0], 0u, 1u) == 0u) {
        err[1] = (uint32_t)ids[first_bad];
        err[2] = (uint32_t)first_bad;
    }
    const int count = owner ? cnt[t] : 0;
    int total_slots, total_tiles;
    const int start = block_exclusive_scan(count, buf, &total_slots);
    const int tiles = (count + R - 1) / R;
    const int tile0 = block_exclusive_scan(tiles, buf, &total_tiles);
    int total_wtiles;
    const int wtiles = (count + RW - 1) / RW;
    const int wtile0 = block_exclusive_scan(wtiles, buf, &total_wtiles);
    int32_t *offsets = ws + 4, *slots = offsets + E + 2, *tile_bucket = slots + S, *tile_start = tile_bucket + Tmax;
    int32_t *wtile_bucket = tile_start + Tmax, *wtile_start = wtile_bucket + WTmax;
    if (owner) {
        offsets[t] = start;
        cursor[t] = start;
        for (int i = 0; i < tiles; ++i) tile_bucket[tile0 + i] = t, tile_start[tile0 + i] = start + i * R;
        for (int i = 0; i < wtiles; ++i) wtile_bucket[wtile0 + i] = t, wtile_start[wtile0 + i] = start + i * RW;
    }
    if (t == 0) {
        ws[0] = total_tiles, ws[1] = (int32_t)S, ws[2] = (int32_t)E, ws[3] = total_wtiles;
        offsets[E + 1] = total_slots;
    }
    __syncthreads();
    for (int64_t s = t; s < S; s += kRouteThreads) {
        const int32_t id = ids[s];
        slots[atomicAdd(&cursor[(id < 0 || id >= E) ? E : id], 1)] = (int32_t)s;
    }
}

// One tile (up to R slots of one expert) x a block of output rows n: linear_q4_kernel's scheme (a row per KS waves,
// lanes over the K chunks, one weight load serving every slot of the tile) with the x rows gathered through the
// slot list. COMBINE: x rows are h[slot], outputs the FP32 per-slot dots into partial; else x rows are x[slot / A],
// outputs y[slot] in the activation dtype. The invalid-id bucket's tiles write NaN.
template <typename T, int G, bool COMBINE, int HB>
__global__ void __launch_bounds__(kThreads)
    grouped_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const int32_t *ws,
                   int64_t A, int64_t N, int64_t K, int KS, T *y, float *partial) {
    __shared__ float part[kThreads / 32][R];
    __shared__ int64_t xoff[R], ooff[R];
    const int32_t n_tiles = ws[0], S = ws[1], E = ws[2];
    const int64_t tile = blockIdx.y;
    if (tile >= n_tiles) return;  // the bound over-provisions tiles; uniform per block, before any barrier
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2, *tile_bucket = slots + S;
    const int64_t Tmax = (int64_t)S / R + E + 2;
    const int32_t *tile_start = tile_bucket + Tmax;
    const int32_t bucket = tile_bucket[tile], st = tile_start[tile];
    const int rows = min(R, offsets[bucket + 1] - st);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * N;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ks = wave % KS;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / 32 / KS) + wave / KS;
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        if (lane == 0 && ks == 0 && n < N)
            for (int r = 0; r < rows; ++r) {
                if (COMBINE) partial[ooff[r] + n] = __int_as_float(0x7fc00000);
                else store(y, ooff[r] + n, __int_as_float(0x7fc00000));
            }
        return;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (n < N) {  // no early return: every wave reaches the barrier below
        const int64_t row = (int64_t)bucket * N + n;
        const uint8_t *rowb = q + row * qsplit_row_bytes<HB>(K);  // HB: kernels/qsplit.hpp (0 Q4, 1 Q5)
        const uint4 *qrow = reinterpret_cast<const uint4 *>(rowb);
        const uint16_t *srow = scale + row * (K / G), *mrow = minv + row * (K / G);
        for (int64_t k = ((int64_t)ks * 32 + lane) * kChunk; k < K; k += (int64_t)KS * 32 * kChunk) {
            const uint4 packed = qrow[k / kChunk];
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const QHigh32<HB> hb = QHigh32<HB>::load(rowb, K, k);
            const float s = bf16_bits_to_f32(srow[k / G]), mn = bf16_bits_to_f32(mrow[k / G]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < rows) {
                    float xv[kChunk];
                    load_x32(x, xoff[r] + k, xv);
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
                    acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor(acc[r], off, 32);
    }
    if (KS > 1) {
        if (lane == 0)
#pragma unroll
            for (int r = 0; r < R; ++r) part[wave][r] = acc[r];
        __syncthreads();
        if (lane == 0 && ks == 0)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = 0.f;
                for (int i = 0; i < KS; ++i) v += part[wave + i][r];
                acc[r] = v;
            }
    }
    if (lane == 0 && ks == 0 && n < N)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (r < rows) {
                if (COMBINE) partial[ooff[r] + n] = acc[r];
                else store(y, ooff[r] + n, acc[r]);
            }
}

// On the matrix units: a tile (up to TR slots of one expert: the R-slot tiles or the RW-slot ones) x 128 expert rows
// per block, 8 waves each a 16-column strip of up to TR / 16 16x16 WMMA tiles - only the ones holding slots run, so a
// weight pass serves up to TR slots. K in steps of kWmmaKC over scaled codes (kernels/wmma_gemm.hpp numerics).
constexpr int kWmmaBN = 128;
template <typename T, int G, bool COMBINE, int TR, int HB>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const int32_t *ws,
                        int64_t A, int64_t N, int64_t K, T *y, float *partial) {
    static_assert(TR == R || TR == RW, "the route kernel's tile sizes");
    constexpr int TM = TR / 16, NSUB = kWmmaKC / wmma_sub<G>();
    __shared__ __attribute__((aligned(16))) uint16_t xs[TR * kWmmaLd], wsm[kWmmaBN * kWmmaLd];
    __shared__ float rs[TR * NSUB];
    constexpr int BITS = HB == 0 ? 4 : 5;  // WmmaWCodes<5>: formats/q5.hpp's rows
    __shared__ uint16_t sw[kWmmaBN * WmmaWCodes<BITS, G, kWmmaBN>::kWin], mw[kWmmaBN * WmmaWCodes<BITS, G, kWmmaBN>::kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2], n_tiles = TR == R ? ws[0] : ws[3];
    const int64_t tile = blockIdx.y;
    if (tile >= n_tiles) return;  // uniform per block, before any barrier
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2;
    const int32_t *tile_bucket = TR == R ? slots + S : slots + S + 2 * Tmax;
    const int32_t *tile_start = tile_bucket + (TR == R ? Tmax : WTmax);
    const int32_t bucket = tile_bucket[tile], st = tile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), tm = (rows + 15) / 16;
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * N;
    }
    __syncthreads();
    const int64_t n0 = (int64_t)blockIdx.x * kWmmaBN;
    const int rows_w = (int)(N - n0 < kWmmaBN ? N - n0 : kWmmaBN), wave = threadIdx.x / 32;
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) partial[ooff[r] + n0 + c] = __int_as_float(0x7fc00000);
            else store(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    wmma_f32x8 acc[TM][1] = {};
    // Software-pipelined: step k0 + kWmmaKC's global loads are in flight during step k0's WMMA work.
    WmmaXTile<T, TR, G> xt;
    WmmaWCodes<BITS, G, kWmmaBN> wt;
    const int64_t wrow = (int64_t)bucket * N + n0;
    xt.fetch(x, xoff, rows, 0);
    wt.fetch(q, scale, minv, wrow, rows_w, K, 0);
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        xt.stage(xs, rs);
        wt.stage(wsm, sw, mw);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            xt.fetch(x, xoff, rows, k0 + kWmmaKC);
            wt.fetch(q, scale, minv, wrow, rows_w, K, k0 + kWmmaKC);
        }
        wmma_step_scaled<TM, 1, G>(xs, wsm, rs, sw, mw, k0, 0, wave * 16, acc, tm);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = 16 * i + wmma_c_row(v), c = wave * 16 + (int)threadIdx.x % 16;
            if (r < rows && c < rows_w) {
                if (COMBINE) partial[ooff[r] + n0 + c] = acc[i][0][v];
                else store(y, ooff[r] + n0 + c, acc[i][0][v]);
            }
        }
}

// F16 on the matrix units (MoeMath::WmmaF16; Q4 experts): the same tiles and grid
// as grouped_wmma_kernel, but no per-group work in the K loop. x goes to LDS as F16 (exact from BF16 at these ranges);
// the weight codes go to LDS still packed (coalesced loads by the block), each wave decodes its own 16 rows to F16
// in registers (code * scale + min, one F16 rounding), and the matrix units accumulate the whole K extent in FP32.
// (Loading each wave's rows straight from memory instead was 1.65x slower: scattered code and 2-byte scale loads,
// what costs is cache lines touched, not bytes.)
using f16x2 = wmma_f16x2;  // kernels/wmma_gemm.hpp's F16 helpers
__device__ inline float silu_f32(float v) { return v / (1.0f + expf(-v)); }  // the form in kernels/swiglu.hip
__device__ inline void q4_word_to_f16(uint32_t w, f16x2 s, f16x2 m, uint32_t *out) { wmma_q4_word_to_f16(w, s, m, out); }
__device__ inline f16x2 bf16_bits_to_f16x2(uint16_t b) { return wmma_bf16_bits_to_f16x2(b); }

// SWI (gather only): the block's waves 0..3 compute gate columns [j0, j0 + 64) and waves 4..7 the matching up columns
// [I + j0, ...) of the stacked gate_up rows (N = 2I); the epilogue meets them in LDS and stores h = silu(g) u [slot, I]
// directly - SwiGLU from the FP32 accumulators, rounded once, instead of storing gu and running kernels/swiglu.
template <typename T, int G, bool COMBINE, int TR, bool SWI = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                            const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(TR == RW, "the route kernel's WMMA tile size");
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    constexpr int kHalf = kWmmaBN / 2;  // SWI: h columns a block
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a 32-code half-step lies in one group");
    constexpr int TM = TR / 16;
    // LDS: x as F16 [TR][kWmmaLd]; the step's codes packed [kWmmaBN rows][2 x 16 bytes + 16 pad] (coalesced global
    // loads, then each wave reads its own rows); scale / min windows of kWin groups per row, refreshed every kWin groups.
    constexpr int kWin = 8, kCodeLd = 3;  // uint4 per code row (2 used)
    __shared__ __attribute__((aligned(16))) uint16_t xs[TR * kWmmaLd];
    __shared__ uint4 cs[kWmmaBN * kCodeLd];
    __shared__ uint16_t sw[kWmmaBN * kWin], mw[kWmmaBN * kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    __shared__ float ub[SWI ? TR * (kHalf + 1) : 1];  // SWI: the up waves' accumulators for the gate waves
    const int32_t S = ws[1], E = ws[2], n_tiles = ws[3];
    const int64_t tile = blockIdx.y;
    if (tile >= n_tiles) return;  // uniform per block, before any barrier
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2;
    const int32_t *tile_bucket = slots + S + 2 * Tmax, *tile_start = tile_bucket + WTmax;
    const int32_t bucket = tile_bucket[tile], st = tile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), tm = (rows + 15) / 16;
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    // SWI: n0 is the block's first h column j0 and rows_w its h columns; otherwise its first and count of output columns.
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols), wave = threadIdx.x / 32;
    // The expert's weight row (from its first) behind staged row r of kWmmaBN, clamped to a valid row: SWI rows
    // [0, 64) are gate rows n0 + r, rows [64, 128) the up rows N / 2 + n0 + r - 64.
    auto wrow = [&](int r) -> int64_t {
        if constexpr (SWI) {
            const int rr = r % kHalf;
            return (r < kHalf ? 0 : N / 2) + n0 + (rr < rows_w ? rr : 0);
        } else {
            return n0 + (r < rows_w ? r : 0);
        }
    };
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int64_t row0 = (int64_t)bucket * N, n_groups = K / G;
    // Global fetch (prefetched a step ahead): thread t, code row t / 2 (clamped to a valid row), 16-byte half t % 2;
    // at a window start, kWin (scale, min) of kWmmaBN * kWin / kWmmaThreads rows - consecutive threads read
    // consecutive groups of a row (a 16-byte run), not one scattered 2-byte value per lane.
    static_assert(kWmmaBN * 2 == kWmmaThreads, "one 16-byte code load per thread and step");
    constexpr int kWinLoads = kWmmaBN * kWin / kWmmaThreads;
    const int crow = (int)threadIdx.x / 2, chalf = (int)threadIdx.x % 2;
    const uint8_t *cptr = q + (row0 + wrow(crow)) * (K / 2) + chalf * 16;
    uint4 cq;
    uint16_t fs[kWinLoads], fm[kWinLoads];
    auto wfetch = [&](int64_t k0) {
        cq = *reinterpret_cast<const uint4 *>(cptr + k0 / 2);
        const int64_t g0 = k0 / G;
        if (k0 % G == 0 && g0 % kWin == 0)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, r = e / kWin;
                const int64_t g = g0 + e % kWin, row = row0 + wrow(r);
                fs[i] = g < n_groups ? scale[row * n_groups + g] : 0;
                fm[i] = g < n_groups ? minv[row * n_groups + g] : 0;
            }
    };
    WmmaXTile<T, TR, G, false> xt;
    constexpr int kSegs = kWmmaKC / 8, kPasses = (TR * kSegs + kWmmaThreads - 1) / kWmmaThreads;
    const int wr = wave * 16 + (int)threadIdx.x % 16;  // this lane's weight row (B column); lanes l, l + 16 the same
    // The K loop for TMA active 16-row tiles, chosen once per block: a per-tile `if` inside the loop made the compiler
    // copy all the accumulators at every join (913 register moves, 241 VGPRs in the ISA).
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;
        wmma_f32x8 acc[TMA] = {};
        xt.fetch(x, xoff, rows, 0);
        wfetch(0);
        for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
            // stage: x as F16 (BF16 -> F16 exact in F16's normal range), the packed codes, a new scale window
#pragma unroll
            for (int p = 0; p < kPasses; ++p) {
                const int seg = (int)threadIdx.x + p * kWmmaThreads;
                if (seg >= TR * kSegs) break;  // wave-uniform
                uint32_t h[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    h[i] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(xt.b[p][i] << 16),
                                                                                   __uint_as_float(xt.b[p][i] & 0xffff0000u)));
                *reinterpret_cast<uint4 *>(xs + (seg / kSegs) * kWmmaLd + (seg % kSegs) * 8) =
                    make_uint4(h[0], h[1], h[2], h[3]);
            }
            cs[crow * kCodeLd + chalf] = cq;
            if (k0 % G == 0 && (k0 / G) % kWin == 0)
#pragma unroll
                for (int i = 0; i < kWinLoads; ++i) {
                    const int e = (int)threadIdx.x + i * kWmmaThreads;
                    sw[e] = fs[i], mw[e] = fm[i];  // [r][e % kWin], r = e / kWin
                }
            __syncthreads();
            if (k0 + kWmmaKC < K) {  // the next step's loads in flight during this one's math
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wfetch(k0 + kWmmaKC);
            }
            const uint4 c0 = cs[wr * kCodeLd], c1 = cs[wr * kCodeLd + 1];
            const uint32_t wv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            const int ga = (int)((k0 / G) % kWin), gb = (int)(((k0 + 32) / G) % kWin);
            const f16x2 s0 = bf16_bits_to_f16x2(sw[wr * kWin + ga]), m0 = bf16_bits_to_f16x2(mw[wr * kWin + ga]),
                        s1 = bf16_bits_to_f16x2(sw[wr * kWin + gb]), m1 = bf16_bits_to_f16x2(mw[wr * kWin + gb]);
#pragma unroll
            for (int kk = 0; kk < kWmmaKC / 16; ++kk) {
                uint32_t b[8];
                const f16x2 s = kk < 2 ? s0 : s1, m = kk < 2 ? m0 : m1;
                q4_word_to_f16(wv[2 * kk], s, m, b);
                q4_word_to_f16(wv[2 * kk + 1], s, m, b + 4);
                const wmma_u32x8 bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
                const wmma_f16x16 bf = __builtin_bit_cast(wmma_f16x16, bv);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
                    acc[i] = wmma_f16(wmma_load_rows_a16_f16(xs + 16 * i * kWmmaLd + kk * 16, kWmmaLd), bf, acc[i]);
            }
            __syncthreads();
        }
        if constexpr (SWI) {
            const int c = (wave % 4) * 16 + (int)threadIdx.x % 16;
            if (wave >= 4)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int v = 0; v < 8; ++v) ub[(16 * i + wmma_c_row(v)) * (kHalf + 1) + c] = acc[i][v];
            __syncthreads();
            if (wave < 4)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * i + wmma_c_row(v);
                        if (r < rows && c < rows_w)
                            store(y, ooff[r] + n0 + c, silu_f32(acc[i][v]) * ub[r * (kHalf + 1) + c]);
                    }
        } else {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int r = 16 * i + wmma_c_row(v), c = wave * 16 + (int)threadIdx.x % 16;
                    if (r < rows && c < rows_w) {
                        if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][v], pbf16);
                        else store(y, ooff[r] + n0 + c, acc[i][v]);
                    }
                }
        }
    };
    static_assert(TM == 4, "the dispatch below covers 1..4 active tiles");
    switch (tm) {  // uniform per block
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// y[m, n] = sum_a coef[m, a] * partial[m * A + a, n], a in order, rounded once; NaN if any of token m's ids is
// invalid (checked here too, so the row is NaN whatever that slot's partial holds).
template <typename T>
__global__ void __launch_bounds__(kThreads)
    combine_reduce_kernel(const float *partial, const int32_t *ids, const float *coef, int64_t M, int64_t A,
                          int64_t E, int64_t N, T *y, bool pbf16) {
    const int64_t m = blockIdx.y;
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    bool ok = true;
    float v = 0.f;
#pragma unroll 4
    for (int64_t a = 0; a < A; ++a) {
        const int32_t id = ids[m * A + a];
        ok = ok && id >= 0 && id < E;
        v = fmaf(coef[m * A + a], get_partial(partial, (m * A + a) * N + n, pbf16), v);
    }
    store(y, m * N + n, ok ? v : __int_as_float(0x7fc00000));
}

// Split-K as linear_q4 (each wave ~2048 codes of a row): K = 2560 and 640 give 1.
int split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K / (2048 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

void check_common(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, int64_t M, int64_t A,
                  const void *ws, size_t ws_bytes, const void *y, Act act) {
    STRIX_CHECK(x != nullptr && y != nullptr && ws != nullptr, fn, ": null pointer (x=", x, ", y=", y, ", ws=", ws, ")");
    STRIX_CHECK(w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn, ": null weight pointer (q=",
                (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.K >= w.G && w.K % w.G == 0 && w.K <= (1 << 24), fn, ": K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and <= 2^24");
    STRIX_CHECK(E >= 1 && E <= kMoeMaxExperts && w.N % E == 0, fn, ": weight has ", w.N, " rows for E = ", E,
                " experts (E must be 1..", kMoeMaxExperts, " and divide the rows)");
    STRIX_CHECK(M >= 1 && A >= 1 && A <= 64 && A <= E && M * A <= kMoeMaxSlots, fn, ": M = ", M, " tokens x A = ", A,
                " slots (A 1..min(64, E = ", E, "), M*A <= ", kMoeMaxSlots, ")");
    STRIX_CHECK(ws_bytes >= moe_route_workspace_bytes(M, A, E), fn, ": workspace of ", ws_bytes, " bytes, needs ",
                moe_route_workspace_bytes(M, A, E), " for M = ", M, ", A = ", A, ", E = ", E);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0, fn,
                ": code array ", (const void *)w.q, " and input ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
}

void check_no_overlap(const char *fn, const char *a_name, const void *a, int64_t a_bytes, const char *b_name,
                      const void *b, int64_t b_bytes) {
    const auto ab = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    STRIX_CHECK(ab + (uintptr_t)a_bytes <= bb || bb + (uintptr_t)b_bytes <= ab, fn, ": ", a_name, " [", a, ", +",
                a_bytes, " B) overlaps ", b_name, " [", b, ", +", b_bytes, " B)");
}

template <typename T, bool COMBINE, int HB, bool SWI = false>
void launch_grouped(const T *x, const Q4DeviceView &w, int64_t E, const void *ws, int64_t M, int64_t A, T *y,
                    float *partial, MoeMath math, hipStream_t stream) {
    const int64_t N = w.N / E;
    STRIX_CHECK(!SWI || (math == MoeMath::WmmaF16 && N % 2 == 0), "moe grouped (SwiGLU): needs MoeMath::WmmaF16 and an ",
                "even gate_up width, got math ", (int)math, ", N = ", N);
    const auto *wsi = static_cast<const int32_t *>(ws);
    if (math == MoeMath::WmmaF16) {
        STRIX_CHECK(HB == 0, "moe grouped (WMMA F16): Q4 experts only (Q5 runs on MoeMath::WmmaBf16)");
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA F16): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).WTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16): ", tiles, " tile slots exceed the grid");
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaBN / 2 : kWmmaBN;  // SWI: h columns
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_kernel<T, 32, COMBINE, RW, SWI><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_kernel<T, 64, COMBINE, RW, SWI><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_kernel<T, 128, COMBINE, RW, SWI><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    if (math == MoeMath::WmmaBf16) {
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA): needs wave32 (gfx11 WMMA)");
        // Always the RW-slot tiles: on the real model (skewed routing: the shared expert takes every token, popular
        // experts many) they beat 16-slot tiles at both chunk sizes - prefill to 8192 786 vs 748 t/s (T = 512), 908 vs
        // 837 (T = 2048) - although uniform routing favours 16 at T = 512.
        const int64_t tiles = route_layout(M, A, E).WTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA): ", tiles, " tile slots exceed the grid");
        dim3 grid((unsigned)((N + kWmmaBN - 1) / kWmmaBN), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_kernel<T, 32, COMBINE, RW, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial); break;
            case 64: grouped_wmma_kernel<T, 64, COMBINE, RW, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial); break;
            default: grouped_wmma_kernel<T, 128, COMBINE, RW, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial); break;
        }
        return;
    }
    STRIX_CHECK(math == MoeMath::F32, "moe grouped: unknown math ", (int)math);
    const int KS = split_k(w.K);
    const int waves = kThreads / device_warp_size(kThreads);
    STRIX_CHECK(waves % KS == 0, "moe grouped: split-K factor ", KS, " must divide the ", waves, " waves");
    const int64_t rows_per_block = waves / KS, Tmax = route_layout(M, A, E).Tmax;
    STRIX_CHECK(Tmax <= 65535 * 16, "moe grouped: ", Tmax, " tile slots exceed the grid");
    dim3 grid((unsigned)((N + rows_per_block - 1) / rows_per_block), (unsigned)Tmax), block(kThreads);
    switch (w.G) {
        case 32: grouped_kernel<T, 32, COMBINE, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, KS, y, partial); break;
        case 64: grouped_kernel<T, 64, COMBINE, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, KS, y, partial); break;
        default: grouped_kernel<T, 128, COMBINE, HB><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, KS, y, partial); break;
    }
}

}  // namespace

size_t moe_route_workspace_bytes(int64_t M, int64_t A, int64_t E) {
    STRIX_CHECK(M >= 1 && A >= 1 && E >= 1 && E <= kMoeMaxExperts && M * A <= kMoeMaxSlots,
                "moe_route_workspace_bytes: M = ", M, ", A = ", A, ", E = ", E, " (E <= ", kMoeMaxExperts,
                ", M*A <= ", kMoeMaxSlots, ")");
    return (size_t)route_layout(M, A, E).ints() * 4;
}

void moe_group_routes(const int32_t *ids, int64_t M, int64_t A, int64_t E, void *ws, size_t ws_bytes, uint32_t *err,
                      hipStream_t stream) {
    const char *fn = "moe_group_routes";
    STRIX_CHECK(ids != nullptr && ws != nullptr && err != nullptr, fn, ": null pointer (ids=", (const void *)ids,
                ", ws=", ws, ", err=", (const void *)err, ")");
    STRIX_CHECK(M >= 1 && A >= 1 && A <= 64 && E >= 1 && E <= kMoeMaxExperts && M * A <= kMoeMaxSlots, fn, ": M = ",
                M, ", A = ", A, ", E = ", E, " (A 1..64, E 1..", kMoeMaxExperts, ", M*A <= ", kMoeMaxSlots, ")");
    STRIX_CHECK(ws_bytes >= moe_route_workspace_bytes(M, A, E), fn, ": workspace of ", ws_bytes, " bytes, needs ",
                moe_route_workspace_bytes(M, A, E));
    const RouteLayout L = route_layout(M, A, E);
    moe_route_kernel<<<1, kRouteThreads, 0, stream>>>(ids, L.S, E, L.Tmax, L.WTmax, static_cast<int32_t *>(ws), err);
    STRIX_HIP_CHECK_LAUNCH(fn, " M=", M, " A=", A, " E=", E);
}

namespace {

template <int HB>
void experts_gather_grouped(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                            size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math, hipStream_t stream) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    check_common(fn, x, w, E, M, A, ws, ws_bytes, y, act);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    check_no_overlap(fn, "y", y, M * A * N * es, "x", x, M * w.K * es);
    if (act == Act::F32)
        launch_grouped<float, false, HB>(static_cast<const float *>(x), w, E, ws, M, A, static_cast<float *>(y), nullptr,
                                     math, stream);
    else
        launch_grouped<uint16_t, false, HB>(static_cast<const uint16_t *>(x), w, E, ws, M, A, static_cast<uint16_t *>(y),
                                        nullptr, math, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math == MoeMath::WmmaBf16 ? " wmma" : math == MoeMath::WmmaF16 ? " wmma-f16" : "", " M=", M, " A=", A, " E=",
                           E, " N=", N, " K=", w.K, " G=", w.G);
}

void experts_gather_grouped_swiglu(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                   size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                   hipStream_t stream) {
    check_common(fn, x, w, E, M, A, ws, ws_bytes, h, act);
    STRIX_CHECK(math == MoeMath::WmmaF16, fn, ": the fused SwiGLU runs on MoeMath::WmmaF16 only, got ", (int)math);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    STRIX_CHECK(N % 2 == 0, fn, ": gate_up has ", N, " rows an expert, expected an even count (gate rows then up rows)");
    check_no_overlap(fn, "h", h, M * A * (N / 2) * es, "x", x, M * w.K * es);
    if (act == Act::F32)
        launch_grouped<float, false, 0, true>(static_cast<const float *>(x), w, E, ws, M, A, static_cast<float *>(h),
                                              nullptr, math, stream);
    else
        launch_grouped<uint16_t, false, 0, true>(static_cast<const uint16_t *>(x), w, E, ws, M, A,
                                                 static_cast<uint16_t *>(h), nullptr, math, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " wmma-f16 M=", M, " A=", A, " E=", E, " N=", N, " K=", w.K,
                           " G=", w.G);
}

template <int HB>
void experts_combine_grouped(const char *fn, const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                             const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M, int64_t A,
                             void *y, Act act, MoeMath math, hipStream_t stream) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    check_common(fn, h, w, E, M, A, ws, ws_bytes, y, act);
    STRIX_CHECK(ids != nullptr && coef != nullptr && partial != nullptr, fn, ": null pointer (ids=",
                (const void *)ids, ", coef=", (const void *)coef, ", partial=", (const void *)partial, ")");
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    check_no_overlap(fn, "y", y, M * N * es, "h", h, M * A * w.K * es);
    check_no_overlap(fn, "y", y, M * N * es, "partial", partial, M * A * N * 4);
    check_no_overlap(fn, "partial", partial, M * A * N * 4, "h", h, M * A * w.K * es);
    dim3 grid((unsigned)((N + kThreads - 1) / kThreads), (unsigned)M), block(kThreads);
    const bool pbf16 = math == MoeMath::WmmaF16;  // the F16 kernel writes BF16 partials (put_partial), the others FP32
    if (act == Act::F32) {
        launch_grouped<float, true, HB>(static_cast<const float *>(h), w, E, ws, M, A, nullptr, partial, math, stream);
        combine_reduce_kernel<float><<<grid, block, 0, stream>>>(partial, ids, coef, M, A, E, N,
                                                                 static_cast<float *>(y), pbf16);
    } else {
        launch_grouped<uint16_t, true, HB>(static_cast<const uint16_t *>(h), w, E, ws, M, A, nullptr, partial, math, stream);
        combine_reduce_kernel<uint16_t><<<grid, block, 0, stream>>>(partial, ids, coef, M, A, E, N,
                                                                    static_cast<uint16_t *>(y), pbf16);
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math == MoeMath::WmmaBf16 ? " wmma" : math == MoeMath::WmmaF16 ? " wmma-f16" : "", " M=", M, " A=", A, " E=",
                           E, " N=", N, " K=", w.K, " G=", w.G);
}

}  // namespace

void linear_q4_experts_gather_grouped(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream) {
    experts_gather_grouped<0>("linear_q4_experts_gather_grouped", x, w, E, ws, ws_bytes, M, A, y, act, math, stream);
}

void linear_q4_experts_gather_grouped_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream) {
    experts_gather_grouped_swiglu("linear_q4_experts_gather_grouped_swiglu", x, w, E, ws, ws_bytes, M, A, h, act, math,
                                  stream);
}

void linear_q4_experts_combine_grouped(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream) {
    experts_combine_grouped<0>("linear_q4_experts_combine_grouped", h, w, E, ids, coef, ws, ws_bytes, partial, M, A,
                               y, act, math, stream);
}

void linear_q5_experts_gather_grouped(const void *x, const Q5DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream) {
    experts_gather_grouped<1>("linear_q5_experts_gather_grouped", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ws,
                              ws_bytes, M, A, y, act, math, stream);
}

void linear_q5_experts_combine_grouped(const void *h, const Q5DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream) {
    experts_combine_grouped<1>("linear_q5_experts_combine_grouped", h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ids,
                               coef, ws, ws_bytes, partial, M, A, y, act, math, stream);
}

}  // namespace strix::kernels
