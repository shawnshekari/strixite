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
// The F16 kernels' x staging: a BF16 pair as F16 bits, rounded toward zero - exact in F16's normal range.
__device__ inline uint32_t bf16x2_to_f16x2(uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(b << 16), __uint_as_float(b & 0xffff0000u)));
}
// The SwiGLU gather's h: BF16 bits, or (XH) the down's staged F16 of them - BF16 rounding, then bf16x2_to_f16x2's
// conversion - so a down kernel that stages F16 h as it is reads what it would have converted from BF16 h.
template <bool XH>
__device__ inline uint16_t h_bits(float v) {
    const uint16_t b = f32_to_bf16_bits(v);
    return XH ? (uint16_t)bf16x2_to_f16x2(b) : b;
}
template <bool XH>
__device__ inline void store_h(uint16_t *p, int64_t i, float v) { p[i] = h_bits<XH>(v); }
template <bool XH>
__device__ inline void store_h(float *p, int64_t i, float v) {
    static_assert(!XH, "F16 h: BF16 activations only");
    p[i] = v;
}
// The combine's partials (one per slot, summed by combine_reduce_kernel): FP32, or - the F16 kernels, MoeMath::WmmaF16,
// the served path - rounded to BF16 (to nearest even) into the same buffer read as uint16_t, half its
// bytes. BF16 partials: the down kernel 88 -> 76 us a token at 8192, the same 400-needle long-context score as FP32
// partials (365 vs 366; F16 partials, as fast, scored 301), NLL inside the noise floor.
__device__ inline void put_partial(float *p, int64_t i, float v, bool bf16) {
    if (bf16) reinterpret_cast<uint16_t *>(p)[i] = f32_to_bf16_bits(v);
    else p[i] = v;
}
__device__ inline float get_partial(const float *p, int64_t i, bool bf16) {
    return bf16 ? bf16_bits_to_f32(reinterpret_cast<const uint16_t *>(p)[i]) : p[i];
}

// A workgroup barrier that orders LDS only (as attention.hip's lds_barrier): __syncthreads' workgroup fence also waits
// for every global load in flight (s_waitcnt vmcnt(0)), which drains the K loops' loads issued ahead.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
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
// FP32 kernels), wtile_bucket [WTmax], wtile_start [WTmax] (kMoeWmmaTileRows-slot tiles, the WMMA kernels),
// xtile_bucket [XTmax], xtile_start [XTmax] (kMoeWmmaWideRows-slot tiles, MoeMath::WmmaF16Wide), then n_xtiles.
constexpr int RW = kMoeWmmaTileRows, RX = kMoeWmmaWideRows;
struct RouteLayout {
    int64_t S, E, Tmax, WTmax, XTmax;
    int64_t offsets() const { return 4; }
    int64_t slots() const { return offsets() + E + 2; }
    int64_t tile_bucket() const { return slots() + S; }
    int64_t tile_start() const { return tile_bucket() + Tmax; }
    int64_t wtile_bucket() const { return tile_start() + Tmax; }
    int64_t wtile_start() const { return wtile_bucket() + WTmax; }
    int64_t xtile_bucket() const { return wtile_start() + WTmax; }
    int64_t xtile_start() const { return xtile_bucket() + XTmax; }
    int64_t n_xtiles() const { return xtile_start() + XTmax; }
    int64_t ints() const { return n_xtiles() + 1; }
};
RouteLayout route_layout(int64_t M, int64_t A, int64_t E) {
    const int64_t S = M * A;
    // sum over buckets of ceil(count / R) <= S / R + (E + 1), likewise for RW and RX
    return {S, E, S / R + E + 2, S / RW + E + 2, S / RX + E + 2};
}

// Exclusive scan of v over the block's kRouteThreads threads (Hillis-Steele in LDS); returns the prefix, *total
// the sum.
__device__ int block_exclusive_scan(int v, int *buf, int *total) {
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int off = 1; off < kRouteThreads; off <<= 1) {
        const int add = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += add;
        __syncthreads();
    }
    const int incl = buf[t];
    *total = buf[kRouteThreads - 1];
    __syncthreads();
    return incl - v;
}

// One block. Counting sort by bucket (expert b, or b == E: invalid ids) with LDS atomics, each thread taking
// every kRouteThreads-th slot: counts, scans for the bucket and tile offsets, then placement through per-bucket
// cursors. The order of slots within a bucket depends on the atomics, and nothing else does: every output row of the
// grouped GEMMs is computed from its own input row only, never from its tile mates. The first invalid slot (in slot
// order, an atomic minimum) is what err reports.
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_kernel(const int32_t *ids, int64_t S, int64_t E, int64_t Tmax, int64_t WTmax, int64_t XTmax, int32_t *ws,
                     uint32_t *err) {
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
    int total_xtiles;
    const int xtiles = (count + RX - 1) / RX;
    const int xtile0 = block_exclusive_scan(xtiles, buf, &total_xtiles);
    int32_t *offsets = ws + 4, *slots = offsets + E + 2, *tile_bucket = slots + S, *tile_start = tile_bucket + Tmax;
    int32_t *wtile_bucket = tile_start + Tmax, *wtile_start = wtile_bucket + WTmax;
    int32_t *xtile_bucket = wtile_start + WTmax, *xtile_start = xtile_bucket + XTmax;
    if (owner) {
        offsets[t] = start;
        cursor[t] = start;
        for (int i = 0; i < tiles; ++i) tile_bucket[tile0 + i] = t, tile_start[tile0 + i] = start + i * R;
        for (int i = 0; i < wtiles; ++i) wtile_bucket[wtile0 + i] = t, wtile_start[wtile0 + i] = start + i * RW;
        for (int i = 0; i < xtiles; ++i) xtile_bucket[xtile0 + i] = t, xtile_start[xtile0 + i] = start + i * RX;
    }
    if (t == 0) {
        ws[0] = total_tiles, ws[1] = (int32_t)S, ws[2] = (int32_t)E, ws[3] = total_wtiles;
        offsets[E + 1] = total_slots;
        xtile_start[XTmax] = total_xtiles;  // n_xtiles
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
    const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, ks = wave % KS;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / warpSize / KS) + wave / KS;
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
        for (int64_t k = ((int64_t)ks * warpSize + lane) * kChunk; k < K; k += (int64_t)KS * warpSize * kChunk) {
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
                    for (int i = 0; i < kChunk; ++i) {
                        dot = fmaf(xv[i], (float)(((words[i / 8] >> (4 * (i % 8))) & 0xFu) | hb.at(i)), dot);
                        xsum += xv[i];
                    }
                    acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int off = warpSize / 2; off > 0; off >>= 1) acc[r] += __shfl_xor(acc[r], off);
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

// F16 on the matrix units (MoeMath::WmmaF16; Q4 or Q5 experts, HB as kernels/qsplit.hpp): the same tiles and grid
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
template <typename T, int G, bool COMBINE, int TR, int HB, bool SWI = false, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                            const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(TR == RW, "the route kernel's WMMA tile size");
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    constexpr int kHalf = kWmmaBN / 2;  // SWI: h columns a block
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a 32-code half-step lies in one group");
    constexpr int TM = TR / 16;
    // LDS: x as F16 [TR][kWmmaLd]; the step's codes packed [kWmmaBN rows][2 x 16 bytes + 16 pad] (coalesced global
    // loads, then each wave reads its own rows) - Q5: the 64 codes' 5th bits in the pad's first 8 bytes; scale / min
    // windows of kWin groups per row, refreshed every kWin groups.
    constexpr int kWin = 8, kCodeLd = 3;  // uint4 per code row (2 used, Q5 3)
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
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int64_t row0 = (int64_t)bucket * N, n_groups = K / G;
    // Global fetch (prefetched a step ahead): thread t, code row t / 2 (clamped to a valid row), 16-byte half t % 2;
    // at a window start, kWin (scale, min) of kWmmaBN * kWin / kWmmaThreads rows - consecutive threads read
    // consecutive groups of a row (a 16-byte run), not one scattered 2-byte value per lane.
    static_assert(kWmmaBN * 2 == kWmmaThreads, "one 16-byte code load per thread and step");
    constexpr int kWinLoads = kWmmaBN * kWin / kWmmaThreads;
    // Q5: the thread's 32 codes' 5th bits too, a 4-byte load from the row's high plane.
    const int crow = (int)threadIdx.x / 2, chalf = (int)threadIdx.x % 2;
    const uint8_t *rptr = q + (row0 + wrow(crow)) * qsplit_row_bytes<HB>(K);
    const uint8_t *cptr = rptr + chalf * 16, *hptr = rptr + K / 2 + chalf * 4;
    uint4 cq;
    uint32_t hq = 0;
    uint16_t fs[kWinLoads], fm[kWinLoads];
    auto wfetch = [&](int64_t k0) {
        cq = *reinterpret_cast<const uint4 *>(cptr + k0 / 2);
        if constexpr (HB == 1) hq = *reinterpret_cast<const uint32_t *>(hptr + k0 / 8);
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
                    h[i] = XH && COMBINE ? xt.b[p][i] : bf16x2_to_f16x2(xt.b[p][i]);  // XH: x is F16 already
                *reinterpret_cast<uint4 *>(xs + (seg / kSegs) * kWmmaLd + (seg % kSegs) * 8) =
                    make_uint4(h[0], h[1], h[2], h[3]);
            }
            cs[crow * kCodeLd + chalf] = cq;
            if constexpr (HB == 1) reinterpret_cast<uint32_t *>(cs + crow * kCodeLd + 2)[chalf] = hq;
            if (k0 % G == 0 && (k0 / G) % kWin == 0)
#pragma unroll
                for (int i = 0; i < kWinLoads; ++i) {
                    const int e = (int)threadIdx.x + i * kWmmaThreads;
                    sw[e] = fs[i], mw[e] = fm[i];  // [r][e % kWin], r = e / kWin
                }
            lds_barrier();
            if (k0 + kWmmaKC < K) {  // the next step's loads in flight during this one's math
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wfetch(k0 + kWmmaKC);
            }
            const uint4 c0 = cs[wr * kCodeLd], c1 = cs[wr * kCodeLd + 1];
            const uint32_t wv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            uint32_t hv[2] = {0, 0};  // Q5: codes 0..31's and 32..63's 5th bits
            if constexpr (HB == 1) {
                const uint2 h2 = *reinterpret_cast<const uint2 *>(cs + wr * kCodeLd + 2);
                hv[0] = h2.x, hv[1] = h2.y;
            }
            const int ga = (int)((k0 / G) % kWin), gb = (int)(((k0 + 32) / G) % kWin);
            const f16x2 s0 = bf16_bits_to_f16x2(sw[wr * kWin + ga]), m0 = bf16_bits_to_f16x2(mw[wr * kWin + ga]),
                        s1 = bf16_bits_to_f16x2(sw[wr * kWin + gb]), m1 = bf16_bits_to_f16x2(mw[wr * kWin + gb]);
#pragma unroll
            for (int kk = 0; kk < kWmmaKC / 16; ++kk) {
                uint32_t b[8];
                const f16x2 s = kk < 2 ? s0 : s1, m = kk < 2 ? m0 : m1;
                if constexpr (HB == 1) {  // word w's 5th bits: byte w % 4 of hv[w / 4]
                    wmma_q5_word_to_f16(wv[2 * kk], hv[kk / 2] >> (16 * (kk % 2)), s, m, b);
                    wmma_q5_word_to_f16(wv[2 * kk + 1], hv[kk / 2] >> (16 * (kk % 2) + 8), s, m, b + 4);
                } else {
                    q4_word_to_f16(wv[2 * kk], s, m, b);
                    q4_word_to_f16(wv[2 * kk + 1], s, m, b + 4);
                }
                const wmma_u32x8 bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
                const wmma_f16x16 bf = __builtin_bit_cast(wmma_f16x16, bv);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
                    acc[i] = wmma_f16(wmma_load_rows_a16_f16(xs + 16 * i * kWmmaLd + kk * 16, kWmmaLd), bf, acc[i]);
            }
            lds_barrier();  // the next step's loads stay in flight
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
                            store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][v]) * ub[r * (kHalf + 1) + c]);
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

// MoeMath::WmmaF16Wide: grouped_wmma_f16_kernel's math on linear_wmma_f16_kernel's structure. A block takes up to
// RX = 128 slots of one expert x 128 weight rows; each K step dequantizes the weight tile once into LDS (every code
// once, by all 256 threads) where the F16 kernel's waves decode their own 16 rows in registers (lanes l and l + 16 the
// same row: twice) for at most 4 row tiles. Waves are 2 (row tiles, interleaved: wave row wm takes tiles wm, wm + 2,
// ..) x 4 (32 weight rows each, 2 column tiles), so a block's active tiles split evenly and the per-wave count is
// block-uniform (one specialized K loop per count, no branch around a WMMA). The F16 weights (wmma_stage_codes_f16 - the same
// wmma_q4_word_to_f16 decode), the F16 x and each output tile's WMMA sequence over K are the F16 kernel's, so the
// outputs are bit-identical to it. SWI: staged rows [0, 64) are gate rows, [64, 128) the matching up rows; waves of
// column group wn < 2 hold gate columns, wn >= 2 the up columns, met in LDS (the tiles' bytes, free after the loop).
// HB 1: Q5 codes (WmmaWCodes<5>, the same F16 decode with the 5th bit).
template <typename T, int G, bool COMBINE, bool SWI, int HB, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                 const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a 16-byte code load lies in one group");
    using WT = WmmaWCodes<HB == 0 ? 4 : 5, G, kWmmaBN>;
    constexpr int TR = RX, kHalf = kWmmaBN / 2, kTileElems = (TR + kWmmaBN) * kWmmaLd;
    static_assert(!SWI || TR * (kHalf + 1) * 4 <= kTileElems * 2, "SWI: the up accumulators fit in the tiles' LDS");
    constexpr int kOutLd = kWmmaBN + 8;  // COMBINE: the BF16 output tile's LDS row stride (16-byte aligned rows)
    static_assert(TR * kOutLd <= kTileElems && kOutLd * 2 % 16 == 0, "COMBINE: the output tile fits in the tiles' LDS");
    __shared__ __attribute__((aligned(16))) uint16_t tiles_lds[kTileElems];
    uint16_t *xs = tiles_lds, *wsm = tiles_lds + TR * kWmmaLd;
    float *ub = reinterpret_cast<float *>(tiles_lds);  // SWI, after the K loop
    __shared__ uint16_t sw[kWmmaBN * WT::kWin], mw[kWmmaBN * WT::kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;  // n_xtiles; uniform per block, before any barrier
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    const int64_t wbase = (int64_t)bucket * N + n0;
    // 16-byte partial stores: every slot row's columns start 16-byte aligned and the block's columns are whole 8-column
    // chunks (uniform per launch)
    const bool vec_out = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0;
    // Staged weight row r -> its global row in the stacked [E * N, K] codes (< 0: past the block's columns). Arithmetic,
    // not an LDS table: the table cost ~30 VGPRs (250 -> occupancy 4; this form 217 -> 6, gfx1151).
    const auto row_of = [&](int r) -> int64_t {
        const int c = SWI ? r % kHalf : r;
        return c < rows_w ? wbase + (SWI && r >= kHalf ? N / 2 - kHalf : 0) + r : -1;
    };
    WmmaXTile<T, TR, G, false> xt;
    WT wt;
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;  // this wave's row tiles: wm, wm + 2, .., 2 (TMA - 1) + wm
        wmma_f32x8 acc[TMA][2] = {};
        xt.fetch(x, xoff, rows, 0);
        wt.fetch_mapped(q, scale, minv, row_of, K, 0);
        for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
            if (wt.win) {  // uniform: depends on k0 only
                wt.stage_window(sw, mw);
                lds_barrier();
            }
            if constexpr (XH && COMBINE) xt.stage(xs, nullptr);  // x is F16 already: staged as it is
            else xt.stage_f16(xs);
            wt.stage_f16(wsm, sw, mw, k0);
            lds_barrier();
            if (k0 + kWmmaKC < K) {  // the next step's loads in flight during this one's math
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wt.fetch_mapped(q, scale, minv, row_of, K, k0 + kWmmaKC);
            }
#pragma unroll
            for (int kk = 0; kk < kWmmaKC; kk += 16) {
                wmma_f16x16 a[TMA], b[2];
#pragma unroll
                for (int i = 0; i < TMA; ++i) a[i] = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int j = 0; j < 2; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 32 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc[i][j] = wmma_f16(a[i], b[j], acc[i][j]);
            }
            lds_barrier();  // the next step's loads stay in flight
        }
        if constexpr (SWI) {
            if (wn >= 2)  // up columns: hand them to the gate waves
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = (wn - 2) * 32 + 16 * j + (int)threadIdx.x % 16;
                            ub[r * (kHalf + 1) + c] = acc[i][j][v];
                        }
            __syncthreads();
            if (wn < 2)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                            if (r < rows && c < rows_w)
                                store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][j][v]) * ub[r * (kHalf + 1) + c]);
                        }
        } else if (COMBINE && pbf16 && vec_out) {
            // BF16 partials through LDS (the tiles' bytes, free after the loop's last barrier): each value rounded as
            // put_partial does, then written a 16-byte chunk (8 columns of one slot's row) a store - where 2-byte
            // stores, a half-wave's 32 bytes each, were the down's epilogue.
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        tiles_lds[r * kOutLd + c] = f32_to_bf16_bits(acc[i][j][v]);
                    }
            __syncthreads();
            uint16_t *pb = reinterpret_cast<uint16_t *>(partial);
            for (int e = (int)threadIdx.x; e < rows * (kWmmaBN / 8); e += kWmmaThreads) {
                const int r = e / (kWmmaBN / 8), c = e % (kWmmaBN / 8) * 8;
                if (c >= rows_w) continue;  // rows_w % 8 == 0 (N % 8 == 0, n0 % kWmmaBN == 0): whole chunks only
                *reinterpret_cast<uint4 *>(pb + ooff[r] + n0 + c) =
                    *reinterpret_cast<const uint4 *>(tiles_lds + r * kOutLd + c);
            }
        } else {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        if (r < rows && c < rows_w) {
                            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][j][v], pbf16);
                            else store(y, ooff[r] + n0 + c, acc[i][j][v]);
                        }
                    }
        }
    };
    switch ((nt + 1) / 2) {  // uniform per block: the busier wave row's tiles
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// MoeMath::WmmaF16Wide256: the wide kernel's math on 128-slot x 256-weight-row blocks, K steps of 32, each wave 4 x 4
// output tiles where the 128 x 128 form held 4 x 2. Per WMMA that halves the fragment loads from LDS and the x tile's
// conversion and loads. Thread t stages weight row t: its 32 codes a step (one 16-byte load, one group) and that
// group's scale / min, loaded by the thread when a step opens a group - no scale-window LDS or its barrier. LDS rows
// are 40 F16 (80 bytes: 8 lanes' 16-byte reads hit distinct banks). SWI: wave column group wn stages gate rows
// [32 wn, 32 wn + 32) of the block's h columns as its tiles 0-1 and the matching up rows as tiles 2-3, so
// h = silu(g) u forms in registers. The staged F16 values (wmma_stage_codes_f16, the x conversion) and each output
// tile's WMMA sequence over K are the wide kernel's, so the outputs are bit-identical to it. HB 1 (Q5): each thread also
// loads its row's 32 5th bits a step (4 bytes of the high plane).
template <typename T, int G, bool COMBINE, bool SWI, int HB, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide256_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                    const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a step's 32 codes lie in one group");
    constexpr int TR = RX, BW = kWmmaThreads, KC = 32, LD = KC + 8, kSegs = KC / 8;
    constexpr int kXPasses = TR * kSegs / kWmmaThreads;
    static_assert(TR * kSegs % kWmmaThreads == 0 && LD * 2 % 16 == 0, "whole x passes, 16-byte aligned LDS rows");
    __shared__ __attribute__((aligned(16))) uint16_t xs[TR * LD], wsm[BW * LD];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;  // n_xtiles; uniform per block, before any barrier
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? BW / 2 : BW;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    // SWI: 16-byte h stores - every slot row's columns start 16-byte aligned and the block's columns are whole 8-column
    // chunks (uniform per launch)
    const bool vec_out = (N / 2) % 8 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0;
    // This thread's staged weight row (< 0: past the block's columns, staged as zeros).
    const int64_t wrow = [&]() -> int64_t {
        const int r = (int)threadIdx.x;
        if constexpr (SWI) {
            const int c = (r / 64) * 32 + r % 32;
            return c < rows_w ? (int64_t)bucket * N + (r % 64 >= 32 ? N / 2 : 0) + n0 + c : -1;
        } else {
            return r < rows_w ? (int64_t)bucket * N + n0 + r : -1;
        }
    }();
    const int64_t groups = K / G;
    uint4 qc = make_uint4(0, 0, 0, 0);  // this step's codes of row wrow
    uint16_t sb = 0, mb = 0;            // their group's scale / min (BF16 bits)
    uint32_t xb[kXPasses][4];           // 8 BF16 x values a pass
    uint32_t hc = 0;                    // Q5: the step's codes' 5th bits
    const auto fetch = [&](int64_t k0) {
        if (wrow >= 0) {
            const uint8_t *row = q + wrow * qsplit_row_bytes<HB>(K);
            qc = *reinterpret_cast<const uint4 *>(row + k0 / 2);
            if constexpr (HB == 1) hc = *reinterpret_cast<const uint32_t *>(row + K / 2 + k0 / 8);
            if (k0 % G == 0) sb = scale[wrow * groups + k0 / G], mb = minv[wrow * groups + k0 / G];
        }
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads, r = seg / kSegs;
            xb[p][0] = xb[p][1] = xb[p][2] = xb[p][3] = 0;
            if (r >= rows) continue;
            const T *src = x + xoff[r] + k0 + (seg % kSegs) * 8;
            if constexpr (sizeof(T) == 2) {
                const uint4 u = *reinterpret_cast<const uint4 *>(src);
                xb[p][0] = u.x, xb[p][1] = u.y, xb[p][2] = u.z, xb[p][3] = u.w;
            } else {
                const float4 f0 = reinterpret_cast<const float4 *>(src)[0], f1 = reinterpret_cast<const float4 *>(src)[1];
                const float v[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    xb[p][i] = (uint32_t)wmma_f32_to_bf16(v[2 * i]) | ((uint32_t)wmma_f32_to_bf16(v[2 * i + 1]) << 16);
            }
        }
    };
    const auto stage = [&]() {  // x as F16 (WmmaXTile::stage_f16's conversion), row wrow's codes dequantized to F16
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads;
            uint32_t h[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                h[i] = XH && COMBINE ? xb[p][i] : bf16x2_to_f16x2(xb[p][i]);  // XH: x is F16 already
            *reinterpret_cast<uint4 *>(xs + (seg / kSegs) * LD + (seg % kSegs) * 8) = make_uint4(h[0], h[1], h[2], h[3]);
        }
        if constexpr (HB == 1)
            wmma_stage_codes_f16<5>(qc, wmma_bf16_bits_to_f16x2(sb), wmma_bf16_bits_to_f16x2(mb), wsm + threadIdx.x * LD, hc);
        else
            wmma_stage_codes_f16<4>(qc, wmma_bf16_bits_to_f16x2(sb), wmma_bf16_bits_to_f16x2(mb), wsm + threadIdx.x * LD);
    };
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;  // this wave's row tiles: wm, wm + 2, .., 2 (TMA - 1) + wm
        wmma_f32x8 acc[TMA][4] = {};
        fetch(0);
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
            stage();
            lds_barrier();
            if (k0 + KC < K) fetch(k0 + KC);  // the next step's loads in flight during this one's math
#pragma unroll
            for (int kk = 0; kk < KC; kk += 16) {
                wmma_f16x16 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 64 + 16 * j) * LD + kk, LD);
#pragma unroll
                for (int i = 0; i < TMA; ++i) {
                    const wmma_f16x16 a = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * LD + kk, LD);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = wmma_f16(a, b[j], acc[i][j]);
                }
            }
            lds_barrier();  // the next step's loads stay in flight
        }
        if constexpr (SWI && sizeof(T) == 2) {
            if (vec_out) {
                // h through LDS (the weight tile's bytes, free after the loop's last barrier) in two 64-row halves,
                // each value rounded as store() does, then written a 16-byte chunk (8 columns of one slot's row) a
                // store, where 2-byte stores, a half-wave's 32 bytes each, were the epilogue.
                constexpr int kOutLd = kCols + 8, kChunks = kCols / 8;
                static_assert(TR / 2 * kOutLd <= BW * LD && kOutLd * 2 % 16 == 0, "SWI: half the h tile fits in LDS");
                uint16_t *yb = reinterpret_cast<uint16_t *>(y);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
#pragma unroll
                    for (int i = 0; i < TMA; ++i) {
                        if (i / 2 != half) continue;  // row tile 2 i + wm lies in rows [64 half, 64 half + 64)
#pragma unroll
                        for (int j = 0; j < 2; ++j)
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                const int r = 16 * (2 * i + wm) + wmma_c_row(v) - 64 * half;
                                const int c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                                wsm[r * kOutLd + c] = h_bits<XH>(silu_f32(acc[i][j][v]) * acc[i][j + 2][v]);
                            }
                    }
                    __syncthreads();
                    for (int e = (int)threadIdx.x; e < TR / 2 * kChunks; e += kWmmaThreads) {
                        const int rl = e / kChunks, r = 64 * half + rl, c = e % kChunks * 8;
                        if (r >= rows || c >= rows_w) continue;  // rows_w % 8 == 0: whole chunks only
                        *reinterpret_cast<uint4 *>(yb + ooff[r] + n0 + c) =
                            *reinterpret_cast<const uint4 *>(wsm + rl * kOutLd + c);
                    }
                    __syncthreads();  // the next half overwrites the staged rows
                }
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < TMA; ++i)
#pragma unroll
            for (int j = 0; j < (SWI ? 2 : 4); ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int r = 16 * (2 * i + wm) + wmma_c_row(v);
                    const int c = wn * (SWI ? 32 : 64) + 16 * j + (int)threadIdx.x % 16;
                    if (r < rows && c < rows_w) {
                        if constexpr (SWI) store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][j][v]) * acc[i][j + 2][v]);
                        else if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][j][v], pbf16);
                        else store(y, ooff[r] + n0 + c, acc[i][j][v]);
                    }
                }
    };
    switch ((nt + 1) / 2) {  // uniform per block: the busier wave row's tiles
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// y[m, n] = sum_a coef[m, a] * partial[m * A + a, n], a in order, rounded once; NaN if any of token m's ids is
// invalid (checked here too, so the row is NaN whatever that slot's partial holds). V consecutive columns a thread
// (V = 8: one 16-byte load of BF16 partials a slot, N % 8 == 0); each column's sum is the V = 1 form's, so the
// outputs are bit-identical. One BF16 column a thread (2-byte loads) moved ~171 GB/s at the served shape.
template <int V> struct PartialLoad;
template <> struct PartialLoad<1> {
    __device__ static void run(const float *p, int64_t i, bool bf16, float (&v)[1]) { v[0] = get_partial(p, i, bf16); }
};
template <> struct PartialLoad<8> {
    __device__ static void run(const float *p, int64_t i, bool bf16, float (&v)[8]) {
        if (bf16) {
            const uint4 u = *reinterpret_cast<const uint4 *>(reinterpret_cast<const uint16_t *>(p) + i);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w[j] << 16), v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        } else {
            const float4 a = reinterpret_cast<const float4 *>(p + i)[0], b = reinterpret_cast<const float4 *>(p + i)[1];
            v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
        }
    }
};
template <typename T, int V> __device__ inline void store_v(T *y, int64_t i, const float (&v)[V]) {
    if constexpr (V == 1) {
        store(y, i, v[0]);
    } else if constexpr (sizeof(T) == 2) {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = (uint32_t)f32_to_bf16_bits(v[2 * j]) | ((uint32_t)f32_to_bf16_bits(v[2 * j + 1]) << 16);
        *reinterpret_cast<uint4 *>(y + i) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        reinterpret_cast<float4 *>(y + i)[0] = make_float4(v[0], v[1], v[2], v[3]);
        reinterpret_cast<float4 *>(y + i)[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
}
template <typename T, int V> __device__ inline void load_v(const T *p, int64_t i, float (&v)[V]) {
    if constexpr (V == 1) {
        v[0] = sizeof(T) == 2 ? bf16_bits_to_f32((uint16_t)p[i]) : (float)p[i];
    } else if constexpr (sizeof(T) == 2) {
        const uint4 u = *reinterpret_cast<const uint4 *>(p + i);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w[j] << 16), v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
    } else {
        const float4 a = reinterpret_cast<const float4 *>(p + i)[0], b = reinterpret_cast<const float4 *>(p + i)[1];
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    }
}
// The value v reads back as once stored in T (BF16: rounded once).
template <typename T> __device__ inline float as_stored(float v) {
    if constexpr (sizeof(T) == 2) return bf16_bits_to_f32(f32_to_bf16_bits(v));
    else return v;
}
// SH: the shared expert's gated add folded in (MoeSharedAdd): y = stored(sum) + g s, moe_shared_add's arithmetic.
template <typename T, int V, bool SH>
__global__ void __launch_bounds__(kThreads)
    combine_reduce_kernel(const float *partial, const int32_t *ids, const float *coef, int64_t M, int64_t A,
                          int64_t E, int64_t N, T *y, bool pbf16, const T *s, const float *gate_logit, int64_t gate_ld,
                          uint32_t gate_index, uint32_t *err) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (i >= M * N) return;
    const int64_t m = i / N, n = i % N;  // N % V == 0: the V columns lie in one row
    bool ok = true;
    float v[V] = {};
    for (int64_t a = 0; a < A; ++a) {
        const int32_t id = ids[m * A + a];
        ok = ok && id >= 0 && id < E;
        const float c = coef[m * A + a];
        float p[V];
        PartialLoad<V>::run(partial, (m * A + a) * N + n, pbf16, p);
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = fmaf(c, p[j], v[j]);
    }
    if (!ok)
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = __int_as_float(0x7fc00000);
    if constexpr (SH) {
        const float gl = gate_logit[m * gate_ld];
        const bool gok = isfinite(gl);
        if (!gok && n == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
            err[1] = (uint32_t)m;
            err[2] = gate_index;
        }
        const float g = gok ? 1.f / (1.f + expf(-gl)) : __builtin_nanf("");
        float sv[V];
        load_v<T, V>(s, i, sv);
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = as_stored<T>(v[j]) + g * sv[j];
    }
    store_v<T, V>(y, i, v);
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

bool is_f16(MoeMath m) {
    return m == MoeMath::WmmaF16 || m == MoeMath::WmmaF16Narrow || m == MoeMath::WmmaF16Wide ||
           m == MoeMath::WmmaF16Wide256;
}
const char *math_label(MoeMath m) {
    switch (m) {
        case MoeMath::F32: return "";
        case MoeMath::WmmaBf16: return " wmma";
        case MoeMath::WmmaF16: return " wmma-f16";
        case MoeMath::WmmaF16Narrow: return " wmma-f16-narrow";
        case MoeMath::WmmaF16Wide: return " wmma-f16-wide";
        case MoeMath::WmmaF16Wide256: return " wmma-f16-wide256";
    }
    return " ?";
}

// MoeMath::WmmaF16's choice: from these token counts on, the gather runs WmmaF16Wide256 and the combine's down
// WmmaF16Wide (its BF16 partials through LDS, 16-byte stores); below them both run WmmaF16Narrow. Measured on real routing (gfx1151, G128; the router ids of a 64k
// repository prompt replayed through the kernels), ms at M = 2048 / 4096 / 8192 / 16384: the SwiGLU gather narrow
// 6.06 / 9.79 / 17.33 / 32.76, wide 6.59 / 9.91 / 16.41 / 29.59, wide256 6.60 / 9.43 / 15.33 / 27.56; the down
// narrow 4.07 / 6.71 / 11.94 / 22.38, wide 4.50 / 7.07 / 12.01 / 21.77, wide256 4.28 / 6.88 / 12.00 / 22.26. Below
// these counts too many experts hold a fraction of a 128-slot tile (51% of 64-slot tile rows carry a slot at 2048, 90%
// at 16384); the down's K = 640 is 20 steps of 32 a block, too few for wide256's longer blocks. The down's epilogue
// through LDS then took wide to 6.65 / 11.26 / 20.41 at 4096 / 8192 / 16384 (4.27 at 2048), below narrow from 4096.
constexpr int64_t kWideMinMGather = 4096, kWideMinMCombine = 4096;

// XH: the F16 kernels' x (COMBINE) is F16 bits, or their SwiGLU h (SWI) is stored as F16 bits (h_bits).
template <typename T, bool COMBINE, int HB, bool SWI = false, bool XH = false>
void launch_grouped(const T *x, const Q4DeviceView &w, int64_t E, const void *ws, int64_t M, int64_t A, T *y,
                    float *partial, MoeMath math, hipStream_t stream) {
    const int64_t N = w.N / E;
    if (math == MoeMath::WmmaF16)
        math = COMBINE ? (M >= kWideMinMCombine ? MoeMath::WmmaF16Wide : MoeMath::WmmaF16Narrow)
                       : (M >= kWideMinMGather ? MoeMath::WmmaF16Wide256 : MoeMath::WmmaF16Narrow);
    STRIX_CHECK(!SWI || (is_f16(math) && N % 2 == 0), "moe grouped (SwiGLU): needs a MoeMath::WmmaF16 kind and an ",
                "even gate_up width, got math ", (int)math, ", N = ", N);
    STRIX_CHECK(!XH || is_f16(math), "moe grouped (F16 h): needs a MoeMath::WmmaF16 kind, got math ", (int)math);
    const auto *wsi = static_cast<const int32_t *>(ws);
    if (math == MoeMath::WmmaF16Wide) {
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA F16 wide): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16 wide): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).XTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16 wide): ", tiles, " tile slots exceed the grid");
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaBN / 2 : kWmmaBN;  // SWI: h columns
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_wide_kernel<T, 32, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_wide_kernel<T, 64, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_wide_kernel<T, 128, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    if (math == MoeMath::WmmaF16Wide256) {
        STRIX_CHECK(w.K % 32 == 0, "moe grouped (WMMA F16 wide256): K = ", w.K, ", expected a multiple of 32 (its K step)");
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16 wide256): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).XTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16 wide256): ", tiles, " tile slots exceed the grid");
        // a block: 256 staged weight rows - output columns, or SWI 128 h columns' gate and up rows
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaThreads / 2 : kWmmaThreads;
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_wide256_kernel<T, 32, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_wide256_kernel<T, 64, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_wide256_kernel<T, 128, COMBINE, SWI, HB, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    if (math == MoeMath::WmmaF16Narrow) {
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA F16): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).WTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16): ", tiles, " tile slots exceed the grid");
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaBN / 2 : kWmmaBN;  // SWI: h columns
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_kernel<T, 32, COMBINE, RW, HB, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_kernel<T, 64, COMBINE, RW, HB, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_kernel<T, 128, COMBINE, RW, HB, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
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
    // grid.y holds Tmax (the 65535 limit is on y and z, not x - the 65535*16 grid-stride idiom of the elementwise
    // kernels does not apply here). Reachable at the domain's edges:
    // Tmax = M*A/16 + E + 2 with M*A up to kMoeMaxSlots = 2^20, plus the expert count.
    STRIX_CHECK(Tmax <= 65535, "moe grouped (F32): ", Tmax, " tile slots exceed grid.y's 65535 (Tmax = ",
                "M*A/16 + E + 2 = ", M, "*", A, "/16 + ", E, " + 2)");
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
    moe_route_kernel<<<1, kRouteThreads, 0, stream>>>(ids, L.S, E, L.Tmax, L.WTmax, L.XTmax, static_cast<int32_t *>(ws),
                                                      err);
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
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math_label(math), " M=", M, " A=", A, " E=",
                           E, " N=", N, " K=", w.K, " G=", w.G);
}

template <int HB>
void experts_gather_grouped_swiglu(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                   size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                   hipStream_t stream, bool h_f16) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    check_common(fn, x, w, E, M, A, ws, ws_bytes, h, act);
    STRIX_CHECK(is_f16(math), fn, ": the fused SwiGLU runs on MoeMath::WmmaF16 (or its Narrow / Wide) only, got ",
                (int)math);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    STRIX_CHECK(N % 2 == 0, fn, ": gate_up has ", N, " rows an expert, expected an even count (gate rows then up rows)");
    check_no_overlap(fn, "h", h, M * A * (N / 2) * es, "x", x, M * w.K * es);
    STRIX_CHECK(!h_f16 || act == Act::BF16, fn, ": F16 h needs BF16 activations, got ", act_name(act));
    if (act == Act::F32)
        launch_grouped<float, false, HB, true>(static_cast<const float *>(x), w, E, ws, M, A, static_cast<float *>(h),
                                               nullptr, math, stream);
    else if (h_f16)
        launch_grouped<uint16_t, false, HB, true, true>(static_cast<const uint16_t *>(x), w, E, ws, M, A,
                                                        static_cast<uint16_t *>(h), nullptr, math, stream);
    else
        launch_grouped<uint16_t, false, HB, true>(static_cast<const uint16_t *>(x), w, E, ws, M, A,
                                                  static_cast<uint16_t *>(h), nullptr, math, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " wmma-f16", h_f16 ? " (F16 h)" : "", " M=", M, " A=", A, " E=", E,
                           " N=", N, " K=", w.K, " G=", w.G);
}

template <int HB>
void experts_combine_grouped(const char *fn, const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                             const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M, int64_t A,
                             void *y, Act act, MoeMath math, hipStream_t stream, const MoeSharedAdd *shared,
                             bool h_f16) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    STRIX_CHECK(!h_f16 || (act == Act::BF16 && is_f16(math)), fn, ": F16 h needs BF16 activations and a ",
                "MoeMath::WmmaF16 kind, got ", act_name(act), " math ", (int)math);
    check_common(fn, h, w, E, M, A, ws, ws_bytes, y, act);
    STRIX_CHECK(ids != nullptr && coef != nullptr && partial != nullptr, fn, ": null pointer (ids=",
                (const void *)ids, ", coef=", (const void *)coef, ", partial=", (const void *)partial, ")");
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    check_no_overlap(fn, "y", y, M * N * es, "h", h, M * A * w.K * es);
    check_no_overlap(fn, "y", y, M * N * es, "partial", partial, M * A * N * 4);
    check_no_overlap(fn, "partial", partial, M * A * N * 4, "h", h, M * A * w.K * es);
    if (shared != nullptr) {
        STRIX_CHECK(shared->s != nullptr && shared->gate_logit != nullptr && shared->err != nullptr, fn,
                    ": shared add with a null pointer (s=", shared->s, ", gate_logit=", (const void *)shared->gate_logit,
                    ", err=", (const void *)shared->err, ")");
        STRIX_CHECK(shared->gate_ld >= 1, fn, ": shared add gate_ld = ", shared->gate_ld, ", expected >= 1");
        check_no_overlap(fn, "y", y, M * N * es, "shared s", shared->s, M * N * es);
        check_no_overlap(fn, "partial", partial, M * A * N * 4, "shared s", shared->s, M * N * es);
    }
    const bool pbf16 = is_f16(math);  // the F16 kernels write BF16 partials (put_partial), the others FP32
    // 8 columns a thread when they are whole 16-byte runs (partial, y and s rows: N % 8 == 0, 16-byte aligned bases).
    const bool vec = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0 &&
                     (shared == nullptr || reinterpret_cast<uintptr_t>(shared->s) % 16 == 0);
    const unsigned blocks = (unsigned)((M * N / (vec ? 8 : 1) + kThreads - 1) / kThreads);
    auto reduce = [&](auto *yt) {
        using T = std::remove_pointer_t<decltype(yt)>;
        const T *s = shared ? static_cast<const T *>(shared->s) : nullptr;
        const float *gl = shared ? shared->gate_logit : nullptr;
        const int64_t gld = shared ? shared->gate_ld : 0;
        const uint32_t gi = shared ? shared->gate_index : 0;
        uint32_t *err = shared ? shared->err : nullptr;
        auto go = [&](auto v_c, auto sh_c) {
            combine_reduce_kernel<T, decltype(v_c)::value, decltype(sh_c)::value><<<blocks, kThreads, 0, stream>>>(
                partial, ids, coef, M, A, E, N, yt, pbf16, s, gl, gld, gi, err);
        };
        using V1 = std::integral_constant<int, 1>;
        using V8 = std::integral_constant<int, 8>;
        if (vec && shared) go(V8{}, std::true_type{});
        else if (vec) go(V8{}, std::false_type{});
        else if (shared) go(V1{}, std::true_type{});
        else go(V1{}, std::false_type{});
    };
    if (act == Act::F32) {
        launch_grouped<float, true, HB>(static_cast<const float *>(h), w, E, ws, M, A, nullptr, partial, math, stream);
        reduce(static_cast<float *>(y));
    } else {
        if (h_f16)
            launch_grouped<uint16_t, true, HB, false, true>(static_cast<const uint16_t *>(h), w, E, ws, M, A, nullptr,
                                                            partial, math, stream);
        else
            launch_grouped<uint16_t, true, HB>(static_cast<const uint16_t *>(h), w, E, ws, M, A, nullptr, partial, math,
                                               stream);
        reduce(static_cast<uint16_t *>(y));
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math_label(math), shared ? " +shared" : "", h_f16 ? " (F16 h)" : "",
                           " M=", M, " A=", A, " E=", E, " N=", N, " K=", w.K, " G=", w.G);
}

}  // namespace

void linear_q4_experts_gather_grouped(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream) {
    experts_gather_grouped<0>("linear_q4_experts_gather_grouped", x, w, E, ws, ws_bytes, M, A, y, act, math, stream);
}

void linear_q4_experts_gather_grouped_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream, bool h_f16) {
    experts_gather_grouped_swiglu<0>("linear_q4_experts_gather_grouped_swiglu", x, w, E, ws, ws_bytes, M, A, h, act,
                                     math, stream, h_f16);
}

void linear_q4_experts_combine_grouped(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                       const MoeSharedAdd *shared, bool h_f16) {
    experts_combine_grouped<0>("linear_q4_experts_combine_grouped", h, w, E, ids, coef, ws, ws_bytes, partial, M, A,
                               y, act, math, stream, shared, h_f16);
}

void linear_q5_experts_gather_grouped(const void *x, const Q5DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream) {
    experts_gather_grouped<1>("linear_q5_experts_gather_grouped", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ws,
                              ws_bytes, M, A, y, act, math, stream);
}

void linear_q5_experts_gather_grouped_swiglu(const void *x, const Q5DeviceView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream, bool h_f16) {
    experts_gather_grouped_swiglu<1>("linear_q5_experts_gather_grouped_swiglu", x, {w.q, w.scale, w.minv, w.N, w.K, w.G},
                                     E, ws, ws_bytes, M, A, h, act, math, stream, h_f16);
}

void linear_q5_experts_combine_grouped(const void *h, const Q5DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                       const MoeSharedAdd *shared, bool h_f16) {
    experts_combine_grouped<1>("linear_q5_experts_combine_grouped", h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, E, ids,
                               coef, ws, ws_bytes, partial, M, A, y, act, math, stream, shared, h_f16);
}

}  // namespace strix::kernels
