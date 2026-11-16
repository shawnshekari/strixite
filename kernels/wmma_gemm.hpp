#pragma once

// Device building blocks of the tiled GEMMs on the gfx11 matrix units (kernels/wmma.hpp): per K step of kWmmaKC, a
// weight tile and an activation tile (contiguous rows or gathered through an offset list) staged into LDS as BF16,
// software-pipelined (fetch the next step into registers during this step's WMMA work), then each wave's 16x16 WMMA
// blocks. Q4 / Q8 weights as scaled codes (linear_wmma, moe_grouped, hc_wmma): the tile holds the raw codes q
// (0..15 / 0..255, exact in BF16) and each group's scale and min go to the FP32 accumulator instead, one group per
// sub-step of wmma_sub(G) = min(G, kWmmaKC) along K: acc += s * sum_k bf16(x_k) q_k + m * sum_k bf16(x_k). So x is
// the only rounded input (a no-op for BF16 activations); the weights enter exactly (as formats/q4 / q8 dequantize,
// up to FP32 rounding), and staging costs ~2 ops per code instead of a dequantize + round (measured: dequant bounded
// these kernels). BF16 weights (the router) go to LDS unchanged. FP32
// accumulation either way (a lower-precision path). Included by .hip files
// only.

#include "kernels/qsplit.hpp"  // q5_bit4_spread
#include "kernels/wmma.hpp"

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

constexpr int kWmmaKC = 64;             // K per step (a multiple of every group size's 32-code chunk)
constexpr int kWmmaLd = kWmmaKC + 8;    // LDS row stride in BF16 elements (padding against bank conflicts)
static_assert(kWmmaLd * 2 % 16 == 0, "rows 16-byte aligned: fragments load with 128-bit reads (wmma_load_rows_a16)");
constexpr int kWmmaThreads = 256;       // 8 waves of 32

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)

__device__ inline uint16_t wmma_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// One K step of a wave's TM x TN block of 16x16 tiles: rows [wr0, wr0 + 16 TM) of the x tile times rows
// [wc0, wc0 + 16 TN) of the weight tile.
// F16 path: one K step (kWmmaKC) of a wave's TM x TN tiles from F16 LDS tiles (x rows, dequantized weight rows),
// accumulating in FP32 - no per-group work.
template <int TM, int TN>
__device__ inline void wmma_step_f16(const uint16_t *xs, const uint16_t *ws, int wr0, int wc0, wmma_f32x8 (&acc)[TM][TN]) {
#pragma unroll
    for (int kk = 0; kk < kWmmaKC; kk += 16) {
        wmma_f16x16 a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) a[i] = wmma_load_rows_a16_f16(xs + (wr0 + 16 * i) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
        for (int j = 0; j < TN; ++j) b[j] = wmma_load_rows_a16_f16(ws + (wc0 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = wmma_f16(a[i], b[j], acc[i][j]);
    }
}

template <int TM, int TN>
__device__ inline void wmma_step(const uint16_t *xs, const uint16_t *ws, int wr0, int wc0,
                                 wmma_f32x8 (&acc)[TM][TN]) {
#pragma unroll
    for (int kk = 0; kk < kWmmaKC; kk += 16) {
        wmma_bf16x16 a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) a[i] = wmma_load_rows_a16(xs + (wr0 + 16 * i) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
        for (int j = 0; j < TN; ++j) b[j] = wmma_load_rows_a16(ws + (wc0 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = wmma_bf16(a[i], b[j], acc[i][j]);
    }
}

// ---- Scaled codes ----

// K per scale group within a step: a step holds kWmmaKC / wmma_sub(G) groups (G 32: 2; 64, 128: 1).
template <int G>
constexpr int wmma_sub() {
    static_assert(G == 32 || G == 64 || G == 128, "group size 32, 64 or 128");
    return G < kWmmaKC ? G : kWmmaKC;
}

// Two byte values (0..255) as a pair of BF16 bits, a in the low half: exact (at most 8 significant bits), one
// v_cvt_f32_ubyte each.
__device__ inline uint32_t wmma_pack_codes(uint32_t a, uint32_t b) {
    return (__float_as_uint((float)a) >> 16) | (__float_as_uint((float)b) & 0xffff0000u);
}

// One 16-byte load of codes (BITS 4: 32 codes, 8: 16) into LDS at dst (16-byte aligned) as BF16 bits, in K order.
template <int BITS>
__device__ inline void wmma_stage_codes(const uint4 &q, uint16_t *dst) {
    constexpr int kCodes = BITS == 4 ? 32 : 16;
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    uint32_t d[kCodes / 2];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (BITS == 4) {  // codes 8w + 2p, 8w + 2p + 1 = byte p's low, high nibble
            const uint32_t lo = words[w] & 0x0f0f0f0fu, hi = (words[w] >> 4) & 0x0f0f0f0fu;
#pragma unroll
            for (int p = 0; p < 4; ++p) d[4 * w + p] = wmma_pack_codes((lo >> (8 * p)) & 0xffu, (hi >> (8 * p)) & 0xffu);
        } else {
            d[2 * w] = wmma_pack_codes(words[w] & 0xffu, (words[w] >> 8) & 0xffu);
            d[2 * w + 1] = wmma_pack_codes((words[w] >> 16) & 0xffu, words[w] >> 24);
        }
    }
    uint4 *out = reinterpret_cast<uint4 *>(dst);
#pragma unroll
    for (int j = 0; j < kCodes / 8; ++j) out[j] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
}

// ---- F16: weights dequantized to F16 (code * scale + min, one rounding), x as
// F16, the matrix units accumulating the whole K extent - no per-group accumulator work.
typedef _Float16 wmma_f16x2 __attribute__((ext_vector_type(2)));

// A BF16 scale / min as an F16 pair (exact for the normal-range values these hold).
__device__ inline wmma_f16x2 wmma_bf16_bits_to_f16x2(uint16_t b) {
    const _Float16 h = (_Float16)__uint_as_float((uint32_t)b << 16);
    return wmma_f16x2{h, h};
}

// Codes c0 (low half) and c1 as F16 pairs code * s + m: a byte code placed in the low mantissa bits of 1024 (0x6400,
// where the F16 unit is 1 up to 2047) is exactly 1024 + code; subtracting 1024 is exact; one F16 FMA rounds once.
// sel: the v_perm_b32 selector putting code bytes into bytes 0 and 2 of the result (0x0c = a zero byte).
__device__ inline uint32_t wmma_codes_to_f16x2(uint32_t hi_src, uint32_t lo_src, uint32_t sel, wmma_f16x2 s, wmma_f16x2 m) {
    const wmma_f16x2 k1024 = {(_Float16)1024.0f, (_Float16)1024.0f};
    const wmma_f16x2 c = __builtin_bit_cast(wmma_f16x2, __builtin_amdgcn_perm(hi_src, lo_src, sel) | 0x64006400u) - k1024;
    return __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(c, s, m));
}

// Codes 2j (byte j of lo) and 2j + 1 (byte j of hi), j = 0..3, as 4 F16 pairs into out[0..3].
__device__ inline void wmma_code_bytes_to_f16(uint32_t lo, uint32_t hi, wmma_f16x2 s, wmma_f16x2 m, uint32_t *out) {
#pragma unroll
    for (int j = 0; j < 4; ++j)  // perm(hi, lo): bytes 0-3 lo's, 4-7 hi's -> [lo_j, 0, hi_j, 0]
        out[j] = wmma_codes_to_f16x2(hi, lo, (uint32_t)j | 0x0c00u | ((uint32_t)(4 + j) << 16) | 0x0c000000u, s, m);
}

// The 8 Q4 codes of one word (code i at bits 4i) as 4 F16 pairs (codes 2j, 2j+1) into out[0..3].
__device__ inline void wmma_q4_word_to_f16(uint32_t w, wmma_f16x2 s, wmma_f16x2 m, uint32_t *out) {
    wmma_code_bytes_to_f16(w & 0x0F0F0F0Fu, (w >> 4) & 0x0F0F0F0Fu, s, m, out);  // even / odd codes, one per byte
}

// The 8 Q5 codes of one word: low nibbles w as Q4's (code i at bits 4i), 5th bits h8 (code i's at bit i; bits 8+
// ignored), as 4 F16 pairs (codes 2j, 2j+1) into out[0..3]. Codes 0..31 are exact in F16; one rounding, as Q4.
__device__ inline void wmma_q5_word_to_f16(uint32_t w, uint32_t h8, wmma_f16x2 s, wmma_f16x2 m, uint32_t *out) {
    wmma_code_bytes_to_f16((w & 0x0F0F0F0Fu) | q5_bit4_spread(h8, 0),
                           ((w >> 4) & 0x0F0F0F0Fu) | q5_bit4_spread(h8, 1), s, m, out);
}

// The 4 Q8 codes of one word (code i in byte i) as 2 F16 pairs (codes 0,1 then 2,3) into out[0..1].
__device__ inline void wmma_q8_word_to_f16(uint32_t w, wmma_f16x2 s, wmma_f16x2 m, uint32_t *out) {
    out[0] = wmma_codes_to_f16x2(w, w, 0x0c010c00u, s, m);  // [b0, 0, b1, 0]
    out[1] = wmma_codes_to_f16x2(w, w, 0x0c030c02u, s, m);  // [b2, 0, b3, 0]
}

// One 16-byte load of codes (BITS 4: 32 codes, 8: 16 - all in one group, scale sc / min mn as F16 pairs) dequantized
// to F16 and written to LDS at dst (16-byte aligned) in K order. BITS 5: q the 32 codes' low nibbles (Q4's order),
// hi their 5th bits (code i's at bit i, formats/q5.hpp's high plane).
template <int BITS>
__device__ inline void wmma_stage_codes_f16(const uint4 &q, wmma_f16x2 sc, wmma_f16x2 mn, uint16_t *dst,
                                            uint32_t hi = 0) {
    static_assert(BITS == 4 || BITS == 5 || BITS == 8, "F16 staging: Q4 / Q5 / Q8 codes");
    constexpr int kCodes = BITS == 8 ? 16 : 32;
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    uint32_t d[kCodes / 2];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (BITS == 4) wmma_q4_word_to_f16(words[w], sc, mn, d + 4 * w);
        else if constexpr (BITS == 5) wmma_q5_word_to_f16(words[w], hi >> (8 * w), sc, mn, d + 4 * w);
        else wmma_q8_word_to_f16(words[w], sc, mn, d + 2 * w);
    }
    uint4 *out = reinterpret_cast<uint4 *>(dst);
#pragma unroll
    for (int j = 0; j < kCodes / 8; ++j) out[j] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
}

// Weight tile of raw codes, in two halves so the next K step's global loads can be in flight during this step's
// WMMA work: fetch() loads this thread's share of BN rows (row_base + i, i < rows_valid) x kWmmaKC codes from k0
// into registers; stage() writes them to LDS [BN][kWmmaLd] as BF16 bits. The rows' scales and mins come in windows
// of kWin groups (a step that opens a window fetches and stages it into sw / mw [BN][kWin], BF16 bits; groups past
// the row's end and rows past rows_valid zero): coalesced - consecutive lanes read consecutive groups - where one
// scattered 2-byte load per row per step cost the experts ~2x.
// BITS 4: a 16-byte load is 32 codes; 8: 16 codes (6 / 5: the specializations below).
template <int BITS, int G, int BN>
struct WmmaWCodes {
    static constexpr int kCodesPerLoad = BITS == 4 ? 32 : 16, kLoadsPerRow = kWmmaKC / kCodesPerLoad;
    static constexpr int kLoads = BN * kLoadsPerRow / kWmmaThreads;
    static constexpr int kWin = 8, kWinLoads = BN * kWin / kWmmaThreads;
    static_assert(BN * kLoadsPerRow % kWmmaThreads == 0 && BN * kWin % kWmmaThreads == 0, "whole loads per thread");
    uint4 q[kLoads];
    uint16_t s[kWinLoads], m[kWinLoads];
    bool win = false;

    // Whether the step at k0 (from k_base) opens a scale window: it starts a group whose index (from k_base) is a
    // multiple of kWin.
    __device__ static bool opens_window(int64_t k0) { return k0 % G == 0 && (k0 / G) % kWin == 0; }

    // k0: the step's K offset from k_base (a multiple of G) - the GEMM's K range starts there (hc_wmma: a stream's).
    __device__ void fetch(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, int64_t row_base,
                          int rows_valid, int64_t K, int64_t k0, int64_t k_base = 0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int64_t k = k_base + k0 + (l % kLoadsPerRow) * kCodesPerLoad;
            q[i] = r < rows_valid ? *reinterpret_cast<const uint4 *>(codes + (row_base + r) * (K * BITS / 8) + k * BITS / 8)
                                  : make_uint4(0, 0, 0, 0);
        }
        win = opens_window(k0);
        if (win) {
            const int64_t groups = K / G, g0 = (k_base + k0) / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, r = e / kWin;
                const int64_t g = g0 + e % kWin, at = (row_base + r) * groups + g;
                const bool ok = r < rows_valid && g < groups;
                s[i] = ok ? scale[at] : (uint16_t)0, m[i] = ok ? minv[at] : (uint16_t)0;
            }
        }
    }
    // fetch() with staged row r taken from global row row_of(r) (< 0: a zero row) - weight rows that aren't one
    // contiguous range (the experts' stacked matrices, SwiGLU's gate and up halves). K, k0: as fetch() with k_base 0.
    template <typename RowFn>
    __device__ void fetch_mapped(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, RowFn row_of,
                                 int64_t K, int64_t k0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int64_t k = k0 + (l % kLoadsPerRow) * kCodesPerLoad, row = row_of(r);
            q[i] = row >= 0 ? *reinterpret_cast<const uint4 *>(codes + row * (K * BITS / 8) + k * BITS / 8)
                            : make_uint4(0, 0, 0, 0);
        }
        win = opens_window(k0);
        if (win) {
            const int64_t groups = K / G, g0 = k0 / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, r = e / kWin;
                const int64_t g = g0 + e % kWin, row = row_of(r);
                const bool ok = row >= 0 && g < groups;
                s[i] = ok ? scale[row * groups + g] : (uint16_t)0, m[i] = ok ? minv[row * groups + g] : (uint16_t)0;
            }
        }
    }
    __device__ void stage(uint16_t *lds, uint16_t *sw, uint16_t *mw) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int kin = (l % kLoadsPerRow) * kCodesPerLoad;
            wmma_stage_codes<BITS>(q[i], lds + r * kWmmaLd + kin);
        }
        stage_window(sw, mw);
    }
    // The scale window alone (when this step opens one) - the F16 path writes it, synchronizes, then decodes.
    __device__ void stage_window(uint16_t *sw, uint16_t *mw) const {
        if (win)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                sw[e] = s[i], mw[e] = m[i];  // [r][e % kWin] with r = e / kWin
            }
    }
    // F16 path: this thread's codes dequantized with their groups' scale / min from the staged window (the step at
    // k0 from k_base), written to LDS [BN][kWmmaLd] as F16 bits. BITS 4 or 8.
    __device__ void stage_f16(uint16_t *lds, const uint16_t *sw, const uint16_t *mw, int64_t k0) const {
        static_assert(BITS == 4 || BITS == 8, "F16 staging: Q4 / Q8 codes");
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int kin = (l % kLoadsPerRow) * kCodesPerLoad;
            // a 16-byte load's codes lie in one group (32 codes at Q4, 16 at Q8; G >= 32)
            const int slot = (int)(((k0 + kin) / G) % kWin);
            wmma_stage_codes_f16<BITS>(q[i], wmma_bf16_bits_to_f16x2(sw[r * kWin + slot]),
                                       wmma_bf16_bits_to_f16x2(mw[r * kWin + slot]), lds + r * kWmmaLd + kin);
        }
    }
};

// Q6 / Q5 (formats/q6.hpp, q5.hpp: per row a low plane [K/2] - Q4's nibbles - then a high plane [K * HB / 8], HB = 2
// or 1 bits a code): the same interface, one unit of 32 codes = 16 bytes of low nibbles + 4 * HB bytes of high bits.
template <int HB, int G, int BN>
struct WmmaWCodesSplit {
    static constexpr int kCodesPerLoad = 32, kLoadsPerRow = kWmmaKC / kCodesPerLoad;
    static constexpr int kLoads = BN * kLoadsPerRow / kWmmaThreads;
    static constexpr int kWin = 8, kWinLoads = BN * kWin / kWmmaThreads;
    static_assert(BN * kLoadsPerRow % kWmmaThreads == 0 && BN * kWin % kWmmaThreads == 0, "whole loads per thread");
    uint4 lo[kLoads];
    uint32_t hi[kLoads][HB];
    uint16_t s[kWinLoads], m[kWinLoads];
    bool win = false;

    __device__ static bool opens_window(int64_t k0) { return k0 % G == 0 && (k0 / G) % kWin == 0; }

    __device__ void fetch(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, int64_t row_base,
                          int rows_valid, int64_t K, int64_t k0, int64_t k_base = 0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int64_t k = k_base + k0 + (l % kLoadsPerRow) * kCodesPerLoad;
            const uint8_t *row = codes + (row_base + r) * (K / 2 + K * HB / 8);
            lo[i] = r < rows_valid ? *reinterpret_cast<const uint4 *>(row + k / 2) : make_uint4(0, 0, 0, 0);
            if constexpr (HB == 2) {
                const uint2 v = r < rows_valid ? *reinterpret_cast<const uint2 *>(row + K / 2 + k / 4) : make_uint2(0, 0);
                hi[i][0] = v.x, hi[i][1] = v.y;
            } else {
                hi[i][0] = r < rows_valid ? *reinterpret_cast<const uint32_t *>(row + K / 2 + k / 8) : 0u;
            }
        }
        win = opens_window(k0);
        if (win) {
            const int64_t groups = K / G, g0 = (k_base + k0) / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, r = e / kWin;
                const int64_t g = g0 + e % kWin, at = (row_base + r) * groups + g;
                const bool ok = r < rows_valid && g < groups;
                s[i] = ok ? scale[at] : (uint16_t)0, m[i] = ok ? minv[at] : (uint16_t)0;
            }
        }
    }
    __device__ void stage(uint16_t *lds, uint16_t *sw, uint16_t *mw) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int kin = (l % kLoadsPerRow) * kCodesPerLoad;
            const uint32_t lw[4] = {lo[i].x, lo[i].y, lo[i].z, lo[i].w};
            auto high = [&](int c) -> uint32_t {
                if constexpr (HB == 2) return (hi[i][c / 16] >> (2 * (c % 16))) & 0x3u;
                else return (hi[i][0] >> c) & 0x1u;
            };
            uint32_t d[16];
#pragma unroll
            for (int c = 0; c < 32; c += 2) {
                const uint32_t a = ((lw[c / 8] >> (4 * (c % 8))) & 0xFu) | (high(c) << 4);
                const uint32_t b = ((lw[(c + 1) / 8] >> (4 * ((c + 1) % 8))) & 0xFu) | (high(c + 1) << 4);
                d[c / 2] = wmma_pack_codes(a, b);
            }
            uint4 *out = reinterpret_cast<uint4 *>(lds + r * kWmmaLd + kin);
#pragma unroll
            for (int j = 0; j < 4; ++j) out[j] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
        }
        stage_window(sw, mw);
    }
    // WmmaWCodes::fetch_mapped's interface: staged row r from global row row_of(r) (< 0: a zero row).
    template <typename RowFn>
    __device__ void fetch_mapped(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, RowFn row_of,
                                 int64_t K, int64_t k0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int64_t k = k0 + (l % kLoadsPerRow) * kCodesPerLoad, row = row_of(r);
            const uint8_t *rp = codes + (row >= 0 ? row : 0) * (K / 2 + K * HB / 8);
            lo[i] = row >= 0 ? *reinterpret_cast<const uint4 *>(rp + k / 2) : make_uint4(0, 0, 0, 0);
            if constexpr (HB == 2) {
                const uint2 v = row >= 0 ? *reinterpret_cast<const uint2 *>(rp + K / 2 + k / 4) : make_uint2(0, 0);
                hi[i][0] = v.x, hi[i][1] = v.y;
            } else {
                hi[i][0] = row >= 0 ? *reinterpret_cast<const uint32_t *>(rp + K / 2 + k / 8) : 0u;
            }
        }
        win = opens_window(k0);
        if (win) {
            const int64_t groups = K / G, g0 = k0 / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, r = e / kWin;
                const int64_t g = g0 + e % kWin, row = row_of(r);
                const bool ok = row >= 0 && g < groups;
                s[i] = ok ? scale[row * groups + g] : (uint16_t)0, m[i] = ok ? minv[row * groups + g] : (uint16_t)0;
            }
        }
    }
    __device__ void stage_window(uint16_t *sw, uint16_t *mw) const {
        if (win)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                sw[e] = s[i], mw[e] = m[i];  // [r][e % kWin] with r = e / kWin
            }
    }
    // WmmaWCodes::stage_f16 for Q5 (HB 1): the codes dequantized with their group's scale / min from the staged window.
    __device__ void stage_f16(uint16_t *lds, const uint16_t *sw, const uint16_t *mw, int64_t k0) const {
        static_assert(HB == 1, "F16 staging: Q5 split codes");
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int kin = (l % kLoadsPerRow) * kCodesPerLoad;
            const int slot = (int)(((k0 + kin) / G) % kWin);  // a load's 32 codes lie in one group (G >= 32)
            wmma_stage_codes_f16<5>(lo[i], wmma_bf16_bits_to_f16x2(sw[r * kWin + slot]),
                                    wmma_bf16_bits_to_f16x2(mw[r * kWin + slot]), lds + r * kWmmaLd + kin, hi[i][0]);
        }
    }
};

template <int G, int BN>
struct WmmaWCodes<6, G, BN> : WmmaWCodesSplit<2, G, BN> {};
template <int G, int BN>
struct WmmaWCodes<5, G, BN> : WmmaWCodesSplit<1, G, BN> {};

// BF16 weight tile (a BF16 weight matrix [N, K], e.g. the router), fetch / stage as WmmaWCodes: BN rows (row_base +
// i, i < rows_valid; the rest zero) x kWmmaKC from k0 into LDS [BN][kWmmaLd] unchanged.
template <int BN>
struct WmmaWBf16 {
    static constexpr int kSegs = kWmmaKC / 8, kLoads = BN * kSegs / kWmmaThreads;
    static_assert(BN * kSegs % kWmmaThreads == 0, "whole loads per thread");
    uint4 v[kLoads];

    __device__ void fetch(const uint16_t *w, int64_t row_base, int rows_valid, int64_t K, int64_t k0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kSegs;
            v[i] = r < rows_valid ? *reinterpret_cast<const uint4 *>(w + (row_base + r) * K + k0 + (l % kSegs) * 8)
                                  : make_uint4(0, 0, 0, 0);
        }
    }
    __device__ void stage(uint16_t *lds) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads;
            *reinterpret_cast<uint4 *>(lds + (l / kSegs) * kWmmaLd + (l % kSegs) * 8) = v[i];
        }
    }
};

// Activation tile, fetch / stage as WmmaWCodes: BM rows x kWmmaKC from k0 (row i at x + xoff[i], the element offset
// of its start; rows past rows_valid zero) into LDS [BM][kWmmaLd] as BF16 bits, plus each row's sum of its staged
// (BF16-rounded) values per sub-step into rs [BM][kWmmaKC / wmma_sub(G)] (FP32, a fixed shuffle tree -
// deterministic; SUMS false: rs untouched, may be null). T: float (rounded to BF16 at fetch) or uint16_t (BF16 bits).
template <typename T, int BM, int G, bool SUMS = true>
struct WmmaXTile {
    static constexpr int kSegs = kWmmaKC / 8, kSub = wmma_sub<G>(), kNsub = kWmmaKC / kSub, kLanes = kSub / 8;
    static constexpr int kPasses = (BM * kSegs + kWmmaThreads - 1) / kWmmaThreads;
    static_assert((BM * kSegs) % 32 == 0, "whole waves per pass: the row sums shuffle across a segment's lanes");
    uint32_t b[kPasses][4];  // 8 BF16 values per pass, pairs little-endian

    __device__ void fetch(const T *x, const int64_t *xoff, int rows_valid, int64_t k0) {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads, r = seg / kSegs;
            b[p][0] = b[p][1] = b[p][2] = b[p][3] = 0;
            if (seg >= BM * kSegs || r >= rows_valid) continue;
            const T *src = x + xoff[r] + k0 + (seg % kSegs) * 8;
            if constexpr (sizeof(T) == 2) {
                const uint4 u = *reinterpret_cast<const uint4 *>(src);
                b[p][0] = u.x, b[p][1] = u.y, b[p][2] = u.z, b[p][3] = u.w;
            } else {
                const float4 f0 = reinterpret_cast<const float4 *>(src)[0], f1 = reinterpret_cast<const float4 *>(src)[1];
                const float v[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    b[p][i] = (uint32_t)wmma_f32_to_bf16(v[2 * i]) | ((uint32_t)wmma_f32_to_bf16(v[2 * i + 1]) << 16);
            }
        }
    }
    // F16 path: the fetched (BF16) values to LDS as F16 bits - exact in F16's normal range (the activations' measured
    // |x| <= 180), round-toward-zero only below it.
    __device__ void stage_f16(uint16_t *lds) const {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads;
            if (seg >= BM * kSegs) break;  // wave-uniform
            uint32_t h[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                h[i] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(b[p][i] << 16),
                                                                               __uint_as_float(b[p][i] & 0xffff0000u)));
            *reinterpret_cast<uint4 *>(lds + (seg / kSegs) * kWmmaLd + (seg % kSegs) * 8) = make_uint4(h[0], h[1], h[2], h[3]);
        }
    }
    __device__ void stage(uint16_t *lds, float *rs) const {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads, r = seg / kSegs;
            if (seg >= BM * kSegs) break;  // wave-uniform (whole waves per pass)
            *reinterpret_cast<uint4 *>(lds + r * kWmmaLd + (seg % kSegs) * 8) = make_uint4(b[p][0], b[p][1], b[p][2], b[p][3]);
            if constexpr (!SUMS) continue;
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) sum += __uint_as_float(b[p][i] << 16) + __uint_as_float(b[p][i] & 0xffff0000u);
#pragma unroll
            for (int off = 1; off < kLanes; off <<= 1) sum += __shfl_xor(sum, off, 32);
            if ((seg % kSegs) % kLanes == 0) rs[r * kNsub + (seg % kSegs) / kLanes] = sum;
        }
    }
};

// F16 activation tile from FP32 values (e.g. the HC mix's h): fetch / stage as WmmaXTile, each value rounded once to
// F16 (round to nearest even) at fetch - not through BF16 - into LDS [BM][kWmmaLd] as F16 bits.
template <int BM>
struct WmmaXTileF32ToF16 {
    static constexpr int kSegs = kWmmaKC / 8, kPasses = (BM * kSegs + kWmmaThreads - 1) / kWmmaThreads;
    uint32_t b[kPasses][4];  // 8 F16 values per pass, pairs little-endian

    __device__ void fetch(const float *x, const int64_t *xoff, int rows_valid, int64_t k0) {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads, r = seg / kSegs;
            b[p][0] = b[p][1] = b[p][2] = b[p][3] = 0;
            if (seg >= BM * kSegs || r >= rows_valid) continue;
            const float *src = x + xoff[r] + k0 + (seg % kSegs) * 8;
            const float4 f0 = reinterpret_cast<const float4 *>(src)[0], f1 = reinterpret_cast<const float4 *>(src)[1];
            const float v[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
                b[p][i] = __builtin_bit_cast(uint32_t, wmma_f16x2{(_Float16)v[2 * i], (_Float16)v[2 * i + 1]});
        }
    }
    __device__ void stage_f16(uint16_t *lds) const {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads;
            if (seg >= BM * kSegs) break;  // wave-uniform
            *reinterpret_cast<uint4 *>(lds + (seg / kSegs) * kWmmaLd + (seg % kSegs) * 8) =
                make_uint4(b[p][0], b[p][1], b[p][2], b[p][3]);
        }
    }
};

// One K step (at k0) of a wave's TM x TN block of 16x16 tiles over scaled codes: per sub-step, the code products p
// (fresh FP32 tiles), then acc += s[col] * p + m[col] * rs[row], s and m from the current scale window. Only the
// first tm_active of the TM row tiles run (uniform per block: tiles past the rows holding data are skipped).
template <int TM, int TN, int G>
__device__ inline void wmma_step_scaled(const uint16_t *xs, const uint16_t *ws, const float *rs, const uint16_t *sw,
                                        const uint16_t *mw, int64_t k0, int wr0, int wc0, wmma_f32x8 (&acc)[TM][TN],
                                        int tm_active = TM) {
    constexpr int kWin = 8;  // WmmaWCodes::kWin
    constexpr int SUB = wmma_sub<G>(), NSUB = kWmmaKC / SUB;
#pragma unroll
    for (int sub = 0; sub < NSUB; ++sub) {
        wmma_f32x8 p[TM][TN] = {};
#pragma unroll
        for (int kk = sub * SUB; kk < (sub + 1) * SUB; kk += 16) {
            wmma_bf16x16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                if (i < tm_active) a[i] = wmma_load_rows_a16(xs + (wr0 + 16 * i) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = wmma_load_rows_a16(ws + (wc0 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
            for (int i = 0; i < TM; ++i)
                if (i < tm_active)
#pragma unroll
                    for (int j = 0; j < TN; ++j) p[i][j] = wmma_bf16(a[i], b[j], p[i][j]);
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = wc0 + 16 * j + (int)threadIdx.x % 16;
            const int slot = (int)(((k0 + sub * SUB) / G) % kWin);
            const float s = __uint_as_float((uint32_t)sw[n * kWin + slot] << 16),
                        m = __uint_as_float((uint32_t)mw[n * kWin + slot] << 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
                if (i < tm_active)
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        acc[i][j][v] = fmaf(s, p[i][j][v], fmaf(m, rs[(wr0 + 16 * i + wmma_c_row(v)) * NSUB + sub], acc[i][j][v]));
        }
    }
}

#endif

}  // namespace strix::kernels
