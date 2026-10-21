#pragma once

// 8-bit K/V cache for the full-attention layers.
//
// Per layer, W = Hk * D values a row (K and V alike), rows = sequence positions:
//   K: asymmetric 8 bits per channel over groups of kKv8Group rows - for group g and channel c one (scale, zero)
//      pair from the min / max of the group's 32 values of that channel (keys after norm + RoPE carry per-channel
//      offsets, which a per-channel range fits; Phase 0: K error 42% below symmetric blocks of 32).
//   V: asymmetric 8 bits per block of kKv8Block consecutive channels of one row.
// The (scale, zero) pair: scale = fp16((max - min) / 255), zero = fp16(min), packed in one word (scale in the low
// 16 bits); code = clamp(rint((x - zero) / scale), 0, 255); value = fma(code, scale, zero) in FP32, rounded to the
// activation dtype where it is read. A range of zero width (or one whose scale rounds to 0 in fp16) stores scale 0,
// codes 0: every value reads back as fp16(min). ~56% of BF16 (1 byte + 4 bytes / 32 a value).
//
// Rows of the newest, incomplete K group can't be quantized yet (their channel ranges aren't known), so the rows
// [g0, pos) with g0 = kKv8Group * (pos / kKv8Group) - K and V both - live in a tail of kKv8Group rows in the
// activation dtype: [2][kKv8Group][W], K rows first. Like the QSA indexer tail it is double-buffered: a call reads
// tail_in and writes tail_out (a verify that is dropped leaves tail_in as it was).
//
// What a query sees depends only on its own position p: rows of the groups complete at p - below
// g0(p) = kKv8Group * ((p + 1) / kKv8Group) - in 8 bits, the rows from g0(p) on exact. Exactly what a token-by-token
// decode sees, so a prompt in one call, in chunks or token by token computes the same values (up to float noise).
// Inside one call the exact rows come from the tail before the call (rows [g0(pos0), pos0)) and the call's own new
// rows (from pos0).
//
// Storage of one layer (kv8_bytes), capacity a multiple of kKv8Group:
//   [0, cap W)                      K codes, row r at r W (row-major)
//   [cap W, + cap / 32 W words)     K params, group g channel c at g W + c
//   then V codes [cap][W], then V params [cap][W / 32] words.

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::kernels {

constexpr int64_t kKv8Group = 32;  // K rows sharing one (scale, zero) per channel
constexpr int64_t kKv8Block = 32;  // V channels of one row sharing one (scale, zero)
// The first row a query at position p sees exact (host side; kv8_math::exact_from is the device twin).
constexpr int64_t kv8_exact_from(int64_t p) { return (p + 1) / kKv8Group * kKv8Group; }

// Bytes of one layer's 8-bit storage (no tail) for capacity rows of W values.
size_t kv8_bytes(int64_t capacity, int64_t W);
// Bytes of one tail buffer: [2][kKv8Group][W] in the activation dtype.
size_t kv8_tail_bytes(int64_t W, Act act);

// What an attention kernel needs to read one layer's 8-bit K/V for a call at positions pos0 .. pos0 + T - 1, after
// kv8_commit of the call's rows: storage (kv8_bytes layout), tail_in (the tail before the call: rows
// [g0(pos0), pos0); unused, may be null, when pos0 is a multiple of kKv8Group) and the call's new rows (row t's K at
// k_new[t * new_ts ..], V at v_new[t * new_ts ..] - the ones kv8_commit read).
// Optional staging (prefill): staged_k / staged_v hold rows [0, staged_rows) already dequantized (kv8_dequant, the
// activation dtype, rows W apart, 16-byte aligned); staged_rows must reach the last query's exact_from
// (kv8_math::exact_from(pos0 + T - 1)). The readers then load those rows instead of unpacking codes per use - the same
// values, so the same result. Unpacking inside the WMMA prefill attention loop repeats per query: it made that kernel
// 2.7x slower than over the BF16 cache.
struct Kv8View {
    const uint8_t *store = nullptr;
    int64_t capacity = 0, W = 0;
    const void *tail_in = nullptr;
    const void *k_new = nullptr, *v_new = nullptr;
    int64_t new_ts = 0, pos0 = 0;
    const void *staged_k = nullptr, *staged_v = nullptr;
    int64_t staged_rows = 0;
    bool hadamard = false;
};

// Append T rows at positions pos0 .. pos0 + T - 1: row t's K at k_new[t * new_ts ..] and V at v_new[t * new_ts ..]
// (W values each, activation dtype). tail_in holds rows [g0(pos0), pos0) (unused, may be null, when pos0 is a
// multiple of kKv8Group); every group completed by this call is quantized into store, and tail_out receives rows
// [g0(pos0 + T), pos0 + T) - a separate buffer from tail_in. A quantization parameter that is not finite in fp16
// (a value range beyond fp16) sets err ([3] words: err[0] = 1, err[1] = the row, err[2] = the channel; the first one
// wins; report_kv8_error throws) - the stored codes of that group are then meaningless.
void kv8_commit(const void *k_new, const void *v_new, int64_t new_ts, int64_t T, int64_t pos0, const void *tail_in,
                void *tail_out, uint8_t *store, int64_t capacity, int64_t W, Act act, uint32_t *err,
                hipStream_t stream);
// err's 3 words already on the host.
void report_kv8_error(const uint32_t e[3], const char *what);

// Dequantize rows [r0, r1) (all below a view's g0) into dst [r1 - r0][W] K and V halves: dst_k / dst_v in the
// activation dtype, rows W apart. For tests and the prefill staging path.
void kv8_dequant(const uint8_t *store, int64_t capacity, int64_t W, int64_t r0, int64_t r1, void *dst_k, void *dst_v,
                 Act act, hipStream_t stream);

}  // namespace strix::kernels
