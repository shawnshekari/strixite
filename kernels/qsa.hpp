#pragma once

// qwen4_exp QSA indexer on the GPU: which
// keys each query of a full-attention layer attends to. The indexer's queries come from the merged
// q|k|v|indexer projection through qk_norm_rope (kernels/attention.hpp: per-head RMSNorm_zc + partial RoPE,
// any head dim); these kernels do the rest:
//
//   qsa_block_keys: when a 4-token block completes, its key K_b = RoPE(RMSNorm_zc(mean of its 4 raw keys))
//     at position 4b (mean in FP32, cast to the activation dtype, as the reference) goes into the block-key
//     cache; the raw keys of the still incomplete block carry over in a tail buffer (<= 3 rows) - the
//     engine option of the arch doc: 128 values per 4 tokens instead of every raw key.
//   qsa_scores: score[t, b] = (1/sqrt(D)) sum_h relu(q_h . K_b), FP32, for the query's complete blocks
//     b < (pos + 1) / 4.
//   qsa_topk: the min(k, #blocks) highest-scoring blocks per query (a set; ties to the lower block index,
//     as the reference's stable order), k = budget / 4 = 512 on the target.
// The attended keys are then the selected blocks' tokens plus the query's (pos + 1) mod 4 tail tokens.
//
// Block-key cache layout: chunk-major - element d of block b at ((d / 8) * cap_blocks + b) * 8 + d % 8 -
// so the score kernel's lanes (one block each) read one contiguous run per 16-byte load.

#include "kernels/norm.hpp"  // Act
#include "kernels/rope.hpp"  // MropePos

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace strix::kernels {

constexpr int64_t kQsaBlock = 4;  // tokens per indexer block (indexer_compress_ratio)

// raw: this call's raw indexer keys, (t, d) at raw[t*raw_ts + d], positions pos0..pos0+T-1 (a column slice
// of the merged projection output). tail_in: the raw keys of positions 4*(pos0/4) .. pos0-1 (rows in
// position order; unused when pos0 % 4 == 0); tail_out receives those of 4*((pos0+T)/4) .. pos0+T-1 - a
// separate buffer (another block may still read tail_in). Writes the block keys of every block completing
// in this call. k_norm [D] FP32 (1 + w); inv_freq: rot_dim/2 device floats (rope_inv_freq or rope_yarn);
// cos_sin_scale: YaRN's attention factor (kernels/rope.hpp), 1 = plain RoPE. pos3 (kernels/rope.hpp MropePos):
// interleaved-mRoPE section triples for the sequence positions the block keys rope at (each block's first key),
// base = 4*(pos0/4), null = scalar (text-only, today's values bit for bit).
// The byte ranges {offset, bytes} of a block-key cache ([D / 8][cap_blocks][8] elements of es bytes: element (b, d) at
// ((d / 8) * cap_blocks + b) * 8 + d % 8) that blocks [0, nb) occupy - one range per 8-dimension chunk - for paged
// storage that maps only what is used.
std::vector<std::pair<size_t, size_t>> qsa_block_keys_used_ranges(int64_t cap_blocks, int64_t D, size_t es, int64_t nb);
void qsa_block_keys(const void *raw, int64_t raw_ts, int64_t T, int64_t pos0, const void *tail_in, void *tail_out,
                    int64_t D, const float *k_norm, float eps, const float *inv_freq, int64_t rot_dim,
                    void *block_keys, int64_t cap_blocks, Act act, hipStream_t stream, float cos_sin_scale = 1.0f,
                    const MropePos &pos3 = {}, int64_t rope_delta = 0);

// q: normed + roped indexer queries, (t, h, d) at q[t*q_ts + h*q_hs + d], H heads (<= 8) x D (% 8 == 0,
// <= 256). scores [T, ld] F32: row t has its (pos0 + t + 1) / 4 complete blocks' scores; the rest of the
// row is left alone. ld >= (pos0 + T) / 4.
// Two ways: Scalar - one thread per (block, query), FP32 FMAs (decode, verifies, F32 activations); Wmma - the matrix
// units, 64 keys held in LDS against every query tile (prefill: Auto from 64 queries with BF16 activations, D % 16 == 0
// and 16-byte aligned query rows). Not bit-identical to each other (summation order, FP32 noise); tests force each.
enum class ScoresPath { Auto, Scalar, Wmma };
void qsa_scores(const void *q, int64_t q_ts, int64_t q_hs, int64_t T, int64_t pos0, int64_t H, int64_t D,
                const void *block_keys, int64_t cap_blocks, float *scores, int64_t ld, Act act, hipStream_t stream,
                ScoresPath path = ScoresPath::Auto);

// sel [T, k] int32: the selected block indices of each query, ascending (gathered attention accumulates in this
// order, so a fixed one keeps its result reproducible run to run); nsel [T] their count (min(k, complete
// blocks)). k <= 1024. Two ways, the same result: Split - across workgroups (5 launches when some query has
// more than k blocks - three histogram passes, the select, a per-row sort - else 1), for many queries (prefill);
// Row - one workgroup per query, one launch (the passes over the row in LDS, the select in index order so no
// sort), for decode and verifies (T <= kQsaTopkRowMaxT), where Split's cost was its launches. Auto picks Row for
// rows of at most kQsaTopkRowMaxBlocks blocks (any T: prefill chunks too, 3-8x faster than Split there), Coop for small T past that (the split's
// workgroups in one cooperative launch: grid syncs between the passes, a select in index order - no sort); either can be forced
// explicitly. Measured (gfx1151, T = 1): Row 7.3 / 12.5 / 33.8 / 64.0 / 124.0 us at 1k / 4k /
// 16k / 32k / 64k blocks, Split 30.5-32.5 us throughout - one workgroup streaming a long row loses to the split's
// parallel passes past ~14k blocks (~56k tokens of context). With the select from per-thread slices (T = 1):
// Row 6.9 / 8.6 / 14.3 / 29.4 us at 1k / 4k / 16k / 32k blocks; T = 5 14.6 at 16k, 55.4 at 32k - so Row up to 16384
// blocks (Coop there: 23.7 at T = 1, 27.8 at T = 5). workspace: >= qsa_topk_workspace_bytes(T), zeroed once after
// allocation (hipMemset); every completed call leaves the words a path needs zeroed at its start zeroed again (Row
// doesn't touch the workspace; Coop's per-workgroup counts are rewritten each call). Not shared between concurrent
// calls.
// Filter (many queries past kQsaTopkRowMaxBlocks: deep prefill chunks; Auto there): a bound from ~1/16 of the row,
// one pass keeping the blocks at or above it, the exact select on those - the row read about once instead of four
// times (details at the kernels). A row whose candidates miss (fewer than k, or past the list's 4096) runs the exact
// select on its whole row: same result. FilterTight: a list of exactly k (it mostly takes the fallback).
enum class TopkPath { Auto, Split, Row, Coop, Filter, FilterTight };
constexpr int64_t kQsaTopkRowMaxT = 8, kQsaTopkRowMaxBlocks = 16384, kQsaTopkCoopMaxBlocksT1 = 40960;
size_t qsa_topk_workspace_bytes(int64_t T);
// Rows of the filter path that fell back to the whole-row select since the last call (synchronizes stream; resets the
// count). For tests and benches.
int64_t qsa_topk_filter_fallbacks(void *workspace, size_t workspace_bytes, hipStream_t stream);
void qsa_topk(const float *scores, int64_t ld, int64_t T, int64_t pos0, int64_t k, int32_t *sel, int32_t *nsel,
              void *workspace, size_t workspace_bytes, hipStream_t stream, TopkPath path = TopkPath::Auto);

}  // namespace strix::kernels
