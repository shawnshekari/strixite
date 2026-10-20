#pragma once

// Full (softmax) attention for the Qwen3.5 / qwen4_exp full-attention layers
// (the model's full-attention mixer), decode-oriented.
//
// Per layer and step, with the merged q|k|v projection output qkv [T, ...]
// (formats/merge_plan; q heads are [query D | gate D]):
//   1. qk_norm_rope on q (in place) and on k (written into the K cache at
//      rows pos0..pos0+T-1); v copied into the V cache (a strided copy).
//   2. attention: softmax(q K^T * scale) V over keys 0..pos0+t for token t
//      (causal), GQA (query head h reads KV head h / (Hq/Hk)), times
//      sigmoid(gate) -> out [T, Hq*D], the o_proj input.
//
// Precision (engine semantics, as for the norms): inputs in the activation
// dtype, all attention math in FP32 (scores, online softmax, P*V), one
// rounding at the output. That is more precise than transformers' eager BF16
// run, which rounds scores, probabilities and P*V to BF16; FP32 activations
// match an FP32 reference tightly.
//
// KV cache: [capacity, Hk, D] per layer for K and V, activation dtype.

#include "kernels/kv8.hpp"   // Kv8View
#include "kernels/norm.hpp"  // Act
#include "kernels/rope.hpp"  // MropePos (interleaved mRoPE triples)

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// Zero-centred RMSNorm over each head's D dims (weight w [D], FP32, applied
// as 1 + w), then RoPE on its first rot_dim dims (kernels/rope.hpp), for T
// tokens x heads heads at positions pos0 + t. Element (t, h, d) is read from
// in[t*in_ts + h*in_hs + d] and written to out[t*out_ts + h*out_hs + d].
// In place (out == in with the same strides) is allowed; otherwise the two
// must not overlap. BF16 rounds the norm output, then RoPE rounds as usual -
// the same values as rmsnorm_zc followed by rope (cos_sin_scale as there: YaRN's attention factor, 1 = plain).
// pos3 (kernels/rope.hpp MropePos): interleaved-mRoPE section positions for the rows, null = scalar pos0 + t
// (the text path's default - a null pos3 keeps today's kernel and values bit-for-bit).
void qk_norm_rope(const void *in, int64_t in_ts, int64_t in_hs, void *out, int64_t out_ts, int64_t out_hs, int64_t T,
                  int64_t heads, int64_t D, const float *w, float eps, int64_t rot_dim, int64_t pos0,
                  const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale = 1.0f,
                  const MropePos &pos3 = {}, bool hadamard = false);

// Several qk_norm_rope jobs (and verbatim head copies) in one launch, for T tokens at positions pos0 + t: an
// attention layer's q (in place), k (into the K cache), v (copied into the V cache), the indexer queries and a verify's
// raw indexer-key save.
// Each job is (in, in_ts, in_hs) -> (out, out_ts, out_hs) over heads heads of D, normed with w then roped on
// rot_dim dims exactly as qk_norm_rope; w = nullptr copies the heads unchanged. Outputs may not overlap any
// input except a job's own exactly in place, nor each other. pos3: interleaved-mRoPE section positions for
// rows [pos0, pos0 + T), null = scalar pos0 + t (the text path's default).
// hadamard: if true, apply an in-place orthonormal 256-point Fast Walsh-Hadamard Transform after RoPE.
struct NormRopeJob {
    const void *in = nullptr;
    int64_t in_ts = 0, in_hs = 0;
    void *out = nullptr;
    int64_t out_ts = 0, out_hs = 0, heads = 0, D = 0;
    const float *w = nullptr;
    bool hadamard = false;
};
constexpr int kMaxNormRopeJobs = 5;
void qk_norm_rope_jobs(const NormRopeJob *jobs, int n, int64_t T, float eps, int64_t rot_dim, int64_t pos0,
                       const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale = 1.0f,
                       const MropePos &pos3 = {});

struct AttentionShape {
    int64_t T = 0;       // query tokens this step (1 for decode)
    int64_t pos0 = 0;    // position of token 0; keys 0..pos0+T-1 must be in the cache
    int64_t Hq = 0, Hk = 0, D = 0;
    int64_t q_ts = 0, q_hs = 0;  // q and gate strides: (t, h, d) at q[t*q_ts + h*q_hs + d]
};

// Bytes of FP32 workspace attention() needs for this shape.
size_t attention_workspace_bytes(const AttentionShape &s);

// out [T, Hq*D] = attention(q, K, V) * sigmoid(gate). k_cache/v_cache:
// [capacity, Hk, D]; capacity >= pos0 + T. D must be 256 (both target
// models); Hq % Hk == 0 with Hq/Hk <= 16 (0.8B: 4, target: 12).
void attention(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
               const AttentionShape &s, float scale, void *out, float *workspace, size_t workspace_bytes, Act act,
               hipStream_t stream);
// The same over the 8-bit K/V cache (kernels/kv8.hpp): kv holds rows [0, pos0 + T) - each query at position p
// reads the groups complete at p from the 8-bit storage and rows from kv8_math::exact_from(p) on from the tail
// before the call / the call's new rows - and kv.capacity stands for the cache's. Values read are the
// dequantized ones rounded to the activation dtype, so the result equals attention() over a cache holding them.
// Likewise for the three below.
void attention(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s, float scale, void *out,
               float *workspace, size_t workspace_bytes, Act act, hipStream_t stream);

// QSA sparse attention (full attention with the QSA indexer): as attention(), but query t
// attends only to the tokens of its selected blocks - sel [T, sel_ld] int32 block indices, nsel [T] their counts
// (kernels/qsa.hpp qsa_topk's output; each block b is tokens 4b..4b+3) - plus its (pos + 1) % 4 tail tokens: at
// most 4 sel_ld + 3 keys. With every complete block selected this is exactly attention(). sel/nsel are device data:
// a block id outside the query's complete blocks, or a count above them, sets err ([3] uint32, zeroed; err[1] = the
// token, err[2] = the value; check_attention_error throws) and is clamped, so nothing outside the cache is read.
size_t attention_gathered_workspace_bytes(const AttentionShape &s, int64_t sel_ld);
void attention_gathered(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
                        const AttentionShape &s, const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale,
                        void *out, float *workspace, size_t workspace_bytes, uint32_t *err, Act act,
                        hipStream_t stream);
void attention_gathered(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s,
                        const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale, void *out,
                        float *workspace, size_t workspace_bytes, uint32_t *err, Act act, hipStream_t stream);
void check_attention_error(const uint32_t *err, const char *what);
// The same check on the 3 words already on the host (read back with the forward's one sync).
void report_attention_error(const uint32_t e[3], const char *what);

// The same two on the matrix units, for multi-token forwards with BF16 activations (the WMMA prefill path): per
// (token, KV head) the Hq/Hk query heads as one 16-row WMMA tile - S = q K^T over 16 keys at a time, online softmax
// in FP32 as above, P * V with P as two BF16 terms (hi + lo: ~2^-17 of P), FP32 accumulation. q, K, V are BF16 bits
// (act must be BF16); the combine and output are as above.
// P * V's precision (config attention-precision): Exact - P as BF16 hi + lo (two
// products); Fast - P as one BF16 term (one product, as flash attention and transformers' BF16 run do).
enum class AttnPrecision { Exact, Fast };
size_t attention_wmma_workspace_bytes(const AttentionShape &s);
void attention_wmma(const void *q, const void *gate, const void *k_cache, const void *v_cache, int64_t capacity,
                    const AttentionShape &s, float scale, void *out, float *workspace, size_t workspace_bytes, Act act,
                    hipStream_t stream, AttnPrecision precision = AttnPrecision::Exact);
void attention_wmma(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s, float scale,
                    void *out, float *workspace, size_t workspace_bytes, Act act, hipStream_t stream,
                    AttnPrecision precision = AttnPrecision::Exact);
size_t attention_gathered_wmma_workspace_bytes(const AttentionShape &s, int64_t sel_ld);
void attention_gathered_wmma(const void *q, const void *gate, const void *k_cache, const void *v_cache,
                             int64_t capacity, const AttentionShape &s, const int32_t *sel, const int32_t *nsel,
                             int64_t sel_ld, float scale, void *out, float *workspace, size_t workspace_bytes,
                             uint32_t *err, Act act, hipStream_t stream,
                             AttnPrecision precision = AttnPrecision::Exact);
void attention_gathered_wmma(const void *q, const void *gate, const Kv8View &kv, const AttentionShape &s,
                             const int32_t *sel, const int32_t *nsel, int64_t sel_ld, float scale, void *out,
                             float *workspace, size_t workspace_bytes, uint32_t *err, Act act, hipStream_t stream,
                             AttnPrecision precision = AttnPrecision::Exact);

}  // namespace strix::kernels
