#pragma once

// Qwen3.8-Flash-Next (qwen4exp) text model on the GPU: embedding into 4
// residual streams, 48 decoder layers (36 GDN + 12 full attention with the QSA indexer; PLE at the layer that
// has it), each sublayer fed by a hyper-connection mix and written back by an injection, MoE MLP (top-10 of
// 512 + the gated shared expert), the final mix, the untied LM head.
//
// Weights come from the converter's weights.strixw (formats/strixw), loaded once into device memory
// (runtime/strixw_loader), every tensor already in its kernel's layout. The dense projections (GDN in/out,
// attention q|k|v|idx and o, PLE key|value, LM head, a separate shared expert) run as Q4 or Q8, whichever the
// layout gave each tensor, and so do the HC mixes; the routed experts are Q4 (their kernels are Q4-only, so the
// converter refuses anything else). The shared expert is expert #512 of the stack when it has the routed experts' format,
// else its own dense projections (meta shared_expert = "separate") with its sigmoid gate applied after. The PLE
// n-gram rows come from the converted table file on SSD, read on demand (runtime/ngram_table), or - for
// tests and interim use, when handed the checkpoint directory - straight from the HF checkpoint's shards.
//
// QSA: while every query of a call sees at most dense_key_limit() keys the indexer would keep them all, so dense
// attention is exact; past that the indexer scores the cached block keys, keeps the top budget / 4 blocks per
// query (plus its tail tokens) and attention runs over that gathered set (kernels/attention.hpp
// attention_gathered). Prefill (>= kWmmaMinTokens / kGroupedMinTokens tokens) runs the WMMA / grouped kernels.
//
// Qwen4ExpSession is one conversation: KV cache + indexer block keys per attention layer, conv + recurrent
// state per GDN layer, the PLE conv state and n-gram id history, the position.

#include "kernels/embedding.hpp"
#include "kernels/logits_topk.hpp"
#include "kernels/attention.hpp"  // AttnPrecision
#include "kernels/gdn.hpp"        // DrPrecision
#include "kernels/norm.hpp"  // Act
#include "runtime/device_buffer.hpp"
#include "runtime/ngram_rows.hpp"
#include "runtime/paged_buffer.hpp"
#include "runtime/ngram_table.hpp"
#include "runtime/ple_hash.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/qweight.hpp"
#include "runtime/strixw_loader.hpp"
#include "runtime/vision_tower.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace strix {

// Fixed by the architecture (config.json, checked against the tensor shapes at load).
struct Qwen4ExpDims {
    int64_t d = 2560, H = 4, r = 320, vocab = 248320, layers = 0;
    std::vector<bool> is_attention;  // per layer
    int64_t ple_layer = -1;
    // GDN: 16 key heads, 48 value heads of 128.
    int64_t gk = 16, gv = 48, conv_c = 10240, gz = 6144, gstride = 16480;
    // Full attention: 24 query heads [query 256 | gate 256], 2 KV heads, RoPE on 64 dims; the QSA indexer's
    // 4 query heads + 1 key of 128 follow q|k|v in the merged projection.
    int64_t hq = 24, hkv = 2, hd = 256, rot = 64, idx_h = 4, idx_d = 128, astride = 13952, idx_col = 13312;
    int64_t qsa_budget = 2048;
    // MoE: 512 experts + the shared one as #512, intermediate 640, top 10 (+ the shared slot).
    int64_t experts = 512, inter = 640, top_k = 10;
    // PLE: key|value projection [4d + d, 2560], conv 4 taps x dilation 3.
    int64_t ple_e = 2560, ple_ld = 12800, ple_taps = 4, ple_dil = 3;
    float eps = 1e-6f, rope_theta = 1e7f;
    // Positions: the checkpoint is trained for 262,144 (config max_position_embeddings, rope_type default). YaRN
    // (kernels/rope.hpp rope_yarn) stretches that by yarn_factor (1 = plain RoPE, as trained); rope_scale is its
    // attention factor on cos/sin (1 without YaRN). Set by Qwen4ExpModel from its yarn_factor argument.
    int64_t trained_positions = 262144;
    float yarn_factor = 1.0f, rope_scale = 1.0f;
    int64_t max_positions() const { return (int64_t)((double)trained_positions * (double)yarn_factor); }
    int64_t dense_key_limit() const { return qsa_budget + 3; }  // up to here the indexer keeps every key
};

// config.json image_token_id: the placeholder whose expanded copies mark the rows where the vision
// tower's merged rows replace the token embeddings (the id is in the vocabulary, below vocab).
constexpr int32_t kVisionImageTokenId = 248056;

class Qwen4ExpModel {
public:
    // weights: a weights.strixw with every decoder layer and the globals. ngram: where the PLE n-gram rows come from
    // - an ngram.table file (tools/convert_ngram_table; its checkpoint fingerprint must equal the weights') or the
    // HF checkpoint directory (interim reader). allow_truncated: also accept a conversion of the first k layers +
    // globals (--layers 0,..,k-1) - a truncated model for wiring tests only, its logits aren't the model's.
    // ngram_cache_rows: the table's row cache (NgramTableRows; only a table file has one - another value with the
    // checkpoint directory is refused). yarn_factor: 1 = RoPE as trained (262,144 positions); > 1 = YaRN over
    // 262,144 * yarn_factor positions (kernels/rope.hpp rope_yarn) - every RoPE user (attention q/k, the QSA indexer,
    // MTP) then rotates with the YaRN table, so caches made under another factor don't match. placement: where the
    // weights live (runtime/strixw_loader.hpp; HostExperts is for test / bench tools, never the server).
    // vision: a vision.strixw (the tower file the converter's --vision pass writes) to load the image tower
    // behind, empty (the default, the served config) = no tower - nothing loaded, Qwen4ExpSession::forward's
    // image overload refuses. Its source_fingerprint must match the text weights' (the same side-file rule the
    // n-gram table follows), so both come from the same checkpoint.
    Qwen4ExpModel(const std::string &weights, const std::string &ngram, kernels::Act act,
                  bool allow_truncated = false, int64_t ngram_cache_rows = NgramTableRows::kDefaultCacheRows,
                  float yarn_factor = 1.0f, WeightPlacement placement = WeightPlacement::Device,
                  const std::string &vision = "");
    const Qwen4ExpDims &dims() const { return dims_; }
    kernels::Act act() const { return act_; }
    const StrixwDevice &weights() const { return *w_; }
    bool truncated() const { return truncated_; }
    bool shared_separate() const { return shared_separate_; }
    // The image tower: loaded only when the constructor got a vision file. vision() throws without one
    // (no fallback path - the session's image forward is the only reader and it checks first).
    bool has_vision() const { return vision_ != nullptr; }
    const VisionTower &vision() const;

    struct Hc {
        QWeightView down;        // [r (+ H), 4d], hc_norm folded in; Q4 or Q8
        QChunkMajorView up;      // [4d, r]; Q4 or Q8
        const float *norm = nullptr;
    };
    struct Layer {
        Hc hc_attn, hc_mlp;
        // GDN
        QWeightView in_proj, out_proj;
        const float *conv_w = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gdn_norm = nullptr;
        // Full attention
        QWeightView qkv, o_proj;
        const float *q_norm = nullptr, *k_norm = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;
        // MoE: the routed experts stacked (+ the shared one as #512 unless separate).
        QWeightView gate_up, down;  // routed experts stacked [E*N, K]: Q4 or Q5 (the expert kernels')
        QWeightView shared_gate_up, shared_down;  // shared_separate() only
        const uint16_t *router = nullptr;  // BF16 [513, d]
        // PLE (ple_layer only)
        QWeightView ple_kv;
        const float *ple_conv = nullptr, *ple_norm_key = nullptr, *ple_norm_query = nullptr, *ple_norm_conv = nullptr;
    };
    // The MTP draft head: fc_embedding / fc_hidden and their pre-norms combine a
    // token with the trunk's streams per stream, one full-attention layer, and hc_mixer - the head's own collapse
    // before the shared LM head.
    struct MtpHead {
        QWeightView fc_embedding, fc_hidden;
        const float *pre_fc_norm_embedding = nullptr;
        const float *pre_fc_norm_hidden = nullptr;
        Hc hc_mixer;
        Layer layer;
    };
    const Layer &layer(int64_t i) const;
    const uint16_t *embed() const { return embed_; }
    bool is_embed_q8() const { return embed_q8_.q != nullptr; }
    const Q8DeviceView &embed_q8() const { return embed_q8_; }
    const QWeightView &lm_head() const { return lm_head_; }
    const Hc &final_mix() const { return final_; }
    const float *inv_freq() const { return inv_freq_.get(); }
    float rope_scale() const { return dims_.rope_scale; }  // cos/sin factor for every RoPE call (YaRN; else 1)
    const NgramRowSource &ngram_rows() const;
    bool has_ngram_rows() const { return ngram_ != nullptr; }
    bool has_mtp() const { return has_mtp_; }
    const MtpHead &mtp() const { return mtp_; }

    // The MTP draft's own Q4 copy of the LM head's first `rows` rows (strixite PR #2 by @TheBeaninator, rewritten;
    // formats/q4_from_q8.hpp). Every draft call scores the first mtp_vocab rows of the LM head; the served head is
    // Q8, so at 65,536 rows a draft reads 178 MB of it. The copy halves that (94 MB at G 64).
    //   - Only drafts use it: lm_head() - the trunk's logits, the verify, sampling - never changes, and the verify
    //     decides every drafted token, so the copy can change how many drafts are accepted, never the output.
    //   - Q8 head: read back from the device, dequantized exactly, quantized as formats/q4 does with group G.
    //     Q4 head: used as loaded (G must then equal the head's own group size). Other widths (5 / 6 bits): refused.
    //   - Sessions draft over it whenever it covers their mtp_vocab() rows (Qwen4ExpSession::run_mtp), else over
    //     lm_head() as before - a model that never makes the copy behaves exactly as before.
    // Preconditions (each a thrown error naming the values): the model has an MTP head; 1 <= rows <= vocab; G is
    // 32 / 64 / 128 and divides the head's K; no copy made yet (call once, at load, before any session drafts -
    // sessions read it without locking). threads: 1..64 CPU workers for the requantization (~0.1 s at 65,536 rows).
    void make_draft_head_q4(int64_t rows, int64_t G = 64, int threads = 16);
    // The copy (bits == 0 until make_draft_head_q4 ran).
    const QWeightView &draft_head_q4() const { return draft_head_; }
    // Device bytes the copy holds (0 when none, or when it is the loaded Q4 head itself).
    size_t draft_head_q4_bytes() const { return draft_q4_ ? draft_q4_bytes_ : 0; }

private:
    Qwen4ExpDims dims_;
    kernels::Act act_;
    std::unique_ptr<StrixwDevice> w_;
    std::unique_ptr<VisionTower> vision_;  // the constructor's vision file, empty = no tower
    std::unique_ptr<NgramRowSource> ngram_;
    std::vector<Layer> layers_;
    const uint16_t *embed_ = nullptr;
    Q8DeviceView embed_q8_{};
    QWeightView lm_head_;
    QWeightView draft_head_;                 // make_draft_head_q4's copy (bits 0: none)
    std::unique_ptr<Q4Device> draft_q4_;     // its storage when made from a Q8 head
    size_t draft_q4_bytes_ = 0;
    bool truncated_ = false, shared_separate_ = false, has_mtp_ = false;
    MtpHead mtp_;
    Hc final_;
    DeviceBuffer<float> inv_freq_;
};

// What strix_server does at load when MTP is on, shared so every tool that loads the model runs the served path: the Q4 draft head copy (Qwen4ExpModel::make_draft_head_q4, group 64) over the rows the drafts
// score - mtp_vocab rows, or the whole vocabulary when mtp_vocab is 0. Returns the rows copied. Throws (from
// make_draft_head_q4) when the model has no MTP head or mtp_vocab is outside 0..vocab.
int64_t make_served_mtp_draft_head(Qwen4ExpModel &model, int64_t mtp_vocab);

// Called with each intermediate under the reference implementation's names where they exist ("embed", "L<i>.attn_in",
// "L<i>.mixer", "L<i>.mlp_in", "L<i>.router_logits", "L<i>.moe", "L<i>.out", "final_mixed") plus what a
// sublayer-by-sublayer check needs ("L<i>.ple_in" / "L<i>.ple_out" (the streams around the PLE),
// "L<i>.attn_w_in" / "L<i>.mlp_w_in" [T, 4] F32, "L<i>.router_ids" [T, A] I32, "L<i>.router_coef" [T, A] F32;
// A = 11 with the shared expert stacked as #512, 10 when it's separate; past the dense key limit "L<i>.qsa_sel"
// [T, budget / 4] I32 (the kept blocks, the first "L<i>.qsa_nsel" [T, 1] of each row valid).
// dev: rows x cols (row stride cols) on the device, valid during the call (the stream is synchronized first).
// F16: "L<i>.experts_h" when prefill's grouped experts keep h as F16 bits (BF16 activations, Q4 experts).
enum class ProbeType { F32, BF16, I32, F16 };
using Qwen4ExpProbe =
    std::function<void(const std::string &name, const void *dev, int64_t rows, int64_t cols, ProbeType type)>;

// How multi-token forwards (prefill) compute their Q4 / Q8 GEMMs: on the matrix units with BF16 inputs and FP32
// accumulation (the default: drift vs the reference outputs equal to the FP32 path's, 2.5x its prefill), or FP32 on the vector units (the reference semantics, for tests and comparison).
// Decode (and forwards below kWmmaMinTokens / kGroupedMinTokens) is the same either way.
enum class PrefillMath { F32, WmmaBf16 };

// Per-step precision / speed choices (the server's gdn-precision and attention-precision). The defaults
// are the exact math; every non-default choice changes the numbers a forward produces, so it is part of the prompt
// cache's state fingerprint and of every logged result.
struct PrecisionOptions {
    kernels::DrPrecision gdn = kernels::DrPrecision::Exact;          // the prefill delta rule's products
    kernels::AttnPrecision attention = kernels::AttnPrecision::Exact;  // the prefill attention's P * V
    bool exact() const { return gdn == kernels::DrPrecision::Exact && attention == kernels::AttnPrecision::Exact; }
    std::string describe() const;  // "gdn=exact attention=exact"
};
// "exact" / "fast" -> DrPrecision; anything else throws, naming the choices.
kernels::DrPrecision parse_gdn_precision(const std::string &v);
kernels::AttnPrecision parse_attention_precision(const std::string &v);

// The sequence state at one position, for resuming there later (prompt reuse): every
// GDN layer's conv + recurrent state, every attention layer's indexer tail (the incomplete QSA block's keys), the
// PLE conv state and n-gram history. The KV cache and block keys below the position aren't copied - later calls only
// write at or past the position they start from, so those stay valid until the session rewinds below the
// snapshot's position (Qwen4ExpSession::can_restore tracks that). With MTP on, also the MTP layer's indexer tail
// and the trunk's streams after the last position (the next MTP row's input). ~110 MiB on the target, allocated
// once by Qwen4ExpSession::make_snapshot and reused.
// The first bytes of an exported state (export_state / import_state): version, then the mRoPE position state.
// Version 1 is the first layout that carries it; a blob without the header (version 0, before Phase A) is
// refused - the prompt cache's disk entries from before it are comparison-only and get dropped, never guessed
// at (no fallback path). The header carries the state's rope_delta, whether its prefix contained image rows,
// and the continuation rule of row n plus the last (up to 3) rows' triples - the same rule Qwen4ExpSnapshot
// carries, and the same data an import needs to rope the first forward's block window exactly (import_state
// accepts a uniform-tail state without the triples, an image-tail one with them, and refuses the rest -
// restore's rule, which has the snapshot's tail already).
struct Qwen4ExpStateHeader {
    static constexpr uint32_t kMagic = 0x3165706D;  // bytes 'm','p','e','1': 829M, above any token id, so it can
                                                     // never alias the version-0 blob's first word (the n-gram
                                                     // history's int64 token id)
    static constexpr uint32_t kVersion = 1;         // the state blob layout (this struct's presence and fields)
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    int64_t rope_delta = 0;        // rows after the last image: rope angle at seq + delta
    int64_t cont_t = 0, cont_h = 0, cont_w = 0;  // the (t,h,w) sections of row n (the state's continuation rule)
    int64_t has_image = 0;         // 1: the prefix contained interleaved (image) rows
    int64_t ntriples = 0;          // 0..3 trailing rows the triples array holds (valid when has_image)
    int64_t triples[9] = {};       // their (t,h,w), oldest first: rows [n - ntriples, n)
};
static_assert(std::is_standard_layout_v<Qwen4ExpStateHeader> && std::is_trivially_copyable_v<Qwen4ExpStateHeader>,
              "the state header is memcpy'd as raw bytes");

// mRoPE (vision Phase A): the interleaved-position state at the saved position - rope_delta and the
// continuation rule, plus the last 3 rows' section triples (enough to continue a uniform text span, the only
// resume point until image serving: mid-image resume isn't a thing yet). has_triples(): the prefix ran
// interleaved; restore() refuses a saved position whose last rows were image rows (with today's serve there are
// no interleaved sessions at all - set_positions has no caller until Phase B - so no served snapshot carries
// triples).
struct Qwen4ExpSeqState;  // the per-conversation state (below); a snapshot belongs to one
class Qwen4ExpSnapshot {
public:
    int64_t pos() const { return pos_; }  // -1 until saved
    // the saved rope_delta (0 for a prefix without images) and the continuation rule for the first row after
    // pos(): all three sections = pos() + rope_delta() when the saved span was uniform, the last saved row's
    // +1 in (h, w) when it was an image row (an image-span resume - restore refuses those today).
    int64_t rope_delta() const { return rope_delta_; }
    const int64_t *first_continuation() const { return first_cont_.data(); }  // [3] for row pos()
    bool has_triples() const { return ntriples_ > 0; }
    int64_t ntriples() const { return ntriples_; }              // 0..3
    const int64_t *triples() const { return triples_.data(); }  // last ntriples_ saved rows' [row][3]
private:
    friend class Qwen4ExpSession;
    int64_t pos_ = -1;
    uint64_t epoch_ = 0;
    std::vector<DeviceBuffer<uint8_t>> conv_, tail_, kv8_tail_;  // kv8_tail_: KvFormat::Q8 only
    std::vector<DeviceBuffer<float>> rec_;
    DeviceBuffer<uint8_t> ple_state_, mtp_tail_, mtp_prev_;
    PleHistory ple_hist_;
    int64_t rope_delta_ = 0;
    std::array<int64_t, 3> first_cont_{0, 0, 0};  // continuation sections of row pos_ (computed by save())
    int64_t ntriples_ = 0;
    std::vector<int64_t> triples_;  // ntriples_ * 3 host int64 (empty until a save with an interleaved prefix)
    friend class Qwen4ExpSession;
    friend struct Qwen4ExpSeqState;
};

// One image for a forward (the vision build's forward extension): the tower's input rows, FP32 on the
// host, [grid_t * grid_h * grid_w, in_channels x temporal_patch x patch x patch] in the processor's
// merge-block order (the packing the tower is gated against - the preprocessing that produces it is a
// later step), and the grid. grid_t must be 1 (images; video isn't built). forward() with an ImageInput
// runs the tower over these rows and scatters its merged rows at the image_token_id positions of ids -
// the caller expands the placeholders first and pairs the call with set_positions (the (t,h,w) triples
// of the image rows: t = the block start, h/w its within-block indices, the walk's rule).
struct ImageInput {
    std::vector<float> pixel_values;  // host FP32, grid_t * grid_h * grid_w rows
    int64_t grid_t = 0, grid_h = 0, grid_w = 0;
};

// The full-attention layers' K/V cache: in the activation dtype, or 8 bits (kernels/kv8.hpp). Q8 keeps ~56% of the
// bytes; the rows of the newest incomplete 32-token group stay in a small tail in the activation dtype.
// KvFormat; the attention layers' K/V cache format. The MTP head's own K/V stays in the activation dtype either way.
enum class KvFormat { Act, Q8 };
const char *kv_format_name(KvFormat f);  // "bf16" / "q8" (Act is named by the config as bf16)
KvFormat parse_kv_format(const std::string &v);  // "bf16" -> Act, "q8" -> Q8; anything else throws

// ---------------------------------------------------------------------------
// The state of one conversation (feat/multi-sequence): everything a Qwen4ExpSession
// carried across forwards for one sequence - the buffers one serving slot owns. A session holds one SeqState per
// slot and its forward / verify API addresses them by run; with one slot this is exactly the old session state.
// Qwen4ExpSnapshot already saved and restored exactly this set (plus the K/V below the position, which stays in
// place), so it is now a member: a snapshot belongs to one state and is valid only against it. Everything else
// in Qwen4ExpSession is per-forward workspace, sized by the largest forward - shared by the slots.
// ---------------------------------------------------------------------------
struct Qwen4ExpSeqState {
    // Allocates the buffers for one position space of `capacity` positions (the MTP head's own K/V cache and
    // indexer keys exist with `mtp`; the K/V layout is `kv`); everything starts at position 0. `chain_rows`:
    // the row count of the session's forward workspaces - the draft chain swaps its stream buffer against the
    // shared mtp_x_, so they must be the same size.
    Qwen4ExpSeqState(const Qwen4ExpModel &model, int64_t capacity, int64_t chain_rows, bool mtp, KvFormat kv);
    // The paged arrays' mapping unit. gfx1151: a 2 MiB page is created in ~45 us and mapped pages read at hipMalloc's
    // speed; slack per array is under one page.
    static constexpr size_t kPageBytes = (size_t)2 << 20;

    int64_t pos = 0;
    // Where the last import_state ended (-1: none since reset): a delta import needs its base's import just
    // before - and pos still there (any forward moves it; restores need snapshots, which an import invalidates).
    int64_t imported_at = -1;
    // Per-layer state (empty buffers for the other kind). The capacity-sized arrays (K / V caches, block keys, the
    // 8-bit K/V below, the MTP head's) are paged: their address range covers the capacity, their pages are mapped as
    // the conversation fills them (Qwen4ExpSession::ensure_rows).
    std::vector<PagedDeviceBuffer> k_cache, v_cache, block_keys;
    std::vector<DeviceBuffer<uint8_t>> tail[2], conv_state;
    std::vector<DeviceBuffer<float>> rec_state;
    int tail_cur = 0;  // the current indexer tails - and with KvFormat::Q8 the current K/V tails
    // KvFormat::Q8 (k_cache / v_cache empty then): per attention layer the 8-bit storage (kernels/kv8.hpp, for the
    // capacity rounded up to whole groups) and the K/V tails [2] (kv8_tail[tail_cur] current, double-buffered).
    std::vector<PagedDeviceBuffer> kv8_store;
    std::vector<DeviceBuffer<uint8_t>> kv8_tail[2];
    DeviceBuffer<uint8_t> ple_state;
    PleHistory ple_hist;
    // Rewinds for can_restore: epoch e started when the state rewound to rewind_to[e] (reset() = 0).
    std::vector<int64_t> rewind_to{0};
    // The MTP head's state (with mtp): the layer's K / V cache, block keys and indexer tails
    // (tail_mtp[mtp_tail_cur] current); mtp_prev [H*d] the trunk's streams after position pos - 1 (zeros at 0),
    // row 0's input for the next MTP rows; and the draft chain's own buffers (step - 1's output streams -
    // swapped against the session's mtp_x_ mid-chain, so only one state may draft at a time - the session
    // enforces it - and the chain's own indexer tails).
    PagedDeviceBuffer k_cache_mtp, v_cache_mtp, block_keys_mtp;
    DeviceBuffer<uint8_t> tail_mtp[2], mtp_prev;
    DeviceBuffer<uint8_t> mtp_chain_prev, mtp_chain_tail[2];
    int mtp_tail_cur = 0;
    int64_t mtp_chain_step = -1, mtp_chain_pos = -1;
    // The mtp_x_ write stamp at the chain's last step (Qwen4ExpSession::mtp_x_version_, bumped by every
    // mtp_input): a chain step's prev is the step before's mtp_x_, so if the stamp moved, another draft or
    // a forward interleaved and the chain must not continue (run_mtp refuses).
    uint64_t mtp_chain_xver = 0;
    // forward_verify's machinery, per state (one verify in flight per state): the spare GDN states it writes
    // (gdn_to_spare during it), what drop_verify puts back, whether a keep / drop is owed, and the verify's
    // saved rows / bookkeeping for keep_verify_prefix.
    std::vector<DeviceBuffer<uint8_t>> conv_spare;
    std::vector<DeviceBuffer<float>> rec_spare;
    DeviceBuffer<uint8_t> ple_state_kept, mtp_prev_kept;
    bool gdn_to_spare = false, verify_pending = false;
    int64_t verify_pos = 0;
    int verify_tail_cur = 0, verify_mtp_tail_cur = 0;
    PleHistory verify_hist;
    // keep_verify_prefix's inputs, saved by the verify forward per layer (kMaxLogits rows each): GDN layers
    // their in_proj rows [T, gstride], attention layers their raw indexer key rows [T, idx_d] (with
    // KvFormat::Q8 followed by their new K | V rows [T, 2 hkv hd]); and the verify's ids / size.
    std::vector<DeviceBuffer<uint8_t>> verify_in;
    std::vector<int32_t> verify_ids;
    // The interleaved mRoPE position state (vision Phase A), per conversation: rope_delta /
    // history_ever_interleaved are the scalars; pos3_host holds the (t,h,w) triples of every row from the first
    // interleaved forward on (rows before it are uniform at rope_delta 0 - never stored), reaching to pos.
    // pending3 is set_positions' payload for this state's next forward. Never touched while no positions are
    // set - the text path's cost is two int64 adds.
    int64_t rope_delta = 0;
    bool history_ever_interleaved = false;  // the current prefix contains interleaved (image) rows
    int64_t pos3_start = 0;
    std::vector<int64_t> pos3_host;
    std::vector<int64_t> pending3;
    // forward_verify's mRoPE bookkeeping: the history / delta / image flag as they were before it (drop_verify
    // restores them; keep keeps, keep_verify_prefix truncates the forward-appended rows in place).
    std::vector<int64_t> verify_hist_before;
    int64_t verify_hist_start_before = 0, verify_delta_before = 0;
    bool verify_ever_before = false;
    // A batched forward's verify rows: how many rows of this state forward_batch_verify took part in (0: it
    // is not in the pending verify's batch), and where its rows sit in the batch's shared row buffers: the
    // row base of the trunk/MTP activations (verify_row_base) and the start of its PLE history window
    // (verify_ple_wbase, the windows laid end to end). keep_verify_prefix_seq's replay reads them back.
    int64_t verify_rows = 0, verify_row_base = 0, verify_ple_wbase = 0;
    // keep_verify(seq, rows, carry)'s carry: the token owed to the run's next row 0 (the MTP rejection's
    // corrected sample; 0: none) - forward_batch / forward_batch_verify check a run of this state starts
    // with it and clear it. Nothing advances on the state for it (see the keep_verify note in the session).
    int32_t verify_carry = 0;
    // A batched forward's PLE n-gram rows: this state's rows gather into its own pinned staging (the session's
    // two ping-pong buffers and the lookahead machinery are the single-run path); the gather's future is joined
    // at the PLE layer of the forward that queued it.
    PinnedHostBuffer ple_stage;
    std::future<void> ple_fut;

    int64_t pos3Hend() const { return pos3_start + (int64_t)pos3_host.size() / 3; }  // == pos when non-empty

    // The MTP head's state exists with mtp (Qwen4ExpSession::forward's MTP catch-up and the draft check this).
    bool mtp = false;

    friend class Qwen4ExpSession;
};

class Qwen4ExpSession {
public:
    // One serving slot = one Qwen4ExpSeqState; n_slots >= 1 (today's server: n_slots 1, the forward API is
    // unchanged; the multi-sequence build addresses slots by run). mtp: keep the MTP draft head's state (its KV
    // cache, filled for every position a forward runs) so forward_mtp can draft; needs a model with the head.
    // Off, forward_mtp throws and forwards skip that work. kv: the attention layers' K/V cache format.
    Qwen4ExpSession(const Qwen4ExpModel &model, int64_t capacity, int64_t max_tokens,
                    PrefillMath prefill_math = PrefillMath::WmmaBf16, bool mtp = false, KvFormat kv = KvFormat::Act,
                    int64_t n_slots = 1);
    PrefillMath prefill_math() const { return prefill_math_; }
    KvFormat kv_format() const { return kv_format_; }
    // The kernel-step precision choices for every forward from now on (default: all exact).
    void set_precision(const PrecisionOptions &p);
    const PrecisionOptions &precision() const { return precision_; }
    ~Qwen4ExpSession();
    Qwen4ExpSession(const Qwen4ExpSession &) = delete;
    Qwen4ExpSession &operator=(const Qwen4ExpSession &) = delete;

    int64_t n_slots() const { return (int64_t)seqs_.size(); }

    void reset();  // every slot, and clears a failed call's mark
    // One slot back to position 0, the others untouched; refused after a failed call (reset() all) or while its
    // verify is pending.
    void reset(int seq);
    // Paged slot K/V: before a slot's arrays map more pages, f(bytes) is asked for
    // the RAM (the server's prompt cache makes room - K/V first) and throws to refuse; unset, growth is refused only
    // when it would leave MemAvailable below the slot floor. A refusal fails the call before any page is mapped.
    void set_kv_budget(std::function<void(uint64_t bytes)> f) { kv_budget_ = std::move(f); }
    // The bytes one slot's paged arrays map at the full capacity (the startup memory plan's deep-slot term).
    uint64_t kv_full_bytes() const;
    // The bytes slot seq's paged arrays map now: the deepest position any of its conversations reached (pages are
    // never given back while the session lives - reset and import keep them).
    uint64_t kv_mapped_bytes(int seq) const;
    bool broken() const { return broken_; }  // an earlier call failed midway: only reset() clears it
    int64_t pos() const { return seqs_[0]->pos; }
    int64_t pos(int seq) const;  // slot seq's position (pos() is seq 0's)
    int64_t capacity() const { return capacity_; }
    int64_t max_tokens() const { return max_tokens_; }

    // ---------------------------------------------------- interleaved mRoPE positions (the vision build plan step 3,
    // Phase A). The trunk's rope is interleaved mRoPE: every row carries (t, h, w) section positions and pair j
    // rotates with triple[j % 3] (kernels/rope.hpp MropePos). Text-only rows have all three sections equal and
    // need no per-row data - after an image block the text rows continue at (max position of the image span + 1),
    // which is a constant rope_delta = (continuation of the last row) - (sequence index of the next row) from
    // then on. Until a caller sets positions (Phase B does, per image prompt) every session is text-only and
    // this whole layer costs exactly nothing: no upload, the scalar kernels unchanged.
    //
    // pos3: 3 * rows host int64 (the (t,h,w) section positions of the next forward's rows, sequence order; not
    // sequence indices). Empty (the default of every text-only call) = today's scalar pos() + t path. The rows
    // must start exactly at pos() (a partial image row is never a forward's row 0) and every section value must
    // be a position the model can represent. set_positions before forward / forward_verify consumes it; before
    // any of the other calls that touch position (forward_mtp, save, import_state) it is refused.
    void set_positions(std::vector<int64_t> pos3);
    int64_t rope_delta() const { return st().rope_delta; }  // rows after the last image: rope angle at seq + delta
    // The rope angle position of sequence position seq on the scalar path (seq + rope_delta). Sequence-index
    // uses (cache rows, QSA block counts, dense-limit checks) stay seq - only rope angles use this.
    int64_t rope_pos(int64_t seq) const { return seq + st().rope_delta; }

    // Snapshots (Qwen4ExpSnapshot): make one (allocates), save the state at pos() into it, and later restore it -
    // which needs every rewind since the save (reset() or a restore()) to have gone no lower than its position.
    Qwen4ExpSnapshot make_snapshot() const;
    // The per-slot forms (feat/multi-sequence): a snapshot belongs to one state - made from it, saved into
    // it, restored onto it, and its epoch is that state's rewind history. The calls above are seq 0.
    Qwen4ExpSnapshot make_snapshot(int seq) const;
    void save(int seq, Qwen4ExpSnapshot &s);
    bool can_restore(int seq, const Qwen4ExpSnapshot &s) const;
    void restore(int seq, const Qwen4ExpSnapshot &s);
    void save(Qwen4ExpSnapshot &s);
    bool can_restore(const Qwen4ExpSnapshot &s) const;
    // throws unless can_restore(s); pos() = s.pos() after. An interleaved snapshot whose saved rows end inside an
    // image span (has_triples, continuation != uniform) is refused: mid-image resume isn't a thing yet
    // (vision Phase A); a snapshot ended after the image resumes at rope_delta as if it had
    // always been scalar.
    void restore(const Qwen4ExpSnapshot &s);

    // Host export / import of a whole state, for the disk prompt cache: the mRoPE header (Qwen4ExpStateHeader:
    // version, rope_delta, has_image, the continuation rule for row n) first, then the n-gram history, then per
    // layer - attention: K and V rows [0, n), block keys of the n / 4 complete blocks
    // (packed [chunk][block][8]; on the device they're chunk-major with the capacity as the stride), the indexer
    // tail; GDN: conv and recurrent state - then the PLE conv state; with MTP on, then the MTP layer's K and V rows,
    // block keys and indexer tail, and the trunk's streams after position n - 1. state_bytes(n) is its size.
    // export_state takes the KV below s.pos() from the session and the states from s (needs can_restore(s));
    // import_state puts a state for positions [0, n) in place (pos() = n) and invalidates every snapshot.
    // Deltas (from > 0; the prompt cache's delta entries, serve/prompt_cache.hpp): the same layout with only the K / V
    // rows [from, n) and the block keys of blocks [from / 4, n / 4) - the block holding position `from` is completed
    // after it - and every per-step part (n-gram history, tails, GDN, PLE, MTP streams) whole, as at n. Rows below
    // `from` are the base state's, unchanged since (a session only ever writes at or past its position). Importing a
    // delta needs the base imported just before (pos() == from, no forward or restore since): import_state(base,
    // .., from), then import_state(delta, .., n, from). from = 0 is the whole state.
    size_t state_bytes(int64_t n, int64_t from = 0) const;
    void export_state(const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from = 0);
    void import_state(const uint8_t *in, size_t bytes, int64_t n, int64_t from = 0);
    // The per-slot forms (feat/multi-sequence): the state of slot seq (its KV, tails, GDN, PLE, MTP); the calls above
    // are seq 0. The snapshot passed to export_state must be one saved from that slot.
    void export_state(int seq, const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from = 0);
    void import_state(int seq, const uint8_t *in, size_t bytes, int64_t n, int64_t from = 0);

    // Runs ids at positions pos()..pos()+T-1, advances pos(), returns the last n_logits tokens' logits
    // [n_logits, vocab] FP32. If it throws, the session refuses further calls until reset().
    static constexpr int64_t kMaxLogits = 16;
    // Forwards of at least this many tokens run the MoE experts grouped by expert (each expert's weights read
    // once per tile, kernels/moe_grouped); shorter ones (decode, MTP verify) the per-slot kernels (grouped measured
    // slower there: +5 ms a 2-token verify). Swept on the served layout (prefill): T = 16 124 vs 99 t/s, 32 136 vs 131, 64 142 vs 154, 128 141 vs 166
    // - break-even ~38 tokens, so 48. Re-swept after the F16 prefill GEMMs (one forward from position 0, ms): T = 6 58.8 per-slot vs 57.0 grouped, 8 72.0 vs 65.9, 12 100.0 vs 84.8, 20 152.6
    // vs 118.4, 40 203.1 vs 109.8 - so 6. A 4-draft MTP verify (5 tokens) stays per-slot. Tool-call turns (12-32 new
    // tokens) were paying 40-70 ms each for the old threshold.
    static constexpr int64_t kGroupedMinTokens = 6;
    // With PrefillMath::WmmaBf16, forwards of at least this many tokens run the dense Q4 / Q8 linears and the HC
    // mixes on the matrix units (a 64-row tile). Swept on the served layout (prefill t/s FP32 vs WMMA): T = 16 124 vs 107,
    // 24 131 vs 129, 32 136 vs 144, 48 145 vs 226 - break-even ~24-26, so 32. Re-swept with the experts grouped
    // (ms): T = 8 65.9 FP32 vs 76.8 WMMA, 10 78.8 vs 80.2, 12 84.7 vs 82.8, 16 101.0 vs 88.6, 24
    // 131.3 vs 97.4 - so 12.
    static constexpr int64_t kWmmaMinTokens = 12;
    // With PrefillMath::WmmaBf16 and BF16 activations, the trunk's full-attention layers run on the matrix units
    // (kernels/attention.hpp attention_wmma: a token's query heads as one 16-row tile) from this many rows - every
    // forward, decode and MTP verify included. Per row the vector kernel's 12 waves each re-read the K / V tile from
    // LDS and the kernel waits on LDS; the matrix units read it once a tile. Per call (2048 gathered keys, Hq 24):
    // vector vs WMMA, BF16 K/V, T = 1 46.8 vs 32.7 us, 4 111.0 vs 29.6, 8 201.2 vs 48.0; 8-bit K/V 46.0 vs 26.6,
    // 114.4 vs 53.7, 214.7 vs 101.6. The MTP head's own layer stays on the vector kernel.
    static constexpr int64_t kWmmaAttnMinTokens = 1;
    // WMMA prefill attention over the 8-bit cache stages the dequantized rows (dense and QSA-gathered) for forwards
    // of at least kKv8StageMinTokens tokens, when the unpacking it saves outweighs the copy: per (query, key row it
    // reads) over the 12 layers unpacking costs kKv8UnpackNs more than a staged load, and staging costs kKv8StageNs a
    // row below the call. A gathered query reads qsa_budget rows, so staging pays from T ~ rows / 512: ~256 tokens at
    // 131k positions, ~1000 at 489k; prefill chunks are far above that, small turn deltas at depth below it.
    static constexpr int64_t kKv8StageMinTokens = 128;
    static constexpr double kKv8UnpackNs = 30.0, kKv8StageNs = 120.0;
    // Whether a T-token forward whose queries each read keys_per_query of the `rows` 8-bit rows stages them.
    static bool kv8_stage_pays(int64_t T, int64_t keys_per_query, int64_t rows) {
        return T >= kKv8StageMinTokens && (double)T * (double)keys_per_query * kKv8UnpackNs >= (double)rows * kKv8StageNs;
    }
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits, const Qwen4ExpProbe &probe = nullptr);
    // The image forward (the vision build's forward extension): runs the tower over the image and
    // scatters its merged rows at the image_token_id positions of ids, then forwards exactly like the
    // overload above - which itself is this one with no image (the text path stays bit-identical and
    // touches nothing new). Needs a model loaded with a vision file. The checks that refuse a mismatch:
    // the tower's rows must equal the ids' image_token_id count; grid_t must be 1; and the call must
    // carry its set_positions triples (the image rows' (t,h,w) - set_positions already validated they
    // continue the walk), so an image can't forward with plain scalar positions by mistake.
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits, const ImageInput *image,
                               const Qwen4ExpProbe &probe = nullptr);
    // The next forward / forward_verify (one call) reduces its logits rows on the GPU to each row's top
    // kernels::kLogitCands candidates over ids [0, n_valid) (kernels/logits_topk) and copies only those back: it
    // returns an empty vector, candidates() holds them. n_valid: 1..vocab. Measured reason: the host
    // copied and scanned ~1 MB a row after every forward.
    // masks (structured output): null, or [mask_rows, mask_words] host words - one allowed-token mask per logits
    // row of that forward (mask_rows must equal its n_logits; mask_words >= ceil(vocab / 32)), copied here into a
    // pinned buffer and uploaded by the forward ahead of logits_topk, which leaves disallowed ids out.
    void want_candidates(int64_t n_valid, const uint32_t *masks = nullptr, int64_t mask_rows = 0, int64_t mask_words = 0);
    struct Candidates {
        int64_t rows = 0;
        std::vector<kernels::LogitCand> cand;  // [rows, kernels::kLogitCands]
        std::vector<uint32_t> nan;             // [rows]
    };
    const Candidates &candidates() const { return cand_host_; }  // from the last forward that wanted them
    // Draft verification without a snapshot copy: forward_verify is forward() writing the GDN states into a spare
    // set (the ones before it kept); then exactly one of keep_verify() (the spare set becomes current: as if it were
    // forward()) or drop_verify() (the session is back where it was before the call - position, GDN, PLE, indexer
    // tails, MTP state), before any other call. The spare set (~110 MiB) is allocated on the first use.
    // pos3 (mRoPE): interleaved triples for the verify's rows (set_positions semantics; consumed here instead of
    // by the next forward). The engine passed them to the draft forwards the tokens were produced with, so it
    // knows them; keep_verify_prefix / drop_verify truncate the history with them.
    std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits,
                                      const std::vector<int64_t> &pos3 = {});
    void keep_verify();
    void drop_verify();
    // ---------------------------------------------------- batched forwards (feat/multi-sequence design)
    // One row group: the slot `seq`, the ids to run at its pos(), and the trailing rows whose logits come
    // back. A batched forward reads the dense / expert / HC weights once for all runs' rows - the gain the
    // design measures - and touches each run's own state (K/V, GDN, PLE, MTP) per run, at the same row
    // offsets a single-sequence forward() uses. The runs' rows are laid into the shared workspaces in batch
    // order (any slot order; the returned logits rows follow it).
    struct BatchRun {
        int seq = 0;                 // the slot; distinct across a batch
        std::vector<int32_t> ids;    // 1..kMaxLogits rows at a batched verify; each run's rows at pos(seq)..
        int64_t n_logits = 1;        // its trailing rows' logits, in the batch's returned rows
    };
    // The batched forward. Returns the concatenated per-run logits rows ([run 0's n_logits rows][run 1's ...]
    // FP32) or, with want_candidates(), the batch's candidates rows in the same order. A batch of one runs
    // exactly what forward() does (bit-identical); the batch-wide path choice (WMMA / grouped at the total rows)
    // is why a batched verify is capped at kMaxLogits rows total - it stays a decode-sized forward. A run's
    // results depend on its own slot and ids only, bit for bit; next to the same run alone they are bit-identical
    // when the batch's total rows and the run's own fall on the same side of 1, kGroupedMinTokens and
    // kWmmaMinTokens, otherwise they differ by rounding (tests: forward_qwen4exp_batch_*). Refused while any
    // verify awaits resolution; with more than one run: no image rows, scalar positions only (the one-sequence
    // vision / mRoPE path keeps its own code - and an image forward runs on seq 0, set_positions' only slot).
    std::vector<float> forward_batch(std::vector<BatchRun> &runs, const ImageInput *image = nullptr,
                                     const Qwen4ExpProbe &probe = nullptr);
    // forward_batch with the verify mechanism (the batched MTP verify): each run's GDN states write to its
    // state's spare set and its rows are saved per layer for the prefix replay; then every run is resolved
    // with keep_verify() / drop_verify() below before anything else runs on the session.
    std::vector<float> forward_batch_verify(std::vector<BatchRun> &runs);
    // Resolve one run of a pending batched verify (the single-sequence calls above address the same machine
    // for one run): keep its first `rows` rows (rows in 1..verify_rows(seq): rows == its rows is the whole
    // verify - today's keep_verify(), a smaller count is today's keep_verify_prefix(rows)). carry: the token
    // the run feeds next at row 0 (the MTP rejection's corrected sample; 0: none), checked against that
    // run's next forward's ids[0] and consumed there - like the single-sequence engine after
    // keep_verify_prefix, the resolution itself advances nothing for it (its trunk rows take the full
    // forward to write). The last resolution frees the session: once every pending run is resolved,
    // forwards and drafts may run again.
    void keep_verify(int seq, int64_t rows, int32_t carry = 0);
    void drop_verify(int seq);
    int64_t verify_rows(int seq) const { return seqs_[(size_t)seq]->verify_rows; }  // a pending verify's rows for keep_verify
    // Can this slot's next forward share one with other slots'? forward_batch's multi-run contract (its
    // declaration) takes scalar positions only, and what breaks it is the mRoPE position state: the interleaved
    // history an image leaves behind, the position delta recomputed from it, set_positions' triples not yet
    // forwarded. The history and the delta are the conversation's for good - resumed out of the prompt cache it
    // comes back with them (the saved has_image) - so this is a property of the conversation, not of the step:
    // a slot that once ran an image never shares a batch again.
    bool batchable(int seq) const {
        const Qwen4ExpSeqState &s = *seqs_[(size_t)seq];
        return s.pending3.empty() && !s.history_ever_interleaved && s.rope_delta == 0;
    }
    // The third way out of a verify: keep only its first `rows` (1 <= rows < its size) - the state as if forward()
    // had run just ids[0, rows), without running them again (an MTP rejection after rows - 1 accepted drafts; the
    // engine then forwards the corrected token alone). From the states before the verify (still intact): the GDN
    // conv + delta rule replayed over the saved in_proj rows, the indexer tails over the saved raw key rows (the MTP
    // layer's straight from the verify's projections), the PLE state a window of the verify's history rows, the MTP
    // streams the verify's row rows - 1. The K / V caches below the new position are already right (causal).
    void keep_verify_prefix(int64_t rows);
    // PLE n-gram rows: gathers run (and seconds spent in them), forwards that waited for their rows at the PLE layer
    // (seconds in total, the longest wait). Safe to read from another thread.
    // Prefetches: jobs run (seconds spent in them), skipped for a full queue, failed (the gather that needs the
    // rows reads them again and reports its own failure).
    struct PleStats {
        int64_t gathers = 0, waits = 0, prefetches = 0, prefetch_skipped = 0, prefetch_failed = 0;
        double gather_seconds = 0, wait_seconds = 0, wait_max_seconds = 0, prefetch_seconds = 0;
    };
    PleStats ple_stats() const;
    // Vision tower: forwards run (one per image; run_tower_and_scatter), patches and merged tokens fed through
    // them, seconds spent (upload + tower + scatter, the span the journal line times), the longest single run.
    // Safe to read from another thread.
    struct VisionStats {
        int64_t images = 0, patches = 0, merged_tokens = 0;
        double tower_seconds = 0, tower_max_seconds = 0;
    };
    VisionStats vision_stats() const;
    // Hint for decode with drafts: ids are the tokens the next forward will start with (after the n-gram history
    // at pos()); the PLE n-gram rows of ids[first, size) go into the table's row cache on a background thread now,
    // so that forward's gather finds them there instead of reading them while the GPU waits. The engine calls it as
    // each token becomes known - the sampled token before drafting, each draft while the next one is drafted.
    // While a forward_verify awaits keep / drop, ids start where the verify started (the engine prefetches a
    // rejection's corrected token before keep_verify_prefix). Changes no state the forward reads (only the cache's
    // contents); a no-op for a source without a row cache.
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first);
    void prefetch_ple(int seq, const std::vector<int32_t> &ids, int64_t first);  // slot seq's ids (the above: seq 0)
    // Hint for a long prompt in chunks: the ids the next forward() will run. The next forward() then gathers their
    // PLE n-gram rows from the table on a worker thread during its own GPU work (after its PLE layer), and the one
    // after uses them - if its ids and n-gram history match exactly (anything else, e.g. a reset or restore in
    // between, just gathers anew). Applies to the next forward() only; empty clears it. 0..max_tokens() ids.
    void set_lookahead(std::vector<int32_t> next_ids);
    bool has_mtp() const { return mtp_; }
    // The MTP head's logits [vocab] FP32 for the token at pos() - the one the last forward's logits chose - i.e.
    // its draft of the token at pos() + 1. Doesn't advance pos() or change the
    // state a later forward reads (its KV row at pos() is the one that forward writes again). Needs MTP on.
    // step > 0 chains the head (longer drafts): the token is step - 1's draft, the input streams step - 1's own
    // output, the position pos() + step; each step must follow step - 1 with no forward, restore or reset between.
    // A chain step's KV row is the draft's guess; the trunk's catch-up writes the real one when the position runs.
    std::vector<float> forward_mtp(int32_t token_id, int64_t step = 0);
    // The draft on a slot (feat/multi-sequence): the same on seq's state. forward_mtp is seq 0.
    std::vector<float> forward_mtp_seq(int seq, int32_t token_id, int64_t step = 0);
    // forward_mtp's draft reduced on the GPU (kernels/mtp_pick): the best id (lowest on a tie), its logit, the
    // runner-up's (the multiset second; -inf for one logit) and whether any logit was NaN - serve/sampler.hpp top2()
    // on forward_mtp's logits, bit for bit on non-NaN rows - with one small copy back instead of all mtp_vocab()
    // logits. The same state effects as forward_mtp.
    struct MtpTop2 {
        int32_t best = 0;
        float best_v = 0, second_v = 0;
        bool nan = false;
    };
    MtpTop2 forward_mtp_top2(int32_t token_id, int64_t step = 0);
    MtpTop2 forward_mtp_top2_seq(int seq, int32_t token_id, int64_t step = 0);  // the draft's pick on a slot
    // The draft's vocabulary: forward_mtp scores only ids [0, n) (the LM head's first n rows - BPE ids run roughly
    // by frequency) and returns n logits. Default: the whole vocabulary. 1..vocab.
    void set_mtp_vocab(int64_t n);
    int64_t mtp_vocab() const { return mtp_vocab_; }

private:
    const Qwen4ExpModel &m_;
    // One sequence state per serving slot (feat/multi-sequence): what the server runs
    // today is one; batched forwards address them by run. seqs_ is built at the top of the constructor, before
    // any st() use.
    std::vector<std::unique_ptr<Qwen4ExpSeqState>> seqs_;
    // The state a single-sequence call runs on (the batched API takes them by run).
    void reset_seq_state(Qwen4ExpSeqState &s);  // one state back to position 0 (reset's per-slot part)
    // The paged arrays of state sq and their byte ranges that positions [0, end) use: f(buffer, ranges) per array.
    void for_paged(Qwen4ExpSeqState &sq, int64_t end, const std::function<void(PagedDeviceBuffer &,
                   const std::vector<PagedDeviceBuffer::Range> &)> &f) const;
    // Maps the pages sq's arrays need for positions [0, end) and a little past (for_paged's headroom for tile
    // over-reads) - every writer calls it before its kernels: prepare_positions (forward bodies, keep_verify_prefix,
    // run_mtp), each run of a multi-run batched forward, import_state - after kv_budget_ (or the floor check) allowed
    // the bytes.
    void ensure_rows(Qwen4ExpSeqState &sq, int64_t end);
    std::function<void(uint64_t)> kv_budget_;
    static uint64_t mem_available_bytes();
    // The slot allocation guard's floor: a unified-memory machine with MemAvailable near 0.5 GB hard-hangs.
    static constexpr uint64_t kSlotFloorBytes = 4ull << 30;
    Qwen4ExpSeqState &st() { return *seqs_.front(); }
    const Qwen4ExpSeqState &st() const { return *seqs_.front(); }
    int64_t capacity_, max_tokens_, cap_blocks_ = 0;
    PrefillMath prefill_math_;
    PrecisionOptions precision_;
    hipStream_t stream_ = nullptr;
    // KvFormat::Q8 (the states' k_cache / v_cache empty then): kv_new_ [max_tokens][2W] the forward's new
    // K | V rows before kv8_commit; kv8_err_ the fp16-overflow flag; per attention layer the 8-bit storage
    // (kernels/kv8.hpp, for kv8_cap_ rows: the capacity rounded up to whole groups) lives in the state.
    KvFormat kv_format_ = KvFormat::Act;
    int64_t kv8_cap_ = 0;
    // An exported state's 8-bit K/V of one layer: the groups [from / 32, n / 32) (codes and params of K, then of V)
    // and the tail (its rows past n % 32 exported as zeros). kv8_copy moves them between the host cursor and the
    // layer's storage / `tail` (to_host: export, else import) and records the unused tail bytes for zeroing.
    size_t kv8_state_bytes(int64_t n, int64_t from) const;
    void kv8_copy(const Qwen4ExpSeqState &sq, uint8_t *&cur, size_t layer, int64_t n, int64_t from,
                  const DeviceBuffer<uint8_t> &tail, bool to_host, std::vector<std::pair<uint8_t *, size_t>> *unused);
    DeviceBuffer<uint8_t> kv_new_;
    // lg_src_: the batched forward's compacted logits rows - each run's last nL rows gathered contiguous so
    // the LM head's weights are read once for the whole batch. kMaxLogits rows, decode-size (a few MB).
    DeviceBuffer<uint8_t> lg_src_;
    // WMMA prefill attention over the 8-bit cache reads a staged copy (kernels/kv8.hpp Kv8View::staged_k): one
    // layer's K | V rows [0, kv8_cap_) (K in F16, V in BF16), re-filled per layer and call (kv8_dequant; kv8_stage_pays). Empty
    // unless the session has the 8-bit cache, WMMA prefill and BF16 activations, and forwards of kKv8StageMinTokens
    // or more.
    DeviceBuffer<uint8_t> kv8_stage_;
    DeviceBuffer<uint32_t> kv8_err_;
    // Workspaces [max_tokens, ...], activation dtype unless noted.
    DeviceBuffer<int32_t> ids_, route_ids_;
    DeviceBuffer<uint8_t> x_, u_, y_, sh_y_, proj_, qkv_, core_, gnorm_, gu_, hh_, ple_e_, ple_kv_, ple_hist_buf_;
    DeviceBuffer<float> h_, w_in_, inv_, beta_, g_, router_logits_, route_coef_, ple_sigma_, attn_ws_, logits_;
    DeviceBuffer<uint32_t> expert_err_, router_err_;
    // MTP (with mtp_ on) workspaces [max_tokens, ...]: the embedding and its norm [d], the normed streams [H*d],
    // fc_embedding's output [d], the combined streams [H*d]; mtp_ones_ [max_tokens, H] = 1 (the per-stream
    // broadcast add). The head's own K/V cache, keys, tails, streams and chain buffers live in the state (a chain
    // step swaps mtp_x_ against the state's mtp_chain_prev_, so only one state may draft at a time).
    bool mtp_ = false;
    int64_t mtp_vocab_ = 0;  // set to the vocabulary by the constructor
    DeviceBuffer<uint8_t> mtp_emb_, mtp_norm_emb_, mtp_norm_hid_, mtp_proj_emb_, mtp_x_;
    DeviceBuffer<float> mtp_ones_;
    // Bumped by every mtp_input (the one writer of mtp_x_'s rows). A draft chain records it after each step
    // (Qwen4ExpSeqState::mtp_chain_xver); a chain step whose stamp is stale was interrupted by another
    // draft / forward overwriting the shared mtp_x_ and is refused.
    uint64_t mtp_x_version_ = 0;
    // The MTP layer's input for the batch rows [row_base, row_base + T): token ids_[row_base + t] with the
    // streams after the position before (prev0 for t = 0, x_ row row_base + t - 1 after it).
    void mtp_input(int64_t row_base, int64_t T, const void *prev0);
    // forward_mtp / forward_mtp_top2's shared body: the head's layer and the LM head's logits into logits_ (queued on
    // stream_; broken_ set until the caller's read-back succeeds, which then calls mtp_done(seq, step)).
    void run_mtp(int seq, int32_t token_id, int64_t step);
    void mtp_done(int seq, int64_t step);
    DeviceBuffer<uint8_t> mtp_pick_;  // kernels::MtpPick on the device (allocated on the first forward_mtp_top2)
    // Grouped experts for multi-token forwards (kernels/moe_grouped): the route grouping and combine's FP32
    // per-slot dots. Allocated when max_tokens reaches kGroupedMinTokens.
    // QSA selection past dense_key_limit(): scores [max_tokens, cap_blocks] F32, the kept blocks [max_tokens,
    // budget / 4] and their counts, qsa_topk's workspace, gathered attention's error word. Allocated when the
    // capacity reaches past the limit.
    DeviceBuffer<float> qsa_scores_;
    DeviceBuffer<int32_t> qsa_sel_, qsa_nsel_;
    DeviceBuffer<uint8_t> qsa_topk_ws_;
    DeviceBuffer<uint32_t> attn_err_;
    size_t qsa_topk_ws_bytes_ = 0;
    DeviceBuffer<float> hc_ws_;  // hc_mix_down_wmma's partials (WMMA prefill)
    size_t hc_ws_bytes_ = 0;
    DeviceBuffer<int32_t> group_ws_;
    DeviceBuffer<float> group_partial_;
    size_t group_ws_bytes_ = 0;
    DeviceBuffer<kernels::EmbeddingError> emb_err_;
    // ---------------------------------------- interleaved mRoPE (vision Phase A), per-state position data in
    // Qwen4ExpSeqState; here the per-forward upload machinery (text-only forwards touch none of it - not one
    // byte uploaded, the cost is two int64 adds). The upload array covers a forward's block span
    // [4*(pos/4), pos+T): the jobs read it at an offset (their rows start at pos), qsa_block_keys needs the
    // block's first key's triple, which can be a tail row before pos.
    DeviceBuffer<int64_t> pos3_dev_;         // the upload array, 3 * capacity_ int64, allocated on first use
    // The spans prepare_positions() last prepared (see .cpp): mk_block_'s rows start at 4*(seq/4), mk_jobs_'s at
    // seq; both null while the last prepared forward is on today's scalar path.
    kernels::MropePos mk_block_{}, mk_jobs_{};
    std::vector<int64_t> stage3_;            // host staging for the upload (no allocation on the scalar path)
    // Upload probe for tests (STRIX_POS_PROBE=<dir> env, read once): appends one UploadTrace per prepared span
    // per forward to <dir>/uploads.bin - lets a test see exactly what the kernels would receive, including that
    // a text-only forward prepares nothing. Null when unset (the normal path; the only per-forward extra is a
    // pointer test).
    struct UploadTrace {
        int64_t seq, n;
        int64_t block_base, block_rows;  // -1: scalar path (null)
        int64_t jobs_base, jobs_rows;
    };
    void trace_upload(const UploadTrace &p) const;
    // Point pos3_dev_ at the forward's block span [4*(seq/4), seq+n) (uploading whenever a window row differs
    // from the scalar values or the session has a rope_delta) and set mk_block_ / mk_jobs_. Called by every
    // forward body (trunk + MTP parts, after its pending rows joined the history), by keep_verify_prefix and
    // by run_mtp; the scalar kernels' angle base gets rope_delta added by the callers.
    void prepare_positions(Qwen4ExpSeqState &sq, int64_t seq, int64_t n);
    // ---------------------------------------- image forward (vision Phase B): the tower's scratch and the
    // one image's tower run + scatter. The scratch starts empty and only ever serves an image forward -
    // a text-only session allocates nothing for it (the text path stays byte-identical, upload_probe
    // included). Sized for the largest grid seen (grow-on-larger; a grid past the attention cap is
    // refused before anything is touched), so a session that has seen a 1080p screenshot holds ~280 MB
    // of tower scratch at the 8192-patch cap - the accepted cost of vision = on.
    DeviceBuffer<uint8_t> vis_pv_, vis_x_, vis_conv_, vis_ln_, vis_proj_, vis_qkv_, vis_mlp_, vis_merg_,
        vis_merged_;
    DeviceBuffer<int32_t> vis_idx_, vis_rows_;
    DeviceBuffer<float> vis_w_;
    int64_t vis_patches_ = 0;  // the scratch's patch count (0: never allocated)
    void ensure_vision_scratch(int64_t grid_h, int64_t grid_w);
    // Validates the image against the model and the id stream, runs the tower, and scatters its rows
    // over the embedding's rows in x_ at the image_token_id positions. Called from forward_impl right
    // after the embedding launch.
    void run_tower_and_scatter(const std::vector<int32_t> &ids, const ImageInput &image);
    // Pinned: the error words at 0 (router, experts, attention - 3 words each), the embedding slot at
    // kReadbackEmbAt, the logits from kReadbackLogitsAt. read_back fills and checks it at the end of a forward.
    static constexpr size_t kReadbackEmbAt = 64, kReadbackLogitsAt = 128;  // error words [0, 12): 48 bytes
    PinnedHostBuffer readback_;
    // n_cand_rows > 0: the candidates (cand_dev_: [rows, kLogitCands] then rows NaN words) come back in place of the
    // logits, into cand_host_.
    std::vector<float> read_back(size_t n_floats, const char *what, int64_t n_cand_rows = 0);
    int64_t cand_n_valid_ = 0;  // want_candidates() for the next forward; 0 = full rows
    // Structured-output masks for the next forward (want_candidates): mask_rows_ rows of mask_words_ in mask_host_
    // (pinned), uploaded into mask_dev_ by that forward. The pinned buffer is written again only after the event of
    // its last upload completed (an ungated pinned-buffer reuse corrupted transfers on gfx1151, measured). Both
    // allocated on the first masked forward.
    int64_t mask_rows_ = 0, mask_words_ = 0;
    PinnedHostBuffer mask_host_;
    DeviceBuffer<uint32_t> mask_dev_;
    hipEvent_t mask_uploaded_ = nullptr;
    bool mask_upload_recorded_ = false;
    DeviceBuffer<uint8_t> cand_dev_, cand_ws_;
    size_t cand_ws_bytes_ = 0;
    Candidates cand_host_;
    size_t attn_ws_bytes_ = 0;
    bool broken_ = false;
    // PLE rows on the host (runtime/ngram_table gather -> the activation dtype), in pinned buffers [max_tokens, ple_e]
    // for an async upload. Two, so the next forward's gather (set_lookahead) fills one while this forward's upload
    // reads the other; a buffer is written again only after the forward that uploaded from it ended with its stream
    // sync - the pinned-reuse gate (as for mask_host_ above). Declared before ple_pending_: its worker is joined
    // first.
    PinnedHostBuffer ple_host_[2];
    struct PleGather {
        bool active = false;
        std::vector<int32_t> ids;  // the tokens gathered for
        PleHistory hist;           // the n-gram history before them
        int buf = 0;               // into ple_host_[buf]
        std::future<void> done;    // get() rethrows a gather failure
    };
    PleGather ple_pending_;              // started for set_lookahead's ids
    std::vector<int32_t> ple_lookahead_;  // set_lookahead's ids, for the next forward to start gathering
    // Small forwards' gathers (decode, MTP verify) run on one persistent thread, started when the forward starts, so
    // they overlap the embedding and layer 0 on the GPU instead of stalling it at the PLE layer - without a thread
    // per token (std::async per token cost decode ~0.2 ms). Jobs run in order; the gathers go in one at a
    // time (submit_alone). Declared after ple_host_ (and ple_pending_), so it's joined before the buffers it writes
    // are freed.
    class PleWorker {
    public:
        PleWorker();
        ~PleWorker();  // runs the queued jobs first
        PleWorker(const PleWorker &) = delete;
        PleWorker &operator=(const PleWorker &) = delete;
        std::future<void> submit_alone(std::function<void()> job);  // waits until no job is queued or running
        bool post(std::function<void()> job, size_t max_queued);    // false (not queued) with max_queued waiting

    private:
        void loop();
        std::mutex mu_;
        std::condition_variable cv_;
        std::deque<std::packaged_task<void()>> jobs_;
        bool busy_ = false, stop_ = false;
        std::thread thread_;
    };
    PleWorker ple_worker_;
    // Where the gathers' time goes (ple_stats): on the worker / in async gathers, and how long forwards waited for
    // rows at the PLE layer - the part the GPU sits idle for.
    std::atomic<int64_t> ple_gathers_{0}, ple_gather_ns_{0}, ple_waits_{0}, ple_wait_ns_{0}, ple_wait_max_ns_{0};
    std::atomic<int64_t> ple_prefetches_{0}, ple_prefetch_ns_{0}, ple_prefetch_skipped_{0}, ple_prefetch_failed_{0};
    // Vision tower counters (vision_stats): images (tower forwards), patches, merged tokens, total / max ns.
    std::atomic<int64_t> vision_images_{0}, vision_patches_{0}, vision_merged_{0}, vision_ns_{0}, vision_max_ns_{0};
    // prefetch_ple's jobs: their own thread, so a prefetch never delays a gather's start (the table's lock orders
    // the two: whichever runs second finds the rows cached). Declared after the counters it writes.
    PleWorker ple_prefetcher_;
    enum class PleRun { Worker, Async };
    std::future<void> start_ple_gather(std::vector<int64_t> rows, int64_t T, PinnedHostBuffer &dst, PleRun run);
    void drop_ple_pending();  // waits for a pending gather (its failure ignored) and forgets it
    // The seq of the first state with a pending verify (-1: none) - the gate every forward / draft checks:
    // the pending verify's replay state lives in the shared workspaces.
    int64_t pending_verify_seq() const;
    // The per-state halves of the verify mechanism, shared by the single-sequence and batched paths:
    // verify_prep allocates the state's spare set, keeps the small states, and records the before-state (the
    // caller then sets st.gdn_to_spare and runs the forward); keep / drop are the resolution bodies.
    void verify_prep(Qwen4ExpSeqState &sq);
    void keep_verify_full(Qwen4ExpSeqState &sq);
    void keep_verify_prefix_seq(Qwen4ExpSeqState &sq, int64_t rows);
    void drop_verify_seq(Qwen4ExpSeqState &sq);
};

}  // namespace strix
