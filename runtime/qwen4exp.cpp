#include "runtime/qwen4exp.hpp"

#include "runtime/ngram_table.hpp"

#include "formats/q4_from_q8.hpp"

#include "common/bf16.hpp"
#include "common/hip_check.hpp"
#include "common/trace.hpp"
#include "kernels/attention.hpp"
#include "kernels/gdn.hpp"
#include "kernels/hc.hpp"
#include "kernels/hc_wmma.hpp"
#include "kernels/linear.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/mtp_pick.hpp"
#include "kernels/ple.hpp"
#include "kernels/qsa.hpp"
#include "kernels/vision.hpp"
#include "kernels/residual.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace strix {

using kernels::Act;

namespace {

const std::string kLm = "model.language_model.";
constexpr int64_t kHD = kernels::kGdnHeadDim;

size_t es(Act a) { return a == Act::F32 ? 4 : 2; }

// (BF16 bit conversions: strix::bf16_bits_to_f32 / f32_to_bf16_bits, common/bf16.hpp)

Qwen4ExpModel::Hc load_hc(const StrixwDevice &w, const std::string &p, const Qwen4ExpDims &D, bool inject) {
    const int64_t n4 = D.H * D.d;
    return {w.qw(p + "mix_down", {D.r + (inject ? D.H : 0), n4}), w.qw_chunk_major(p + "mix_up", {n4, D.r}),
            w.f32(p + "hc_norm.weight", {n4})};
}

}  // namespace

Qwen4ExpModel::Qwen4ExpModel(const std::string &weights, const std::string &ngram, Act act,
                             bool allow_truncated, int64_t ngram_cache_rows, float yarn_factor,
                             WeightPlacement placement, const std::string &vision)
    : act_(act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "Qwen4ExpModel: unsupported activation dtype ", (int)act);
    STRIX_CHECK(std::isfinite(yarn_factor) && yarn_factor >= 1.0f && yarn_factor <= 8.0f, "Qwen4ExpModel: yarn_factor = ",
                yarn_factor, ", expected 1 (RoPE as trained) .. 8");
    w_ = std::make_unique<StrixwDevice>(weights, 8, placement);
    const StrixwFile &f = w_->file();
    Qwen4ExpDims &D = dims_;
    while (f.find(kLm + "layers." + std::to_string(D.layers) + ".attn_hyper_connection.mix_down")) ++D.layers;
    STRIX_CHECK(D.layers >= 1, "Qwen4ExpModel: no decoder layers in '", weights, "'");
    std::string prefix;  // "0,1,..,k-1": what a truncated conversion's layers metadata reads
    for (int64_t i = 0; i < D.layers; ++i) prefix += (i ? "," : "") + std::to_string(i);
    const std::string &layers_meta = f.meta("layers");
    truncated_ = layers_meta != "all";
    STRIX_CHECK(f.meta("arch") == "qwen4exp" && f.meta("globals") == "yes" &&
                    (layers_meta == "all" || (allow_truncated && layers_meta == prefix)),
                "Qwen4ExpModel: '", weights, "' is arch '", f.meta("arch"), "', layers '", layers_meta, "', globals '",
                f.meta("globals"), "' - needs a qwen4exp conversion with the globals and every layer",
                allow_truncated ? " (or the first k: --layers 0,..,k-1)" : " (tools/convert_qwen4exp without --layers)");
    // Shared expert stacked as #512 or separate: the tensors decide; the metadata, where present, must agree.
    shared_separate_ = f.find(kLm + "layers.0.mlp.shared_expert.gate_up") != nullptr;
    if (f.meta().count("shared_expert"))
        STRIX_CHECK(f.meta("shared_expert") == (shared_separate_ ? "separate" : "stacked"), "Qwen4ExpModel: '", weights,
                    "' has shared_expert metadata '", f.meta("shared_expert"), "' but layer 0's shared expert is ",
                    shared_separate_ ? "separate" : "stacked");
    const int64_t n4 = D.H * D.d, Estack = D.experts + (shared_separate_ ? 0 : 1);
    const auto &emb_info = f.get(kLm + "embed_tokens.weight");
    if (emb_info.encoding == StrixwEncoding::Q8RowMajor) {
        embed_q8_ = w_->q8(kLm + "embed_tokens.weight", {D.vocab, D.d});
    } else {
        embed_ = w_->bf16(kLm + "embed_tokens.weight", {D.vocab, D.d});
    }
    lm_head_ = w_->qw("lm_head.weight", {D.vocab, D.d});
    final_ = load_hc(*w_, kLm + "hyper_connection_mixer.", D, false);
    D.yarn_factor = yarn_factor;
    if (yarn_factor > 1.0f) {
        const kernels::YarnRope yarn = kernels::rope_yarn(D.rope_theta, D.rot, yarn_factor, D.trained_positions);
        inv_freq_ = DeviceBuffer<float>::from_host(yarn.inv_freq, "inv_freq (YaRN)");
        D.rope_scale = yarn.cos_sin_scale;
    } else {
        inv_freq_ = DeviceBuffer<float>::from_host(kernels::rope_inv_freq(D.rope_theta, D.rot), "inv_freq");
        D.rope_scale = 1.0f;
    }
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kLm + "layers." + std::to_string(i) + ".";
        const bool attn = f.find(L + "self_attn.qkv") != nullptr;
        STRIX_CHECK(attn != (f.find(L + "linear_attn.in_proj") != nullptr), "Qwen4ExpModel: layer ", i,
                    " must have exactly one of self_attn.qkv / linear_attn.in_proj");
        D.is_attention.push_back(attn);
        Layer l;
        l.hc_attn = load_hc(*w_, L + "attn_hyper_connection.", D, true);
        l.hc_mlp = load_hc(*w_, L + "mlp_hyper_connection.", D, true);
        if (attn) {
            const std::string A = L + "self_attn.";
            l.qkv = w_->qw(A + "qkv", {D.astride, D.d});
            l.o_proj = w_->qw(A + "o_proj.weight", {D.d, D.hq * D.hd});
            l.q_norm = w_->f32(A + "q_norm.weight", {D.hd});
            l.k_norm = w_->f32(A + "k_norm.weight", {D.hd});
            l.idx_q_norm = w_->f32(A + "indexer.q_layernorm.weight", {D.idx_d});
            l.idx_k_norm = w_->f32(A + "indexer.k_layernorm.weight", {D.idx_d});
        } else {
            const std::string G = L + "linear_attn.";
            l.in_proj = w_->qw(G + "in_proj", {D.gstride, D.d});
            l.out_proj = w_->qw(G + "out_proj.weight", {D.d, D.gz});
            l.conv_w = w_->f32(G + "conv1d.weight", {D.conv_c, 1, kernels::kGdnConvWidth});
            l.A_log = w_->f32(G + "A_log", {D.gv});
            l.dt_bias = w_->f32(G + "dt_bias", {D.gv});
            l.gdn_norm = w_->f32(G + "norm.weight", {kHD});
        }
        l.gate_up = w_->qw(L + "mlp.experts_gate_up", {Estack * 2 * D.inter, D.d});
        l.down = w_->qw(L + "mlp.experts_down", {Estack * D.d, D.inter});
        l.router = w_->bf16(L + "mlp.router", {D.experts + 1, D.d});  // + the shared expert's gate, either way
        const bool sep = f.find(L + "mlp.shared_expert.gate_up") != nullptr;
        STRIX_CHECK(sep == shared_separate_, "Qwen4ExpModel: layer ", i, "'s shared expert is ",
                    sep ? "separate" : "stacked", " but layer 0's is ", shared_separate_ ? "separate" : "stacked");
        if (sep) {
            l.shared_gate_up = w_->qw(L + "mlp.shared_expert.gate_up", {2 * D.inter, D.d});
            l.shared_down = w_->qw(L + "mlp.shared_expert.down_proj.weight", {D.d, D.inter});
        }
        if (f.find(L + "ple.kv_proj")) {
            STRIX_CHECK(D.ple_layer < 0, "Qwen4ExpModel: PLE at layers ", D.ple_layer, " and ", i, "; expected one");
            D.ple_layer = i;
            const std::string P = L + "ple.";
            l.ple_kv = w_->qw(P + "kv_proj", {D.ple_ld, D.ple_e});
            l.ple_conv = w_->f32(P + "conv1d.weight", {n4, 1, D.ple_taps});
            l.ple_norm_key = w_->f32(P + "norm_key.weight", {n4});
            l.ple_norm_query = w_->f32(P + "norm_query.weight", {n4});
            l.ple_norm_conv = w_->f32(P + "norm_conv.weight", {n4});
            if (std::filesystem::is_regular_file(ngram)) {
                auto table = std::make_unique<NgramTableRows>(ngram, ngram_cache_rows);
                const std::string &want = f.meta("source_fingerprint"), &got = table->file().info().source_fingerprint;
                STRIX_CHECK(got == want, "Qwen4ExpModel: n-gram table '", ngram, "' was converted from checkpoint ", got,
                            ", the weights '", weights, "' from ", want, " - convert both from the same checkpoint");
                ngram_ = std::move(table);
            } else {
                STRIX_CHECK(std::filesystem::is_directory(ngram), "Qwen4ExpModel: n-gram source '", ngram,
                            "' is neither an ngram.table file nor the HF checkpoint directory");
                STRIX_CHECK(ngram_cache_rows == NgramTableRows::kDefaultCacheRows, "Qwen4ExpModel: ngram_cache_rows = ",
                            ngram_cache_rows, " with the checkpoint directory '", ngram,
                            "' as the n-gram source - only a table file has a row cache");
                ngram_ = std::make_unique<NgramRowsFromShards>(ngram, P + "ple_embedding.");
            }
            STRIX_CHECK(ngram_->row_dim() * 16 == D.ple_e, "Qwen4ExpModel: 16 n-gram rows of ", ngram_->row_dim(),
                        " don't make the PLE input width ", D.ple_e);
        }
        layers_.push_back(l);
    }
    has_mtp_ = f.find("mtp.fc_embedding.weight") != nullptr;
    if (has_mtp_) {
        mtp_.fc_embedding = w_->qw("mtp.fc_embedding.weight", {D.d, D.d});
        mtp_.fc_hidden = w_->qw("mtp.fc_hidden.weight", {D.d, D.d});
        mtp_.pre_fc_norm_embedding = w_->f32("mtp.pre_fc_norm_embedding.weight", {D.d});
        mtp_.pre_fc_norm_hidden = w_->f32("mtp.pre_fc_norm_hidden.weight", {n4});
        mtp_.hc_mixer = load_hc(*w_, "mtp.hyper_connection_mixer.", D, false);
        const std::string M = "mtp.layers.0.";
        Layer &l = mtp_.layer;
        l.hc_attn = load_hc(*w_, M + "attn_hyper_connection.", D, true);
        l.hc_mlp = load_hc(*w_, M + "mlp_hyper_connection.", D, true);
        const std::string A = M + "self_attn.";
        l.qkv = w_->qw(A + "qkv", {D.astride, D.d});
        l.o_proj = w_->qw(A + "o_proj.weight", {D.d, D.hq * D.hd});
        l.q_norm = w_->f32(A + "q_norm.weight", {D.hd});
        l.k_norm = w_->f32(A + "k_norm.weight", {D.hd});
        l.idx_q_norm = w_->f32(A + "indexer.q_layernorm.weight", {D.idx_d});
        l.idx_k_norm = w_->f32(A + "indexer.k_layernorm.weight", {D.idx_d});
        l.gate_up = w_->qw(M + "mlp.experts_gate_up", {Estack * 2 * D.inter, D.d});
        l.down = w_->qw(M + "mlp.experts_down", {Estack * D.d, D.inter});
        l.router = w_->bf16(M + "mlp.router", {D.experts + 1, D.d});
        if (shared_separate_) {
            l.shared_gate_up = w_->qw(M + "mlp.shared_expert.gate_up", {2 * D.inter, D.d});
            l.shared_down = w_->qw(M + "mlp.shared_expert.down_proj.weight", {D.d, D.inter});
        }
    }
    // The image tower (vision): a second file, loaded last (it's the optional part), with the side-file
    // fingerprint rule VisionTower applies to its own index; the one cross-check that spans the files is
    // here - the merger's output width must be the trunk's embedding width, or the scatter has no place.
    if (!vision.empty()) {
        vision_ = std::make_unique<VisionTower>(vision, f, act);
        STRIX_CHECK(vision_->dims().out_hidden == D.d, "Qwen4ExpModel: vision tower '", vision_, "' outputs ",
                    vision_->dims().out_hidden, " columns, the trunk's embed width is ", D.d,
                    " - the merged rows wouldn't fit the embedding rows they replace");
    }
}

const VisionTower &Qwen4ExpModel::vision() const {
    STRIX_CHECK(vision_ != nullptr, "Qwen4ExpModel: no vision tower - construct with the vision.strixw path "
                "to load one (image forwards refuse without it, there is no fallback path)");
    return *vision_;
}

void Qwen4ExpModel::make_draft_head_q4(int64_t rows, int64_t G, int threads) {
    const char *fn = "Qwen4ExpModel::make_draft_head_q4";
    STRIX_CHECK(has_mtp_, fn, ": the model has no MTP head - nothing drafts");
    STRIX_CHECK(draft_head_.bits == 0, fn, ": a draft head copy of ", draft_head_.N(), " rows was already made (call once)");
    STRIX_CHECK(lm_head_.bits != 0, fn, ": the LM head isn't loaded");
    const int64_t N = lm_head_.N(), K = lm_head_.K();
    STRIX_CHECK(N == dims_.vocab, fn, ": the LM head has ", N, " rows, expected the vocabulary's ", dims_.vocab);
    STRIX_CHECK(rows >= 1 && rows <= N, fn, ": rows ", rows, ", expected 1..", N, " (the LM head's rows)");
    STRIX_CHECK(q4_group_size_supported(G), fn, ": group size ", G, ", expected 32, 64 or 128");
    STRIX_CHECK(K % G == 0, fn, ": the LM head's K = ", K, " is not a multiple of group size ", G);
    STRIX_CHECK(threads >= 1 && threads <= 64, fn, ": threads ", threads, ", expected 1..64");

    if (lm_head_.bits == 4) {  // already Q4: the draft reads the loaded head (a row prefix of it)
        STRIX_CHECK(lm_head_.q4.G == G, fn, ": the LM head is Q4 with group size ", lm_head_.q4.G, ", asked for ", G,
                    " - a Q4 head is used as loaded, not requantized");
        draft_head_ = lm_head_;
        draft_head_.q4.N = rows;
        return;
    }
    STRIX_CHECK(lm_head_.bits == 8, fn, ": the LM head has ", lm_head_.bits, " bits, expected 8 (requantized) or 4 (used ",
                "as loaded)");

    // Read the Q8 rows back (row-major: the first `rows` rows are a prefix of each array).
    const Q8DeviceView &v = lm_head_.q8;
    STRIX_CHECK(v.q && v.scale && v.minv && v.K == K && v.G >= 1 && K % v.G == 0, fn, ": the Q8 LM head view is ",
                "incomplete (codes ", (const void *)v.q, ", scales ", (const void *)v.scale, ", mins ",
                (const void *)v.minv, ", K ", v.K, ", G ", v.G, ")");
    Q8Weight q8;
    q8.N = rows, q8.K = K, q8.G = v.G;
    q8.q.resize((size_t)(rows * K));
    q8.scale.resize((size_t)(rows * (K / v.G)));
    q8.minv.resize(q8.scale.size());
    STRIX_HIP_CHECK(hipMemcpy(q8.q.data(), v.q, q8.q.size(), hipMemcpyDeviceToHost), fn, ": read back ", q8.q.size(),
                    " bytes of Q8 codes");
    STRIX_HIP_CHECK(hipMemcpy(q8.scale.data(), v.scale, q8.scale.size() * 2, hipMemcpyDeviceToHost), fn, ": read back ",
                    q8.scale.size(), " Q8 scales");
    STRIX_HIP_CHECK(hipMemcpy(q8.minv.data(), v.minv, q8.minv.size() * 2, hipMemcpyDeviceToHost), fn, ": read back ",
                    q8.minv.size(), " Q8 mins");

    const Q4Weight q4 = quantize_q4_from_q8(q8, G, threads);
    draft_q4_ = std::make_unique<Q4Device>(Q4Device::upload(q4, "lm_head.mtp_draft_q4"));
    draft_q4_bytes_ = q4.bytes();
    draft_head_ = qweight(draft_q4_->view());
    STRIX_CHECK(draft_head_.N() == rows && draft_head_.K() == K, fn, ": the uploaded copy is [", draft_head_.N(), ", ",
                draft_head_.K(), "], expected [", rows, ", ", K, "]");
}

int64_t make_served_mtp_draft_head(Qwen4ExpModel &model, int64_t mtp_vocab) {
    const int64_t vocab = model.dims().vocab;
    STRIX_CHECK(mtp_vocab >= 0 && mtp_vocab <= vocab, "make_served_mtp_draft_head: mtp_vocab ", mtp_vocab, ", expected 0..",
                vocab, " (0 = the whole vocabulary)");
    const int64_t rows = mtp_vocab == 0 ? vocab : mtp_vocab;
    model.make_draft_head_q4(rows, /*G=*/64, /*threads=*/16);
    return rows;
}

const Qwen4ExpModel::Layer &Qwen4ExpModel::layer(int64_t i) const {
    STRIX_CHECK(i >= 0 && i < (int64_t)layers_.size(), "Qwen4ExpModel::layer: ", i, " outside [0, ", layers_.size(), ")");
    return layers_[(size_t)i];
}

const NgramRowSource &Qwen4ExpModel::ngram_rows() const {
    STRIX_CHECK(ngram_ != nullptr, "Qwen4ExpModel: no PLE layer, so no n-gram rows");
    return *ngram_;
}

std::string PrecisionOptions::describe() const {
    return std::string("gdn=") + (gdn == kernels::DrPrecision::Fast ? "fast" : gdn == kernels::DrPrecision::Half ? "half" : gdn == kernels::DrPrecision::HalfU ? "half-u" : "exact") +
           " attention=" + (attention == kernels::AttnPrecision::Fast ? "fast" : "exact");
}

kernels::AttnPrecision parse_attention_precision(const std::string &v) {
    if (v == "exact") return kernels::AttnPrecision::Exact;
    if (v == "fast") return kernels::AttnPrecision::Fast;
    STRIX_FAIL("attention-precision: '", v, "', expected exact (P as BF16 hi + lo) or fast (P as one BF16 term)");
}

kernels::DrPrecision parse_gdn_precision(const std::string &v) {
    if (v == "exact") return kernels::DrPrecision::Exact;
    if (v == "fast") return kernels::DrPrecision::Fast;
    if (v == "half") return kernels::DrPrecision::Half;
    if (v == "half-u") return kernels::DrPrecision::HalfU;
    STRIX_FAIL("gdn-precision: '", v, "', expected exact (BF16 hi + lo products), fast (one BF16 product), half (one F16 "
               "product) or half-u (half, the U-path products as F16 hi + lo)");
}

void Qwen4ExpSession::set_precision(const PrecisionOptions &p) {
    STRIX_CHECK(p.gdn == kernels::DrPrecision::Exact || p.gdn == kernels::DrPrecision::Fast || p.gdn == kernels::DrPrecision::Half ||
                    p.gdn == kernels::DrPrecision::HalfU,
                "Qwen4ExpSession::set_precision: unknown gdn precision ", (int)p.gdn);
    STRIX_CHECK(p.attention == kernels::AttnPrecision::Exact || p.attention == kernels::AttnPrecision::Fast,
                "Qwen4ExpSession::set_precision: unknown attention precision ", (int)p.attention);
    precision_ = p;
}

const char *kv_format_name(KvFormat f) {
    STRIX_CHECK(f == KvFormat::Act || f == KvFormat::Q8 || f == KvFormat::Q8H, "kv_format_name: unknown KvFormat ", (int)f);
    switch (f) {
        case KvFormat::Q8: return "q8";
        case KvFormat::Q8H: return "q8h";
        case KvFormat::Act: return "bf16";
    }
    return "unknown";
}

KvFormat parse_kv_format(const std::string &v) {
    if (v == "bf16") return KvFormat::Act;
    if (v == "q8") return KvFormat::Q8;
    if (v == "q8h") return KvFormat::Q8H;
    STRIX_FAIL("parse_kv_format: '", v, "', expected bf16, q8, or q8h");
}

namespace {
inline bool is_q8_format(KvFormat f) { return f == KvFormat::Q8 || f == KvFormat::Q8H; }
}  // namespace

Qwen4ExpSession::Qwen4ExpSession(const Qwen4ExpModel &model, int64_t capacity, int64_t max_tokens,
                                 PrefillMath prefill_math, bool mtp, KvFormat kv)
    : m_(model), capacity_(capacity), max_tokens_(max_tokens), prefill_math_(prefill_math), kv_format_(kv),
      mtp_(mtp) {
    STRIX_CHECK(kv == KvFormat::Act || kv == KvFormat::Q8 || kv == KvFormat::Q8H, "Qwen4ExpSession: unknown KV format ", (int)kv);
    STRIX_CHECK(prefill_math == PrefillMath::F32 || prefill_math == PrefillMath::WmmaBf16,
                "Qwen4ExpSession: unknown prefill math ", (int)prefill_math);
    STRIX_CHECK(!mtp || model.has_mtp(), "Qwen4ExpSession: MTP asked for, but the loaded weights have no MTP head "
                "(no mtp.fc_embedding.weight)");
    const Qwen4ExpDims &D = m_.dims();
    STRIX_CHECK(capacity >= 1 && capacity <= (1ll << 24), "Qwen4ExpSession: capacity = ", capacity, ", expected 1..2^24");
    STRIX_CHECK(capacity <= D.max_positions(), "Qwen4ExpSession: capacity = ", capacity, " is past the model's ",
                D.max_positions(), " positions (", D.trained_positions, " trained x YaRN factor ", D.yarn_factor,
                ") - positions beyond it aren't meaningful; load the model with a larger yarn_factor");
    STRIX_CHECK(max_tokens >= 1 && max_tokens <= capacity && max_tokens <= 16384, "Qwen4ExpSession: max_tokens = ",
                max_tokens, ", expected 1..min(capacity = ", capacity, ", 16384)");
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "Qwen4ExpSession: stream");
    const size_t e = es(m_.act());
    const int64_t M = max_tokens, n4 = D.H * D.d, A = D.top_k + 1;
    cap_blocks_ = capacity / kernels::kQsaBlock + 1;
    const bool q8 = kv == KvFormat::Q8 || kv == KvFormat::Q8H;
    kv8_cap_ = (capacity + kernels::kKv8Group - 1) / kernels::kKv8Group * kernels::kKv8Group;
    const int64_t W = D.hkv * D.hd;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "layer " + std::to_string(i);
        const bool attn = D.is_attention[(size_t)i];
        k_cache_.emplace_back(attn && !q8 ? DeviceBuffer<uint8_t>((size_t)(capacity * W) * e, "K cache " + tag)
                                          : DeviceBuffer<uint8_t>());
        v_cache_.emplace_back(attn && !q8 ? DeviceBuffer<uint8_t>((size_t)(capacity * W) * e, "V cache " + tag)
                                          : DeviceBuffer<uint8_t>());
        kv8_store_.emplace_back(attn && q8 ? DeviceBuffer<uint8_t>(kernels::kv8_bytes(kv8_cap_, W), "8-bit K/V " + tag)
                                           : DeviceBuffer<uint8_t>());
        for (int k = 0; k < 2; ++k)
            kv8_tail_[k].emplace_back(attn && q8 ? DeviceBuffer<uint8_t>(kernels::kv8_tail_bytes(W, m_.act()),
                                                                         "K/V tail " + std::to_string(k) + " " + tag)
                                                 : DeviceBuffer<uint8_t>());
        block_keys_.emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(cap_blocks_ * D.idx_d) * e, "block keys " + tag)
                                      : DeviceBuffer<uint8_t>());
        for (int k = 0; k < 2; ++k)
            tail_[k].emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "indexer tail " + tag)
                                       : DeviceBuffer<uint8_t>());
        conv_state_.emplace_back(!attn ? DeviceBuffer<uint8_t>((size_t)((kernels::kGdnConvWidth - 1) * D.conv_c) * e,
                                                               "conv state " + tag)
                                       : DeviceBuffer<uint8_t>());
        rec_state_.emplace_back(!attn ? DeviceBuffer<float>((size_t)(D.gv * kHD * kHD), "recurrent state " + tag)
                                      : DeviceBuffer<float>());
    }
    auto act_buf = [&](int64_t per_token, const std::string &name) {
        return DeviceBuffer<uint8_t>((size_t)(M * per_token) * e, name);
    };
    ids_ = DeviceBuffer<int32_t>((size_t)M, "token ids");
    x_ = act_buf(n4, "residual streams");
    u_ = act_buf(D.d, "sublayer input");
    y_ = act_buf(D.d, "sublayer output");
    if (m_.shared_separate()) sh_y_ = act_buf(D.d, "shared expert output");
    proj_ = act_buf(std::max(D.gstride, D.astride), "projection");
    qkv_ = act_buf(D.conv_c, "gdn q|k|v");
    STRIX_CHECK(D.gz == D.gv * kHD, "Qwen4ExpSession: GDN z width ", D.gz, " != ", D.gv, " value heads x ", kHD,
                " (rmsnorm_gated reads z in place as gv heads of kHD a token)");
    core_ = act_buf(std::max(D.gz, D.hq * D.hd), "mixer core");
    gnorm_ = act_buf(D.gz, "gdn normed");
    gu_ = act_buf(A * 2 * D.inter, "experts gate|up");
    hh_ = act_buf(A * D.inter, "experts swiglu");
    h_ = DeviceBuffer<float>((size_t)(M * D.r), "hc low-rank");
    w_in_ = DeviceBuffer<float>((size_t)(M * D.H), "hc w_in");
    inv_ = DeviceBuffer<float>((size_t)(M * D.H), "hc inverse rms");
    beta_ = DeviceBuffer<float>((size_t)(M * D.gv), "gdn beta");
    g_ = DeviceBuffer<float>((size_t)(M * D.gv), "gdn g");
    router_logits_ = DeviceBuffer<float>((size_t)(M * (D.experts + 1)), "router logits");
    route_ids_ = DeviceBuffer<int32_t>((size_t)(M * A), "expert ids");
    route_coef_ = DeviceBuffer<float>((size_t)(M * A), "expert coefficients");
    logits_ = DeviceBuffer<float>((size_t)(std::min(M, kMaxLogits) * D.vocab), "logits");
    readback_ = PinnedHostBuffer(kReadbackLogitsAt + logits_.size() * 4, "forward read-back (errors + logits)");
    {
        const int64_t rows = std::min(M, kMaxLogits);
        cand_dev_ = DeviceBuffer<uint8_t>((size_t)rows * (kernels::kLogitCands * sizeof(kernels::LogitCand) + 4),
                                          "logits candidates + NaN flags");
        cand_ws_bytes_ = kernels::logits_topk_workspace_bytes(rows, D.vocab);
        cand_ws_ = DeviceBuffer<uint8_t>(cand_ws_bytes_, "logits_topk workspace");
        STRIX_CHECK(cand_dev_.size() <= logits_.size() * 4, "Qwen4ExpSession: the candidates (", cand_dev_.size(),
                    " bytes) don't fit the logits read-back area (", logits_.size() * 4, ")");
    }
    if (D.ple_layer >= 0) {
        const int64_t S = (D.ple_taps - 1) * D.ple_dil;
        ple_state_ = DeviceBuffer<uint8_t>((size_t)(S * n4) * e, "PLE conv state");
        ple_e_ = act_buf(D.ple_e, "PLE n-gram rows");
        for (int k = 0; k < 2; ++k)
            ple_host_[k] = PinnedHostBuffer((size_t)(M * D.ple_e) * e, "PLE n-gram rows (host) " + std::to_string(k));
        ple_kv_ = act_buf(D.ple_ld, "PLE key|value");
        ple_hist_buf_ = DeviceBuffer<uint8_t>((size_t)((S + M) * n4) * e, "PLE history");
        ple_sigma_ = DeviceBuffer<float>((size_t)(M * D.H), "PLE sigma");
    }
    const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
    const bool qsa = capacity > D.dense_key_limit();
    const bool wmma_attn = prefill_math_ == PrefillMath::WmmaBf16 && m_.act() == Act::BF16;
    for (int64_t T = 1; T <= M; ++T) {
        const kernels::AttentionShape s{T, capacity - T, D.hq, D.hkv, D.hd, D.astride, 2 * D.hd};
        attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_workspace_bytes(s));
        if (qsa) attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_gathered_workspace_bytes(s, qsa_k));
        if (wmma_attn && T >= kWmmaMinTokens) {
            // The matrix-unit attention for prefill-sized forwards, dense at any depth below the limit.
            for (int64_t p0 : {(int64_t)0, std::min(capacity, D.dense_key_limit()) - T})
                if (p0 >= 0)
                    attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_wmma_workspace_bytes(
                                                                  {T, p0, D.hq, D.hkv, D.hd, D.astride, 2 * D.hd}));
            if (qsa) attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_gathered_wmma_workspace_bytes(s, qsa_k));
        }
    }
    if (qsa) {
        qsa_scores_ = DeviceBuffer<float>((size_t)(M * cap_blocks_), "QSA block scores");
        qsa_sel_ = DeviceBuffer<int32_t>((size_t)(M * qsa_k), "QSA kept blocks");
        qsa_nsel_ = DeviceBuffer<int32_t>((size_t)M, "QSA kept block counts");
        qsa_topk_ws_bytes_ = kernels::qsa_topk_workspace_bytes(M);
        qsa_topk_ws_ = DeviceBuffer<uint8_t>(qsa_topk_ws_bytes_, "QSA top-k workspace");
        STRIX_HIP_CHECK(hipMemset(qsa_topk_ws_.get(), 0, qsa_topk_ws_bytes_), "zero the QSA top-k workspace");
    }
    attn_err_ = DeviceBuffer<uint32_t>(3, "attention error");
    STRIX_HIP_CHECK(hipMemset(attn_err_.get(), 0, 12), "zero attention error");
    if (q8 && prefill_math_ == PrefillMath::WmmaBf16 && m_.act() == Act::BF16 && max_tokens >= kKv8StageMinTokens)
        kv8_stage_ = DeviceBuffer<uint8_t>((size_t)(2 * kv8_cap_ * W) * 2, "staged K | V (8-bit prefill)");
    if (q8) {
        kv_new_ = act_buf(2 * W, "new K | V rows");
        kv8_err_ = DeviceBuffer<uint32_t>(3, "8-bit K/V error");
        STRIX_HIP_CHECK(hipMemset(kv8_err_.get(), 0, 12), "zero 8-bit K/V error");
    }
    attn_ws_ = DeviceBuffer<float>(attn_ws_bytes_ / 4 + 1, "attention workspace");
    expert_err_ = DeviceBuffer<uint32_t>(3, "expert error");
    if (prefill_math_ == PrefillMath::WmmaBf16 && M >= kWmmaMinTokens) {
        hc_ws_bytes_ = kernels::hc_wmma_workspace_bytes(M, D.H, D.r + D.H);
        hc_ws_ = DeviceBuffer<float>(hc_ws_bytes_ / 4, "hc mix partials");
    }
    if (M >= kGroupedMinTokens) {
        const int64_t Estack = D.experts + (m_.shared_separate() ? 0 : 1);
        group_ws_bytes_ = kernels::moe_route_workspace_bytes(M, A, Estack);
        group_ws_ = DeviceBuffer<int32_t>(group_ws_bytes_ / 4, "expert grouping");
        group_partial_ = DeviceBuffer<float>((size_t)(M * A * D.d), "experts combine partials");
    }
    router_err_ = DeviceBuffer<uint32_t>(3, "router error");
    emb_err_ = DeviceBuffer<kernels::EmbeddingError>(1, "embedding error");
    STRIX_HIP_CHECK(hipMemset(expert_err_.get(), 0, 12), "zero expert error");
    STRIX_HIP_CHECK(hipMemset(router_err_.get(), 0, 12), "zero router error");
    mtp_vocab_ = D.vocab;
    if (mtp_) {
        mtp_emb_ = act_buf(D.d, "MTP embedding");
        mtp_norm_emb_ = act_buf(D.d, "MTP normed embedding");
        mtp_norm_hid_ = act_buf(n4, "MTP normed streams");
        mtp_proj_emb_ = act_buf(D.d, "MTP fc_embedding output");
        mtp_x_ = act_buf(n4, "MTP streams");
        mtp_prev_ = DeviceBuffer<uint8_t>((size_t)n4 * e, "MTP previous streams");
        k_cache_mtp_ = DeviceBuffer<uint8_t>((size_t)(capacity * D.hkv * D.hd) * e, "MTP K cache");
        v_cache_mtp_ = DeviceBuffer<uint8_t>((size_t)(capacity * D.hkv * D.hd) * e, "MTP V cache");
        block_keys_mtp_ = DeviceBuffer<uint8_t>((size_t)(cap_blocks_ * D.idx_d) * e, "MTP block keys");
        for (int k = 0; k < 2; ++k)
            tail_mtp_[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "MTP indexer tail " + std::to_string(k));
        // mtp_x_'s twin: a chain step swaps the two, so the step before's output is kept without a copy.
        mtp_chain_prev_ = act_buf(n4, "MTP chain previous streams");
        for (int k = 0; k < 2; ++k)
            mtp_chain_tail_[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "MTP chain indexer tail " + std::to_string(k));
        std::vector<float> ones((size_t)(M * D.H), 1.0f);
        mtp_ones_ = DeviceBuffer<float>(ones.size(), "MTP ones");
        STRIX_HIP_CHECK(hipMemcpy(mtp_ones_.get(), ones.data(), ones.size() * 4, hipMemcpyHostToDevice), "MTP ones");
    }
    reset();
}

Qwen4ExpSession::~Qwen4ExpSession() {
    drop_ple_pending();
    if (stream_) {
        (void)hipStreamSynchronize(stream_);  // no upload may still read a pinned buffer when it's freed
        (void)hipStreamDestroy(stream_);
    }
    if (mask_uploaded_) (void)hipEventDestroy(mask_uploaded_);
}

Qwen4ExpSession::PleWorker::PleWorker() : thread_([this] { loop(); }) {}

Qwen4ExpSession::PleWorker::~PleWorker() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();  // a job in flight finishes first
}

std::future<void> Qwen4ExpSession::PleWorker::submit_alone(std::function<void()> job) {
    STRIX_CHECK(job != nullptr, "Qwen4ExpSession::PleWorker::submit_alone: empty job");
    std::packaged_task<void()> task(std::move(job));
    std::future<void> f = task.get_future();
    {
        // One at a time: normally the forward that submitted the last job waited for it; after a forward that failed
        // midway it may still be running - and it writes the same pinned buffer - so wait for it.
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return jobs_.empty() && !busy_; });
        jobs_.push_back(std::move(task));
    }
    cv_.notify_all();
    return f;
}

bool Qwen4ExpSession::PleWorker::post(std::function<void()> job, size_t max_queued) {
    STRIX_CHECK(job != nullptr, "Qwen4ExpSession::PleWorker::post: empty job");
    STRIX_CHECK(max_queued >= 1, "Qwen4ExpSession::PleWorker::post: max_queued = 0");
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (jobs_.size() >= max_queued) return false;
        jobs_.emplace_back(std::move(job));
    }
    cv_.notify_all();
    return true;
}

void Qwen4ExpSession::PleWorker::loop() {
    for (;;) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) return;  // stop_, nothing left
            task = std::move(jobs_.front()), busy_ = true;
            jobs_.pop_front();
        }
        task();  // an exception goes into the future
        {
            std::lock_guard<std::mutex> lock(mu_);
            busy_ = false;
        }
        cv_.notify_all();  // a submit_alone waiting for this job
    }
}

Qwen4ExpSession::PleStats Qwen4ExpSession::ple_stats() const {
    PleStats s;
    s.gathers = ple_gathers_.load(), s.waits = ple_waits_.load();
    s.gather_seconds = (double)ple_gather_ns_.load() * 1e-9, s.wait_seconds = (double)ple_wait_ns_.load() * 1e-9;
    s.wait_max_seconds = (double)ple_wait_max_ns_.load() * 1e-9;
    s.prefetches = ple_prefetches_.load(), s.prefetch_skipped = ple_prefetch_skipped_.load();
    s.prefetch_failed = ple_prefetch_failed_.load(), s.prefetch_seconds = (double)ple_prefetch_ns_.load() * 1e-9;
    return s;
}

Qwen4ExpSession::VisionStats Qwen4ExpSession::vision_stats() const {
    VisionStats s;
    s.images = vision_images_.load(), s.patches = vision_patches_.load(), s.merged_tokens = vision_merged_.load();
    s.tower_seconds = (double)vision_ns_.load() * 1e-9;
    s.tower_max_seconds = (double)vision_max_ns_.load() * 1e-9;
    return s;
}

void Qwen4ExpSession::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    const int64_t T = (int64_t)ids.size();
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::prefetch_ple: ", T, " ids, expected 1..", max_tokens_);
    STRIX_CHECK(first >= 0 && first < T, "Qwen4ExpSession::prefetch_ple: first = ", first, ", expected 0..", T - 1);
    if (m_.dims().ple_layer < 0) return;
    // Rows of every id (the history runs through them), then the ones asked for. While a verify awaits its outcome
    // the ids start where the verify did (a rejection's kept prefix + the corrected token).
    std::vector<int64_t> tok(ids.begin(), ids.end()), rows((size_t)(T * 16));
    PleHistory h = verify_pending_ ? verify_hist_ : ple_hist_;
    ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), T, h, rows.data());
    rows.erase(rows.begin(), rows.begin() + first * 16);
    // A prefetch that doesn't keep up is worth nothing: past a few queued, skip (the gather reads the rows itself).
    constexpr size_t kMaxQueued = 8;
    const bool queued = ple_prefetcher_.post(
        [this, rows = std::move(rows)] {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                m_.ngram_rows().prefetch(rows.data(), (int64_t)rows.size());
            } catch (const std::exception &ex) {
                // Not the forward's failure: the gather that needs these rows reads them and reports its own.
                if (ple_prefetch_failed_++ < 10)
                    std::fprintf(stderr, "Qwen4ExpSession::prefetch_ple: prefetch failed (the gather retries): %s\n",
                                 ex.what());
            }
            ++ple_prefetches_;
            ple_prefetch_ns_ +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        },
        kMaxQueued);
    if (!queued) ++ple_prefetch_skipped_;
}

// Row ids [T, 16] -> ple_host_[buf] as the activation dtype: on the persistent worker (small forwards) or an async
// thread (prefill chunks, lookahead). The caller guarantees nothing reads the buffer (no upload from it pending)
// until the returned future is ready.
std::future<void> Qwen4ExpSession::start_ple_gather(std::vector<int64_t> rows, int64_t T, int buf, PleRun run) {
    const int64_t pe = m_.dims().ple_e;
    STRIX_CHECK(buf == 0 || buf == 1, "Qwen4ExpSession: PLE host buffer ", buf);
    STRIX_CHECK(T >= 1 && T <= max_tokens_ && (int64_t)rows.size() == T * 16, "Qwen4ExpSession: PLE gather of ", T,
                " tokens with ", rows.size(), " row ids, expected 1..", max_tokens_, " tokens x 16");
    STRIX_CHECK(ple_host_[buf].size() >= (size_t)(T * pe) * es(m_.act()), "Qwen4ExpSession: PLE host buffer ", buf,
                " holds ", ple_host_[buf].size(), " bytes, needs ", (size_t)(T * pe) * es(m_.act()));
    auto job = [this, rows = std::move(rows), T, pe, buf] {
        const auto t0 = std::chrono::steady_clock::now();
        const NgramRowSource &ng = m_.ngram_rows();
        if (m_.act() == Act::BF16) {
            ng.gather(rows.data(), (int64_t)rows.size(), static_cast<uint16_t *>(ple_host_[buf].get()));
        } else {
            std::vector<uint16_t> bits((size_t)(T * pe));
            ng.gather(rows.data(), (int64_t)rows.size(), bits.data());
            float *f = static_cast<float *>(ple_host_[buf].get());
            for (size_t k = 0; k < bits.size(); ++k) f[k] = bf16_bits_to_f32(bits[k]);
        }
        ++ple_gathers_;
        ple_gather_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    };
    if (run == PleRun::Worker) return ple_worker_.submit_alone(std::move(job));
    return std::async(std::launch::async, std::move(job));
}

void Qwen4ExpSession::drop_ple_pending() {
    if (ple_pending_.active && ple_pending_.done.valid()) {
        try {
            ple_pending_.done.get();
        } catch (...) {  // a hint's gather: its failure only matters to a forward that uses it
        }
    }
    ple_pending_ = PleGather{};
}

void Qwen4ExpSession::set_lookahead(std::vector<int32_t> next_ids) {
    STRIX_CHECK((int64_t)next_ids.size() <= max_tokens_, "Qwen4ExpSession::set_lookahead: ", next_ids.size(),
                " ids, expected 0..", max_tokens_);
    ple_lookahead_ = std::move(next_ids);
}

namespace {
bool triple_eq(const int64_t *a, const int64_t *b) { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }
bool triple_uniform(const int64_t *tr) { return tr[0] == tr[1] && tr[1] == tr[2]; }
std::string triple_str(const int64_t *tr) {
    return "(" + std::to_string(tr[0]) + ", " + std::to_string(tr[1]) + ", " + std::to_string(tr[2]) + ")";
}

// How the row after a history's final row continues, per the get_vision_position_ids rule the triples come from
// (the vision position-walk design note: image blocks fill block-major, w = w0 + col resetting to w0 at each block
// row; the first text row after a block is uniform at the block's max position + 1). `last` is the history's
// last row, `hist` its rows (nrows of them); returns what the next row must be / the accepted set):
//  - kUniform: uniform span - all three sections step by 1;
//  - kCell: last is an image row whose block row provably has more cells (the block row's width is derived
//    from the row before it, which the history holds, and not all of it is saved) - the next row is that
//    next cell (t, h, w + 1);
//  - kCell: the next row is the same block row's next cell (t, h, w + 1) - true while a block row runs,
//    whether or not more of the block is derivable;
//  - kEndOrImage: the saved image rows end at a complete block row but the block's height isn't derivable:
//    the next row is either the block's end - uniform at max(last row's h + 1, t + width), the max position
//    + 1 rule - or the first cell of a next block row (t, h + 1, w) whose w lies in the observed range. The
//    accepted set is `uniform` at exact / (t, h, w in [w_lo, w_hi]).
// There is no case that derives nothing: even a history that is one partial block row offers its next cell
// (only the block-end alternative needs the width - which is why a snapshot with such a tail can't
// persist a continuation value, see continuation_value_of).
enum class ContKind { kUniform, kCell, kEndOrImage };
struct Cont {
    ContKind kind = ContKind::kUniform;
    std::array<int64_t, 3> exact{};               // kUniform / kCell: the one valid continuation; kEndOrImage:
                                                  // the uniform block-end value
    int64_t t = 0, h = 0, w_lo = 0, w_hi = 0;     // kEndOrImage: accepted image steps (t, h, w in [w_lo, w_hi])
};
inline bool accepts(const Cont &c, const int64_t *got) {
    switch (c.kind) {
        case ContKind::kUniform:
        case ContKind::kCell: return triple_eq(got, c.exact.data());
        case ContKind::kEndOrImage:
            return triple_uniform(got) ? got[0] == c.exact[0]  // the block's end position
                                       : (got[0] == c.t && got[1] == c.h && got[2] >= c.w_lo && got[2] <= c.w_hi);
    }
    return false;
}
Cont continuation_of(const int64_t *last, const int64_t *hist, int64_t nrows) {
    Cont c{};
    if (triple_uniform(last)) {
        c.kind = ContKind::kUniform;
        c.exact = {last[0] + 1, last[1] + 1, last[2] + 1};
        return c;
    }
    // Walk back over the last row's block row (same t and h, w descending contiguously) to its start.
    int64_t k = nrows - 1;  // history index of the last row
    int64_t w_start = last[2];
    while (k > 0) {
        const int64_t *p = hist + (size_t)((k - 1) * 3);
        if (p[0] != last[0] || p[1] != last[1] || p[2] != w_start - 1) break;
        --k, w_start = p[2];
    }
    if (k == 0) {  // the whole history is one block row: the next cell (same block row) is derivable, the
        c.kind = ContKind::kCell;  // row's start / the width aren't (accepts only offers that one value;
        c.exact = {last[0], last[1], last[2] + 1};  // block-end steps need the width - kUnknown-grade states
        c.w_lo = c.w_hi = last[2] + 1;              // persist nothing, snapshot save throws on them)
        return c;
    }
    const int64_t saved = nrows - k;  // rows of the last block row that the history holds
    const int64_t width = hist[(size_t)((k - 1) * 3 + 2)] - w_start + 1;  // the row before ends its block row
    if (saved < width) {
        c.kind = ContKind::kCell;
        c.exact = {last[0], last[1], last[2] + 1};
        c.w_lo = c.w_hi = last[2] + 1;
        return c;
    }
    c.kind = ContKind::kEndOrImage;
    // Block end = max position + 1 over the block = max(h0 + height, w0 + width) = max(last h + 1, t + width):
    // h0 + height is the last block row's next step (known), w0 + width the last cell's next column.
    c.exact = {std::max(last[1] + 1, last[0] + width), std::max(last[1] + 1, last[0] + width),
               std::max(last[1] + 1, last[0] + width)};
    c.t = last[0], c.h = last[1] + 1, c.w_lo = w_start, c.w_hi = last[2];  // a next block row isn't ruled out
    return c;
}

// The single continuation value a persisted state carries (snapshot's first_cont_, the exported header's
// cont_*): kUniform / kEndOrImage take their one value (the block end - the only continuation a resume may
// use); kCell (the saved rows end inside a block row) throws - a state saved mid-block can't persist a
// resumable position (restore would refuse it anyway; failing at save states it where the cause is visible).
std::array<int64_t, 3> continuation_value_of(const int64_t *last, const int64_t *hist, int64_t nrows,
                                             const char *fn) {
    const Cont c = continuation_of(last, hist, nrows);
    STRIX_CHECK(c.kind != ContKind::kCell, fn, ": the saved rows end at ", triple_str(last), ", inside a block "
                "row - can't derive a continuation to persist (a snapshot taken after the image's last "
                "block row can resume; resuming mid-image isn't implemented)");
    return c.exact;
}
}  // namespace

void Qwen4ExpSession::set_positions(std::vector<int64_t> pos3) {
    const char *fn = "Qwen4ExpSession::set_positions";
    STRIX_CHECK(pos3.size() % 3 == 0, fn, ": ", pos3.size(), " values, expected a whole number of (t,h,w) triples");
    if (pos3.empty()) {  // clear: the next forward is on the scalar path again (any text-only call may do this)
        pending3_.clear();
        return;
    }
    const int64_t rows = (int64_t)pos3.size() / 3;
    STRIX_CHECK(!verify_pending_, fn, ": a forward_verify awaits keep_verify / drop_verify - the next forward's "
                "rows are undefined until then");
    STRIX_CHECK(pending3_.empty(), fn, ": positions for the next forward are already set (", pending3_.size() / 3,
                " rows) - forward them or clear with an empty vector");
    STRIX_CHECK(rows >= 1 && rows <= max_tokens_, fn, ": ", rows, " rows, expected 1..", max_tokens_,
                " (one forward of at most max_tokens rows)");
    STRIX_CHECK(pos_ + rows <= capacity_, fn, ": rows [", pos_, ", ", pos_ + rows, ") exceed the capacity ",
                capacity_);
    const int64_t max_pos = m_.dims().max_positions();
    for (int64_t r = 0; r < rows; ++r)
        for (int k = 0; k < 3; ++k) {
            const int64_t v = pos3[(size_t)(3 * r + k)];
            STRIX_CHECK(v >= 0 && v < max_pos, fn, ": row ", r, " section ", (k == 0 ? "t" : k == 1 ? "h" : "w"),
                        " = ", v, ", expected a position in [0, ", max_pos, ") (value ", 3 * r + k, " of ",
                        pos3.size(), ")");
        }
    // The span must continue the history exactly (the vision position-walk design note). Rows at or past the
    // position are the pending span's - a set_positions while a previous span is still un-forwarded was
    // refused above, and no other call advances pos_ with a span pending.
    // The span must continue the history exactly (the vision position-walk design note). While the history is
    // empty (a text-only session that never set positions, or one resumed after the image) there is nothing
    // to continue - the first interleaved span can start anywhere valid. Once non-empty it must reach pos_
    // (every forward extends it), and rows at or past pos_ are the pending span's.
    if (!pos3_host_.empty()) {
        STRIX_CHECK(pos3Hend() == pos_, fn, ": the triple history covers [", pos3_start_, ", ", pos3Hend(),
                    ") but the position is ", pos_, " (internal: every forward extends the history to pos_)");
        const int64_t *last = pos3_host_.data() + (size_t)((pos3Hend() - pos3_start_ - 1) * 3);
        const Cont want = continuation_of(last, pos3_host_.data(), (int64_t)pos3_host_.size() / 3);
        STRIX_CHECK(accepts(want, pos3.data()), fn, ": row ", pos_, " is ", triple_str(pos3.data()), " but the "
                    "history's row ", pos_ - 1, " ", triple_str(last),
                    " continues at " + triple_str(want.exact.data()) +
                        (want.kind == ContKind::kEndOrImage ? " (or an image step at the block edge)"
                                                            : std::string()));
    } else {
        // First interleaved span of a text-only (or after-image resumed) session: its first row continues the
        // uniform prefix, so it sits at the scalar continuation of the current position.
        const int64_t want[3] = {pos_ + rope_delta_, pos_ + rope_delta_, pos_ + rope_delta_};
        STRIX_CHECK(triple_eq(pos3.data(), want), fn, ": row ", pos_, " is ", triple_str(pos3.data()), " but the "
                    "uniform prefix continues at ", triple_str(want), " (the first interleaved row must sit at "
                    "the walk's current position ", pos_, " + delta)");
    }
    history_ever_interleaved_ = history_ever_interleaved_ || rope_delta_ != 0 || !pos3_host_.empty();  // (delta is
    // <= 0 by invariant - nonzero only after an image packed positions ahead of the sequence indices)
    pending3_ = std::move(pos3);
}

// Point the rope triple array at the forward's block span [4*(seq/4), seq+n) and fill mk_block_ / mk_jobs_.
// The span starts at the block boundary because qsa_block_keys ropes a completing block at its first key,
// which for the first block can be a tail row before seq; the jobs read rows [seq, seq+n) at the same
// upload's offset. Two independent decisions, both free for text-only sessions:
//  - jobs: null unless a row of [seq, seq+n) has a non-scalar triple - the jobs take the shifted scalar base
//    rope_pos(seq) otherwise (their kernel indexes rows relative to pos0, so the shift is safe there);
//  - block keys: null only while rope_delta == 0 and no window row differs. Their kernel derives block and
//    tail array indices from pos0 too, so a shifted scalar base would shift the block partition - once the
//    session has an image, the window's triples (which carry the shifted values) go up instead.
// Called once per forward (after its pending rows joined the history - its MTP part runs the same rows), by
// keep_verify_prefix and by run_mtp. seq is a sequence index - the scalar kernels' angle positions get
// rope_delta added by the callers, never here.
void Qwen4ExpSession::prepare_positions(int64_t seq, int64_t n) {
    const char *fn = "Qwen4ExpSession::prepare_positions";
    STRIX_CHECK(n >= 1 && seq >= 0 && seq + n <= capacity_, fn, ": span [", seq, ", ", seq + n, ") outside [0, ",
                capacity_, ")");
    STRIX_CHECK(pending3_.empty(), fn, ": set_positions rows for the forward are not in the history yet "
                "(forward consumes them first)");
    STRIX_CHECK(rope_delta_ <= 0, fn, ": rope_delta_ = ", rope_delta_, ", expected <= 0 (internal: an image "
                "block's h*w rows consume max(h,w) positions, so positions only ever pack ahead of sequence "
                "indices)");
    mk_block_ = kernels::MropePos{}, mk_jobs_ = kernels::MropePos{};
    const int64_t t0 = (seq / kernels::kQsaBlock) * kernels::kQsaBlock;  // the block span's first row
    const int64_t hstart = pos3_start_, hend = pos3Hend();  // the history holds rows [hstart, hend)
    bool span_diff = false;  // a row of the forward's span [seq, seq+n) with a non-scalar triple
    bool window_diff = false;  // ... or anywhere in the block window [t0, seq+n)
    for (int64_t r = std::max(t0, hstart); r < std::min(hend, seq + n); ++r) {
        const int64_t *tr = pos3_host_.data() + (size_t)((r - hstart) * 3);
        const int64_t v = rope_pos(r);
        const bool differs = !(tr[0] == v && tr[1] == v && tr[2] == v);
        span_diff = span_diff || (r >= seq && differs);
        window_diff = window_diff || differs;
    }
    const bool uploaded = window_diff || rope_delta_ != 0;  // (delta is <= 0 once a block packed positions ahead)
    UploadTrace probe{seq, n, -1, -1, -1, -1};
    auto finish = [&] {
        if (uploaded) {  // hand the kernels their views (the calls re-validate base / rows / alignment)
            mk_block_ = kernels::MropePos{pos3_dev_.get(), seq + n - t0, t0};
            probe.block_base = t0, probe.block_rows = seq + n - t0;
        }
        if (span_diff) {
            mk_jobs_ = kernels::MropePos{pos3_dev_.get() + (size_t)((seq - t0) * 3), n, seq};
            probe.jobs_base = seq, probe.jobs_rows = n;
        }
        trace_upload(probe);
    };
    if (!uploaded) return finish();  // span_diff implies window_diff implies uploaded - pure text: both null
    // Stage [t0, seq + n): scalar-materialized rows below / past the history, real history rows in between
    // (rows below a restored tail history are the scalar continuation - restore refuses image resumes).
    const int64_t rows = seq + n - t0;
    stage3_.clear();
    stage3_.reserve((size_t)(rows * 3));
    const int64_t hs = std::max(t0, hstart), he = std::max(hs, std::min(hend, seq + n));
    auto push_scalar = [&](int64_t r) {
        const int64_t v = rope_pos(r);
        stage3_.insert(stage3_.end(), {v, v, v});
    };
    for (int64_t r = t0; r < hs; ++r) push_scalar(r);
    if (hs < he)
        stage3_.insert(stage3_.end(), pos3_host_.begin() + (hs - hstart) * 3, pos3_host_.begin() + (he - hstart) * 3);
    for (int64_t r = he; r < seq + n; ++r) push_scalar(r);
    STRIX_CHECK((int64_t)stage3_.size() == rows * 3, fn, ": staged ", stage3_.size() / 3, " rows for the span [", t0,
                ", ", seq + n, ")");
    if (pos3_dev_.size() == 0)  // (get() throws on the empty buffer - size() is the allocation test)
        pos3_dev_ = DeviceBuffer<int64_t>((size_t)capacity_ * 3, "mRoPE triples");
    STRIX_HIP_CHECK(hipMemcpyAsync(pos3_dev_.get(), stage3_.data(), (size_t)(rows * 3) * sizeof(int64_t),
                                   hipMemcpyHostToDevice, stream_),
                    fn, ": uploading rows [", t0, ", ", seq + n, ")");
    finish();
}

void Qwen4ExpSession::trace_upload(const UploadTrace &p) const {
    static const char *dir = std::getenv("STRIX_POS_PROBE");  // read once; tests set it before the session
    if (dir == nullptr || dir[0] == 0) return;
    const std::string path = std::string(dir) + "/uploads.bin";
    FILE *f = std::fopen(path.c_str(), "ab");
    STRIX_CHECK(f != nullptr, "Qwen4ExpSession::trace_upload: cannot append to '", path, "'");
    const bool ok = std::fwrite(&p, sizeof(p), 1, f) == 1;
    std::fclose(f);
    STRIX_CHECK(ok, "Qwen4ExpSession::trace_upload: short write to '", path, "'");
}

void Qwen4ExpSession::reset() {
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::reset");
    // The kernels' error words are one-way flags (set by atomicCAS, never cleared by the readers or the
    // report that throws): left set, every later forward's read_back re-reports this stale error - the
    // engine resets after a failed request expecting a clean slate, so a reset must clear them (the
    // embedding slot has its own per-forward reset kernel).
    STRIX_HIP_CHECK(hipMemset(router_err_.get(), 0, 12), "Qwen4ExpSession::reset: zero router error");
    STRIX_HIP_CHECK(hipMemset(expert_err_.get(), 0, 12), "Qwen4ExpSession::reset: zero expert error");
    STRIX_HIP_CHECK(hipMemset(attn_err_.get(), 0, 12), "Qwen4ExpSession::reset: zero attention error");
    if (is_q8_format(kv_format_))
        STRIX_HIP_CHECK(hipMemset(kv8_err_.get(), 0, 12), "Qwen4ExpSession::reset: zero 8-bit K/V error");
    pos_ = 0, tail_cur_ = 0, mtp_tail_cur_ = 0, broken_ = false, mtp_chain_step_ = -1, verify_pending_ = false;  // GDN states reset on the first call (reset_state); KV overwritten
    rope_delta_ = 0, history_ever_interleaved_ = false, verify_delta_before_ = 0, verify_ever_before_ = false,
        verify_hist_start_before_ = 0;
    pos3_start_ = 0;
    pos3_host_.clear(), pending3_.clear(), verify_hist_before_.clear();
    mk_block_ = kernels::MropePos{}, mk_jobs_ = kernels::MropePos{};
    imported_at_ = -1;
    if (mtp_)  // position 0's MTP row pairs its token with no trunk state: zeros
        STRIX_HIP_CHECK(hipMemset(mtp_prev_.get(), 0, mtp_prev_.size()), "Qwen4ExpSession: zero MTP previous streams");
    ple_lookahead_.clear();  // a pending gather stays: the next forward's ids / history decide whether it's used
    rewind_to_.push_back(0);
    if (m_.dims().ple_layer >= 0) {
        STRIX_HIP_CHECK(hipMemset(ple_state_.get(), 0, ple_state_.size()), "Qwen4ExpSession: zero PLE state");
        ple_hist_ = PleHistory::fresh(m_.ngram_rows().hash_params());
    }
}

Qwen4ExpSnapshot Qwen4ExpSession::make_snapshot() const {
    const Qwen4ExpDims &D = m_.dims();
    Qwen4ExpSnapshot s;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "snapshot layer " + std::to_string(i);
        const size_t l = (size_t)i;
        s.conv_.emplace_back(conv_state_[l].size() ? DeviceBuffer<uint8_t>(conv_state_[l].size(), "conv state " + tag)
                                                    : DeviceBuffer<uint8_t>());
        s.rec_.emplace_back(rec_state_[l].size() ? DeviceBuffer<float>(rec_state_[l].size(), "recurrent state " + tag)
                                                  : DeviceBuffer<float>());
        s.tail_.emplace_back(tail_[0][l].size() ? DeviceBuffer<uint8_t>(tail_[0][l].size(), "indexer tail " + tag)
                                                 : DeviceBuffer<uint8_t>());
        s.kv8_tail_.emplace_back(kv8_tail_[0][l].size() ? DeviceBuffer<uint8_t>(kv8_tail_[0][l].size(), "K/V tail " + tag)
                                                         : DeviceBuffer<uint8_t>());
    }
    if (ple_state_.size()) s.ple_state_ = DeviceBuffer<uint8_t>(ple_state_.size(), "snapshot PLE conv state");
    if (mtp_) {
        s.mtp_tail_ = DeviceBuffer<uint8_t>(tail_mtp_[0].size(), "snapshot MTP indexer tail");
        s.mtp_prev_ = DeviceBuffer<uint8_t>(mtp_prev_.size(), "snapshot MTP previous streams");
    }
    return s;
}

namespace {
template <typename T>
void copy_state(const DeviceBuffer<T> &from, const DeviceBuffer<T> &to, hipStream_t stream, const char *what) {
    STRIX_CHECK(from.size() == to.size(), "Qwen4ExpSession snapshot: ", what, " has ", from.size(), " elements, expected ",
                to.size(), " (a snapshot from another session?)");
    if (from.size())
        STRIX_HIP_CHECK(hipMemcpyAsync(to.get(), from.get(), from.size() * sizeof(T), hipMemcpyDeviceToDevice, stream),
                        "Qwen4ExpSession snapshot: copying ", what);
}
}  // namespace

void Qwen4ExpSession::save(Qwen4ExpSnapshot &s) {
    STRIX_CHECK(!broken_, "Qwen4ExpSession::save: an earlier call failed midway; the state is not worth saving");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::save: a forward_verify awaits keep_verify / drop_verify");
    STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::save: set_positions holds ", pending3_.size() / 3,
                " interleaved rows the position doesn't include yet - forward them first");
    STRIX_CHECK(s.conv_.size() == conv_state_.size(), "Qwen4ExpSession::save: snapshot has ", s.conv_.size(),
                " layers, the session ", conv_state_.size(), " (make it with make_snapshot)");
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        copy_state(conv_state_[l], s.conv_[l], stream_, "conv state");
        copy_state(rec_state_[l], s.rec_[l], stream_, "recurrent state");
        copy_state(tail_[tail_cur_][l], s.tail_[l], stream_, "indexer tail");
        copy_state(kv8_tail_[tail_cur_][l], s.kv8_tail_[l], stream_, "K/V tail");
    }
    copy_state(ple_state_, s.ple_state_, stream_, "PLE conv state");
    copy_state(tail_mtp_[mtp_tail_cur_], s.mtp_tail_, stream_, "MTP indexer tail");
    copy_state(mtp_prev_, s.mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::save");
    s.ple_hist_ = ple_hist_;
    s.pos_ = pos_;
    s.epoch_ = rewind_to_.size() - 1;
    // mRoPE: rope_delta and the last three history rows (the continuation rule a resume needs; whole-history
    // resume isn't a thing until image serving refuses less, vision Phase A - and restore
    // refuses an image-row tail outright, so what a snapshot really carries is the delta and the proof the
    // tail is text). Throws when the tail can't derive a continuation (a saved row in the middle of a single
    // image block row with nothing to derive the width from) - guessing is not an option.
    s.rope_delta_ = rope_delta_;
    const int64_t hlen = (int64_t)pos3_host_.size() / 3;
    s.ntriples_ = std::min<int64_t>(3, hlen);
    s.triples_.assign(pos3_host_.end() - (size_t)(s.ntriples_ * 3), pos3_host_.end());
    s.first_cont_ = hlen > 0
                        ? continuation_value_of(pos3_host_.data() + (size_t)((hlen - 1) * 3), pos3_host_.data(),
                                                hlen, "Qwen4ExpSession::save")
                        : std::array<int64_t, 3>{pos_ + rope_delta_, pos_ + rope_delta_, pos_ + rope_delta_};
}

bool Qwen4ExpSession::can_restore(const Qwen4ExpSnapshot &s) const {
    if (s.pos_ < 0 || s.conv_.size() != conv_state_.size() || s.epoch_ >= rewind_to_.size()) return false;
    for (size_t e = s.epoch_ + 1; e < rewind_to_.size(); ++e)
        if (rewind_to_[e] < s.pos_) return false;
    // Every rewind since was to >= s.pos_, and forwards only write at or past the position they start from, so
    // KV and block keys below s.pos_ are the ones the snapshot's state was built with.
    return true;
}

void Qwen4ExpSession::restore(const Qwen4ExpSnapshot &s) {
    STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::restore: set_positions holds ", pending3_.size() / 3,
                " interleaved rows the snapshot doesn't cover - forward them or clear first");
    STRIX_CHECK(s.pos_ >= 0, "Qwen4ExpSession::restore: the snapshot was never saved");
    // Image-position resumes are refused (vision Phase A decision): a snapshot saved inside or
    // at the end of an image span can't continue as scalar rows. The block end (the max-position + 1 rule)
    // can't be told from a next image block row when only one block row of the block is saved - those resume
    // with the three carried triples below, the exact block-end continuation stays the pure-scalar one.
    const bool resume_ok = !s.has_triples() || triple_uniform(s.triples() + (s.ntriples_ - 1) * 3);
    const bool needs_tail =
        !resume_ok && (s.first_continuation()[0] == s.first_continuation()[1] &&
                       s.first_continuation()[1] == s.first_continuation()[2]) &&
        s.first_continuation()[0] == s.pos() + s.rope_delta();
    STRIX_CHECK(resume_ok || needs_tail,
                "Qwen4ExpSession::restore: the snapshot at position ", s.pos(), " was saved inside an image "
                "span (its last saved row ", triple_str(s.triples() + (s.ntriples_ - 1) * 3),
                " continues at ", triple_str(s.first_continuation()),
                ", not uniformly) - resuming mid-image isn't implemented; rewind or restart the prompt");
    verify_pending_ = false;  // a restore replaces all state a pending verify could have left
    STRIX_CHECK(can_restore(s), "Qwen4ExpSession::restore: the session rewound below the snapshot's position ", s.pos_,
                " since it was saved, so its KV cache no longer matches");
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        copy_state(s.conv_[l], conv_state_[l], stream_, "conv state");
        copy_state(s.rec_[l], rec_state_[l], stream_, "recurrent state");
        copy_state(s.tail_[l], tail_[tail_cur_][l], stream_, "indexer tail");
        copy_state(s.kv8_tail_[l], kv8_tail_[tail_cur_][l], stream_, "K/V tail");
    }
    copy_state(s.ple_state_, ple_state_, stream_, "PLE conv state");
    copy_state(s.mtp_tail_, tail_mtp_[mtp_tail_cur_], stream_, "MTP indexer tail");
    copy_state(s.mtp_prev_, mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::restore");
    ple_hist_ = s.ple_hist_;
    pos_ = s.pos_;
    // mRoPE: the scalar continuation state; with a needs_tail resume also the last saved rows, so the first
    // forward's block window can rope its tail keys with them (rows below them never re-enter a window).
    rope_delta_ = s.rope_delta_;
    history_ever_interleaved_ = s.has_triples();
    pos3_host_.clear();
    if (needs_tail) pos3_host_ = s.triples_;
    pos3_start_ = pos_ - (int64_t)pos3_host_.size() / 3;
    STRIX_CHECK(pos3Hend() == pos_, "Qwen4ExpSession::restore: history [", pos3_start_, ", ", pos3Hend(),
                ") doesn't end at the restored position ", pos_, " (internal)");
    broken_ = false, mtp_chain_step_ = -1;
    rewind_to_.push_back(s.pos_);
}

size_t Qwen4ExpSession::state_bytes(int64_t n, int64_t from) const {
    STRIX_CHECK(n >= 0 && n <= capacity_, "Qwen4ExpSession::state_bytes: n = ", n, ", expected 0..", capacity_);
    STRIX_CHECK(from >= 0 && from <= n, "Qwen4ExpSession::state_bytes: from = ", from, ", expected 0..n = ", n);
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    // The rows [from, n) and the blocks [from / 4, n / 4) (see export_state).
    const size_t kv = 2 * (size_t)((n - from) * D.hkv * D.hd) * e,
                 keys = (size_t)((n / kernels::kQsaBlock - from / kernels::kQsaBlock) * D.idx_d) * e;
    size_t b = sizeof(PleHistory) + sizeof(Qwen4ExpStateHeader);
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l])
            b += (is_q8_format(kv_format_) ? kv8_state_bytes(n, from) : kv) + keys + tail_[0][l].size();
        else
            b += conv_state_[l].size() + rec_state_[l].size() * sizeof(float);
    }
    b += ple_state_.size();
    if (mtp_) b += kv + keys + tail_mtp_[0].size() + mtp_prev_.size();
    return b;
}

namespace {
// Copies between one device region and the next `bytes` of a host cursor, in either direction.
// The block keys of blocks [b0, b1): the cache is chunk-major (kernels/qsa.hpp: element d of block b at
// ((d / 8) * cap_blocks + b) * 8 + d % 8), so they're D / 8 strided rows of (b1 - b0) * 8 elements - packed on the
// host as [chunk][block][8], which doesn't depend on the session's capacity.
void block_keys_copy(uint8_t *&cur, const void *dev, int64_t b0, int64_t b1, int64_t cap_blocks, int64_t D, size_t e,
                     bool to_host, hipStream_t stream) {
    STRIX_CHECK(b0 >= 0 && b0 <= b1 && b1 <= cap_blocks, "Qwen4ExpSession block keys copy: blocks [", b0, ", ", b1,
                ") of ", cap_blocks);
    if (b1 == b0) return;
    const size_t row = (size_t)((b1 - b0) * 8) * e, pitch = (size_t)(cap_blocks * 8) * e, rows = (size_t)(D / 8);
    uint8_t *d = static_cast<uint8_t *>(const_cast<void *>(dev)) + (size_t)(b0 * 8) * e;
    if (to_host)
        STRIX_HIP_CHECK(hipMemcpy2DAsync(cur, row, d, pitch, row, rows, hipMemcpyDeviceToHost, stream),
                        "Qwen4ExpSession export: block keys");
    else
        STRIX_HIP_CHECK(hipMemcpy2DAsync(d, pitch, cur, row, row, rows, hipMemcpyHostToDevice, stream),
                        "Qwen4ExpSession import: block keys");
    cur += row * rows;
}

void host_copy(uint8_t *&cur, const void *dev, size_t bytes, bool to_host, hipStream_t stream, const char *what) {
    if (!bytes) return;
    if (to_host)
        STRIX_HIP_CHECK(hipMemcpyAsync(cur, dev, bytes, hipMemcpyDeviceToHost, stream), "Qwen4ExpSession export: ", what);
    else
        STRIX_HIP_CHECK(hipMemcpyAsync(const_cast<void *>(dev), cur, bytes, hipMemcpyHostToDevice, stream),
                        "Qwen4ExpSession import: ", what);
    cur += bytes;
}
}  // namespace

size_t Qwen4ExpSession::kv8_state_bytes(int64_t n, int64_t from) const {
    STRIX_CHECK(is_q8_format(kv_format_), "Qwen4ExpSession::kv8_state_bytes: the session's K/V is ",
                kv_format_name(kv_format_), ", not q8 or q8h");
    STRIX_CHECK(from >= 0 && from <= n && n <= kv8_cap_, "Qwen4ExpSession::kv8_state_bytes: rows [", from, ", ", n,
                ") outside the 8-bit capacity ", kv8_cap_);
    const Qwen4ExpDims &D = m_.dims();
    const int64_t W = D.hkv * D.hd, G = kernels::kKv8Group, ga = from / G, gb = n / G, rows = (gb - ga) * G;
    return (size_t)(rows * W) * 2 + (size_t)((gb - ga) * W) * 4 + (size_t)(rows * (W / kernels::kKv8Block)) * 4 +
           kernels::kv8_tail_bytes(W, m_.act());
}

void Qwen4ExpSession::kv8_copy(uint8_t *&cur, size_t layer, int64_t n, int64_t from, const DeviceBuffer<uint8_t> &tail,
                               bool to_host, std::vector<std::pair<uint8_t *, size_t>> *unused) {
    const Qwen4ExpDims &D = m_.dims();
    const int64_t W = D.hkv * D.hd, G = kernels::kKv8Group, cap = kv8_cap_, ga = from / G, gb = n / G;
    STRIX_CHECK(layer < kv8_store_.size() && kv8_store_[layer].size() == kernels::kv8_bytes(cap, W),
                "Qwen4ExpSession::kv8_copy: layer ", layer, " has no 8-bit K/V storage");
    STRIX_CHECK(tail.size() == kernels::kv8_tail_bytes(W, m_.act()), "Qwen4ExpSession::kv8_copy: tail of ",
                tail.size(), " bytes, expected ", kernels::kv8_tail_bytes(W, m_.act()));
    uint8_t *st = kv8_store_[layer].get();
    const size_t kc = (size_t)(cap * W), kp = (size_t)((cap / G) * W) * 4, nb = (size_t)(W / kernels::kKv8Block);
    host_copy(cur, st + (size_t)(ga * G * W), (size_t)((gb - ga) * G * W), to_host, stream_, "8-bit K codes");
    host_copy(cur, st + kc + (size_t)(ga * W) * 4, (size_t)((gb - ga) * W) * 4, to_host, stream_, "8-bit K params");
    host_copy(cur, st + kc + kp + (size_t)(ga * G * W), (size_t)((gb - ga) * G * W), to_host, stream_, "8-bit V codes");
    host_copy(cur, st + 2 * kc + kp + (size_t)(ga * G) * nb * 4, (size_t)((gb - ga) * G) * nb * 4, to_host, stream_,
              "8-bit V params");
    // The tail: rows [gb * G, n) valid in each half (K, V).
    const size_t es_ = es(m_.act()), half = (size_t)(G * W) * es_, valid = (size_t)((n - gb * G) * W) * es_;
    if (unused)
        for (int h = 0; h < 2; ++h) unused->emplace_back(cur + h * half + valid, half - valid);
    host_copy(cur, tail.get(), tail.size(), to_host, stream_, "K/V tail");
}

void Qwen4ExpSession::export_state(const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from) {
    STRIX_CHECK(out != nullptr, "Qwen4ExpSession::export_state: null output");
    STRIX_CHECK(can_restore(s), "Qwen4ExpSession::export_state: the snapshot (position ", s.pos(),
                ") is not valid for this session's KV cache any more");
    const int64_t n = s.pos();
    STRIX_CHECK(from >= 0 && from < n, "Qwen4ExpSession::export_state: from = ", from, ", expected 0..", n - 1,
                " (the snapshot's position - 1)");
    STRIX_CHECK(bytes == state_bytes(n, from), "Qwen4ExpSession::export_state: buffer ", bytes, " bytes, the state at ",
                n, from > 0 ? " (a delta from " + std::to_string(from) + ")" : std::string(), " is ", state_bytes(n, from));
    // Rows [from, n), blocks [from / 4, n / 4).
    const int64_t b0 = from / kernels::kQsaBlock, b1 = n / kernels::kQsaBlock;
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = out;
    // The mRoPE header first (state_bytes reserves it): the snapshot's continuation state - export is the
    // snapshot's bytes, not the session's live ones.
    Qwen4ExpStateHeader hdr;
    hdr.rope_delta = s.rope_delta_;
    hdr.cont_t = s.first_cont_[0], hdr.cont_h = s.first_cont_[1], hdr.cont_w = s.first_cont_[2];
    hdr.has_image = history_ever_interleaved_ ? 1 : 0;  // the snapshot's own flag would miss an after-image
                                                        // resume (its history was scalarized at restore)
    hdr.ntriples = s.has_triples() ? s.ntriples_ : 0;
    for (int64_t k = 0; k < hdr.ntriples * 3; ++k) hdr.triples[k] = s.triples_[(size_t)k];
    std::memcpy(cur, &hdr, sizeof(hdr));
    cur += sizeof(hdr);
    std::memcpy(cur, &s.ple_hist_, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    // An indexer tail holds 3 rows but only n % 4 are state; the rest is whatever an earlier forward left there
    // (after an MTP rejection: the rejected tokens' keys). Exported as zeros, so a state's bytes depend only on its
    // sequence. The zeroing waits for the copies.
    const size_t tail_valid = (size_t)((n % kernels::kQsaBlock) * D.idx_d) * e;
    std::vector<std::pair<uint8_t *, size_t>> tail_unused;
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            if (is_q8_format(kv_format_)) {
                kv8_copy(cur, l, n, from, s.kv8_tail_[l], true, &tail_unused);
            } else {
                host_copy(cur, k_cache_[l].get() + (size_t)from * row, kv, true, stream_, "K cache");
                host_copy(cur, v_cache_[l].get() + (size_t)from * row, kv, true, stream_, "V cache");
            }
            block_keys_copy(cur, block_keys_[l].get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
            STRIX_CHECK(s.tail_[l].size() >= tail_valid, "Qwen4ExpSession::export_state: indexer tail of ",
                        s.tail_[l].size(), " bytes, ", tail_valid, " valid");
            tail_unused.emplace_back(cur + tail_valid, s.tail_[l].size() - tail_valid);
            host_copy(cur, s.tail_[l].get(), s.tail_[l].size(), true, stream_, "indexer tail");
        } else {
            host_copy(cur, s.conv_[l].get(), s.conv_[l].size(), true, stream_, "conv state");
            host_copy(cur, s.rec_[l].get(), s.rec_[l].size() * sizeof(float), true, stream_, "recurrent state");
        }
    }
    host_copy(cur, s.ple_state_.get(), s.ple_state_.size(), true, stream_, "PLE conv state");
    if (mtp_) {
        const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
        host_copy(cur, k_cache_mtp_.get() + (size_t)from * row, kv, true, stream_, "MTP K cache");
        host_copy(cur, v_cache_mtp_.get() + (size_t)from * row, kv, true, stream_, "MTP V cache");
        block_keys_copy(cur, block_keys_mtp_.get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
        STRIX_CHECK(s.mtp_tail_.size() >= tail_valid, "Qwen4ExpSession::export_state: MTP indexer tail of ",
                    s.mtp_tail_.size(), " bytes, ", tail_valid, " valid");
        tail_unused.emplace_back(cur + tail_valid, s.mtp_tail_.size() - tail_valid);
        host_copy(cur, s.mtp_tail_.get(), s.mtp_tail_.size(), true, stream_, "MTP indexer tail");
        host_copy(cur, s.mtp_prev_.get(), s.mtp_prev_.size(), true, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::export_state");
    for (const auto &[at, len] : tail_unused) std::memset(at, 0, len);
    STRIX_CHECK((size_t)(cur - out) == bytes, "Qwen4ExpSession::export_state: wrote ", cur - out, " of ", bytes, " bytes");
}

void Qwen4ExpSession::import_state(const uint8_t *in, size_t bytes, int64_t n, int64_t from) {
    STRIX_CHECK(in != nullptr, "Qwen4ExpSession::import_state: null input");
    STRIX_CHECK(n >= 1 && n < capacity_, "Qwen4ExpSession::import_state: n = ", n, ", expected 1..", capacity_ - 1);
    STRIX_CHECK(from >= 0 && from < n, "Qwen4ExpSession::import_state: from = ", from, ", expected 0..", n - 1);
    STRIX_CHECK(bytes >= sizeof(Qwen4ExpStateHeader), "Qwen4ExpSession::import_state: ", bytes,
                " bytes, shorter than a state header (", sizeof(Qwen4ExpStateHeader), ")");
    // A delta lands on the base it was cut from: the rows below `from` must be that base's, imported just before.
    STRIX_CHECK(from == 0 || (pos_ == from && imported_at_ == from), "Qwen4ExpSession::import_state: a delta from ",
                from, " needs its base imported just before (position ", pos_, ", last import ended at ", imported_at_,
                ")");
    STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::import_state: set_positions holds ", pending3_.size() / 3,
                " interleaved rows the state doesn't cover - forward them or clear first");
    // The mRoPE header comes first; a blob without it (version 0, exported before this header existed) is
    // named as such and refused - never guessed at (no fallback path). Checked before the exact-size test so a
    // whole v0 entry (whose size differs by the header) says why, not "another model".
    Qwen4ExpStateHeader hdr;
    std::memcpy(&hdr, in, sizeof(hdr));
    const int64_t max_pos = m_.dims().max_positions();
    STRIX_CHECK(hdr.magic == Qwen4ExpStateHeader::kMagic && hdr.version == Qwen4ExpStateHeader::kVersion,
                "Qwen4ExpSession::import_state: blob has no mRoPE state header (magic 0x",
                std::hex, hdr.magic, " version ", std::dec, hdr.version, ") - a state exported before the header "
                "existed: drop the prompt cache entries made by that version");
    STRIX_CHECK(bytes == state_bytes(n, from), "Qwen4ExpSession::import_state: ", bytes, " bytes, the state at ", n,
                from > 0 ? " (a delta from " + std::to_string(from) + ")" : std::string(), " is ", state_bytes(n, from),
                " (from another model, layout or activation dtype?)");
    // The header's fields.
    STRIX_CHECK(hdr.rope_delta <= 0 && hdr.rope_delta >= -max_pos, "Qwen4ExpSession::import_state: header "
                "rope_delta = ", hdr.rope_delta, ", expected ", -max_pos, "..0 (image blocks pack positions "
                "ahead of sequence indices)");
    const int64_t cont[3] = {hdr.cont_t, hdr.cont_h, hdr.cont_w};
    for (int k = 0; k < 3; ++k)
        STRIX_CHECK(cont[k] >= 0 && cont[k] <= max_pos, "Qwen4ExpSession::import_state: header continuation ",
                    (k == 0 ? "t" : k == 1 ? "h" : "w"), " = ", cont[k], ", expected 0..", max_pos);
    STRIX_CHECK(hdr.has_image == 0 || hdr.has_image == 1, "Qwen4ExpSession::import_state: header has_image = ",
                hdr.has_image, ", expected 0 or 1");
    STRIX_CHECK(hdr.ntriples >= 0 && hdr.ntriples <= 3 && (hdr.ntriples > 0 ? hdr.has_image == 1 : true),
                "Qwen4ExpSession::import_state: header carries ", hdr.ntriples, " triples with has_image = ",
                hdr.has_image);
    // The resume rule (restore's, vision Phase A): rows at or past n must continue uniformly at
    // rope_delta - the state's continuation rule; image-tail states carry their last rows so the first forward
    // can rope its block window with them.
    const bool cont_uniform = cont[0] == cont[1] && cont[1] == cont[2];
    STRIX_CHECK(cont_uniform && cont[0] == n + hdr.rope_delta,
                "Qwen4ExpSession::import_state: row ", n, " continues at ", triple_str(cont),
                ", not uniformly at n + rope_delta = ", n + hdr.rope_delta, " - the state ends inside an image "
                "span (image rows whose block's end isn't its continuation): resuming mid-image from an "
                "exported state isn't implemented (the prompt cache should only save whole turns)");
    const bool carry_tail = hdr.has_image == 1 && hdr.ntriples > 0 && !triple_uniform(hdr.triples + (hdr.ntriples - 1) * 3);
    const int64_t b0 = from / kernels::kQsaBlock, b1 = n / kernels::kQsaBlock;
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_state");
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = const_cast<uint8_t *>(in);
    cur += sizeof(Qwen4ExpStateHeader);  // parsed and validated above, straight off `in`
    PleHistory hist;
    std::memcpy(&hist, cur, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            if (is_q8_format(kv_format_)) {
                kv8_copy(cur, l, n, from, kv8_tail_[tail_cur_][l], false, nullptr);
            } else {
                host_copy(cur, k_cache_[l].get() + (size_t)from * row, kv, false, stream_, "K cache");
                host_copy(cur, v_cache_[l].get() + (size_t)from * row, kv, false, stream_, "V cache");
            }
            block_keys_copy(cur, block_keys_[l].get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
            host_copy(cur, tail_[tail_cur_][l].get(), tail_[tail_cur_][l].size(), false, stream_, "indexer tail");
        } else {
            host_copy(cur, conv_state_[l].get(), conv_state_[l].size(), false, stream_, "conv state");
            host_copy(cur, rec_state_[l].get(), rec_state_[l].size() * sizeof(float), false, stream_, "recurrent state");
        }
    }
    host_copy(cur, ple_state_.get(), ple_state_.size(), false, stream_, "PLE conv state");
    if (mtp_) {
        const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
        host_copy(cur, k_cache_mtp_.get() + (size_t)from * row, kv, false, stream_, "MTP K cache");
        host_copy(cur, v_cache_mtp_.get() + (size_t)from * row, kv, false, stream_, "MTP V cache");
        block_keys_copy(cur, block_keys_mtp_.get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
        host_copy(cur, tail_mtp_[mtp_tail_cur_].get(), tail_mtp_[mtp_tail_cur_].size(), false, stream_,
                  "MTP indexer tail");
        host_copy(cur, mtp_prev_.get(), mtp_prev_.size(), false, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_state");
    ple_hist_ = hist;
    rope_delta_ = hdr.rope_delta;
    history_ever_interleaved_ = hdr.has_image == 1;
    pos3_host_.clear();
    if (carry_tail) pos3_host_.assign(hdr.triples, hdr.triples + (size_t)(hdr.ntriples * 3));
    pos3_start_ = n - (int64_t)pos3_host_.size() / 3;
    STRIX_CHECK((size_t)(cur - in) == bytes, "Qwen4ExpSession::import_state: read ", cur - in, " of ", bytes, " bytes");
    pos_ = n, broken_ = false, mtp_chain_step_ = -1, verify_pending_ = false;
    imported_at_ = n;
    rewind_to_.push_back(0);  // the KV cache below n was rewritten (or completed): no older snapshot matches it
}

// The end of a forward: the error words and the logits come back in one go - async copies into pinned memory on
// stream_ behind the forward, one synchronize, then the checks on the host copies. It used to be a synchronize and
// five blocking hipMemcpy calls (~40 us each on gfx1151), per forward - and the MTP
// cycle runs several forwards per token. The pinned area is reused only after this synchronize (the gfx1151
// pinned-reuse hazard).
std::vector<float> Qwen4ExpSession::read_back(size_t n_floats, const char *what, int64_t n_cand_rows) {
    STRIX_CHECK(n_floats * 4 <= readback_.size() - kReadbackLogitsAt, "Qwen4ExpSession::read_back: ", n_floats,
                " logits, the pinned area holds ", (readback_.size() - kReadbackLogitsAt) / 4);
    STRIX_CHECK(n_cand_rows >= 0 && (n_cand_rows == 0 || n_floats == 0), "Qwen4ExpSession::read_back: ", n_cand_rows,
                " candidate rows with ", n_floats, " logits (one or the other)");
    const size_t cand_bytes = (size_t)n_cand_rows * kernels::kLogitCands * sizeof(kernels::LogitCand);
    STRIX_CHECK(cand_bytes + (size_t)n_cand_rows * 4 <= cand_dev_.size(), "Qwen4ExpSession::read_back: ", n_cand_rows,
                " candidate rows, the device area holds ", cand_dev_.size(), " bytes");
    auto *host = static_cast<uint8_t *>(readback_.get());
    auto *words = reinterpret_cast<uint32_t *>(host);  // router [0, 3), experts [3, 6), attention [6, 9), 8-bit K/V [9, 12)
    auto *emb = reinterpret_cast<kernels::EmbeddingError *>(host + kReadbackEmbAt);
    auto d2h = [&](void *to, const void *from, size_t bytes, const char *name) {
        STRIX_HIP_CHECK(hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToHost, stream_), "Qwen4ExpSession::read_back",
                        what, ": ", name);
    };
    d2h(words, router_err_.get(), 12, "router error");
    d2h(words + 3, expert_err_.get(), 12, "expert error");
    d2h(words + 6, attn_err_.get(), 12, "attention error");
    if (is_q8_format(kv_format_)) d2h(words + 9, kv8_err_.get(), 12, "8-bit K/V error");
    else std::memset(words + 9, 0, 12);
    d2h(emb, emb_err_.get(), sizeof(kernels::EmbeddingError), "embedding error");
    if (n_floats) d2h(host + kReadbackLogitsAt, logits_.get(), n_floats * 4, "logits");
    if (n_cand_rows) d2h(host + kReadbackLogitsAt, cand_dev_.get(), cand_bytes + (size_t)n_cand_rows * 4, "candidates");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::read_back", what, ": end of the forward");
    const std::string tag = std::string("Qwen4ExpSession") + what;
    kernels::report_router_error(words, (tag + " router").c_str());
    kernels::report_expert_error(words + 3, (tag + " experts").c_str());
    kernels::report_attention_error(words + 6, (tag + " QSA attention").c_str());
    kernels::report_kv8_error(words + 9, (tag + " K/V cache").c_str());
    kernels::report_embedding_error(*emb, m_.dims().vocab);
    if (n_cand_rows) {
        const auto *c = reinterpret_cast<const kernels::LogitCand *>(host + kReadbackLogitsAt);
        const auto *f = reinterpret_cast<const uint32_t *>(host + kReadbackLogitsAt + cand_bytes);
        cand_host_.rows = n_cand_rows;
        cand_host_.cand.assign(c, c + n_cand_rows * kernels::kLogitCands);
        cand_host_.nan.assign(f, f + n_cand_rows);
    }
    const float *l = reinterpret_cast<const float *>(host + kReadbackLogitsAt);
    return std::vector<float>(l, l + n_floats);
}

void Qwen4ExpSession::want_candidates(int64_t n_valid, const uint32_t *masks, int64_t mask_rows, int64_t mask_words) {
    const char *fn = "Qwen4ExpSession::want_candidates";
    const int64_t vocab = m_.dims().vocab;
    STRIX_CHECK(n_valid >= 1 && n_valid <= vocab, fn, ": n_valid = ", n_valid, ", expected 1..", vocab);
    if (masks == nullptr) {
        STRIX_CHECK(mask_rows == 0 && mask_words == 0, fn, ": mask_rows = ", mask_rows, ", mask_words = ", mask_words,
                    " without masks (pass 0 for both)");
    } else {
        STRIX_CHECK(mask_rows >= 1 && mask_rows <= kMaxLogits, fn, ": mask_rows = ", mask_rows, ", expected 1..",
                    kMaxLogits);
        STRIX_CHECK(mask_words >= (vocab + 31) / 32, fn, ": mask_words = ", mask_words, " covers ", mask_words * 32,
                    " ids, the vocabulary has ", vocab);
        const size_t bytes = (size_t)(mask_rows * mask_words) * 4;
        if (mask_host_.size() < bytes) {
            STRIX_CHECK(!mask_upload_recorded_ || hipEventSynchronize(mask_uploaded_) == hipSuccess, fn,
                        ": waiting for the last mask upload before replacing its buffer");
            mask_upload_recorded_ = false;
            const size_t capacity = (size_t)(kMaxLogits * mask_words) * 4;
            mask_host_ = PinnedHostBuffer(capacity, "structured-output masks (host)");
            mask_dev_ = DeviceBuffer<uint32_t>(capacity / 4, "structured-output masks");
            if (mask_uploaded_ == nullptr)
                STRIX_HIP_CHECK(hipEventCreateWithFlags(&mask_uploaded_, hipEventDisableTiming), fn, ": mask upload event");
        }
        // The gate: the previous upload from this pinned buffer has completed before it is written again.
        if (mask_upload_recorded_) STRIX_HIP_CHECK(hipEventSynchronize(mask_uploaded_), fn, ": last mask upload");
        std::memcpy(mask_host_.get(), masks, bytes);
    }
    mask_rows_ = masks == nullptr ? 0 : mask_rows;
    mask_words_ = masks == nullptr ? 0 : mask_words;
    cand_n_valid_ = n_valid;
}

std::vector<float> Qwen4ExpSession::forward(const std::vector<int32_t> &ids, int64_t n_logits, const Qwen4ExpProbe &probe) {
    return forward(ids, n_logits, nullptr, probe);
}

void Qwen4ExpSession::ensure_vision_scratch(int64_t grid_h, int64_t grid_w) {
    const VisionTower &V = m_.vision();
    const int64_t patches = grid_h * grid_w;
    if (patches <= vis_patches_) return;  // the scratch already serves a grid this big (it never shrinks)
    const TowerScratchBytes b = V.scratch_bytes(grid_h, grid_w);
    const size_t e = es(m_.act());
    // One allocation set for the largest grid the session has seen (grow-on-larger): every buffer's size
    // is a pure function of the patch count, so equal sizes are reused as-is. The 8192-patch worst case
    // is ~280 MB of activations - the accepted size of vision = on.
    vis_pv_ = DeviceBuffer<uint8_t>((size_t)(b.patch_in) * e, "vision pixel values");
    vis_x_ = DeviceBuffer<uint8_t>((size_t)(b.hidden) * e, "vision x");
    vis_conv_ = DeviceBuffer<uint8_t>((size_t)(b.hidden) * e, "vision conv");
    vis_ln_ = DeviceBuffer<uint8_t>((size_t)(b.hidden) * e, "vision ln");
    vis_proj_ = DeviceBuffer<uint8_t>((size_t)(b.hidden) * e, "vision proj");
    vis_qkv_ = DeviceBuffer<uint8_t>((size_t)(b.qkv) * e, "vision qkv");
    vis_mlp_ = DeviceBuffer<uint8_t>((size_t)(b.mlp) * e, "vision mlp");
    vis_merg_ = DeviceBuffer<uint8_t>((size_t)(b.merg) * e, "vision merge block");
    vis_merged_ = DeviceBuffer<uint8_t>((size_t)(b.merged) * e, "vision merged rows");
    vis_idx_ = DeviceBuffer<int32_t>((size_t)(patches * 4), "vision tap indices");
    vis_w_ = DeviceBuffer<float>((size_t)(patches * 4), "vision tap weights");
    vis_rows_ = DeviceBuffer<int32_t>((size_t)(patches / (V.dims().merge * V.dims().merge)),
                                      "vision scatter row indices");
    vis_patches_ = patches;
}

void Qwen4ExpSession::run_tower_and_scatter(const std::vector<int32_t> &ids, const ImageInput &image) {
    const char *fn = "Qwen4ExpSession::run_tower_and_scatter";
    // Timing: the tower kernels are the cost center of an image prefill - log the span to stderr (the
    // journal) with the grid. The two syncs exist only for this line.
    const auto t_start = std::chrono::steady_clock::now();
    const VisionTower &V = m_.vision();
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const int64_t n = image.grid_h * image.grid_w, m2 = V.dims().merge * V.dims().merge;
    STRIX_CHECK(V.dims().out_hidden == D.d, fn, ": the tower outputs ", V.dims().out_hidden, " columns, the "
                "trunk's embedding width is ", D.d, " (the model load checked this)");

    // Host-side per-grid arrays: the position-table taps (grid_taps) and the image_token_id rows in
    // sequence order - the same order as the merger output's rows, which the caller's expansion and
    // set_positions triples follow (one merged row per image_token_id, block-major).
    std::vector<int32_t> idx;
    std::vector<float> wts;
    V.grid_taps(image.grid_h, image.grid_w, idx, wts);
    std::vector<int32_t> rows;
    rows.reserve((size_t)(n / m2));
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == kVisionImageTokenId) rows.push_back((int32_t)i);
    STRIX_CHECK((int64_t)rows.size() == n / m2, fn, ": internal: ", rows.size(), " image_token_id rows vs "
                "the merged row count ", n / m2, " (forward counted them before the launch)");

    ensure_vision_scratch(image.grid_h, image.grid_w);
    // Pixel values arrive FP32 (the processor's contract) and enter the tower in the activation dtype -
    // one rounding, the same point the oracle's `.to(dtype)` uses. The uploads join this forward's stream
    // ahead of the tower's reads; the host vectors die after the sync at the forward's end.
    if (act == Act::F32) {
        STRIX_HIP_CHECK(hipMemcpyAsync(vis_pv_.get(), image.pixel_values.data(),
                                        (size_t)(n * V.dims().patch_in) * 4, hipMemcpyHostToDevice, stream_),
                        fn, ": uploading ", n * V.dims().patch_in, " FP32 pixel values");
    } else {
        std::vector<uint16_t> bits((size_t)(n * V.dims().patch_in));
        for (size_t i = 0; i < bits.size(); ++i) bits[i] = f32_to_bf16_bits(image.pixel_values[i]);
        STRIX_HIP_CHECK(hipMemcpyAsync(vis_pv_.get(), bits.data(), bits.size() * 2, hipMemcpyHostToDevice,
                                        stream_),
                        fn, ": uploading ", bits.size(), " BF16-rounded pixel values");
    }
    STRIX_HIP_CHECK(hipMemcpyAsync(vis_idx_.get(), idx.data(), (size_t)(n * 4) * 4, hipMemcpyHostToDevice,
                                    stream_),
                    fn, ": uploading ", n * 4, " tap indices");
    STRIX_HIP_CHECK(hipMemcpyAsync(vis_w_.get(), wts.data(), (size_t)(n * 4) * 4, hipMemcpyHostToDevice,
                                    stream_),
                    fn, ": uploading ", n * 4, " tap weights");
    STRIX_HIP_CHECK(hipMemcpyAsync(vis_rows_.get(), rows.data(), (size_t)rows.size() * 4,
                                    hipMemcpyHostToDevice, stream_),
                    fn, ": uploading ", rows.size(), " scatter row indices");

    const TowerScratch s{vis_pv_.get(), vis_x_.get(), vis_conv_.get(), vis_ln_.get(), vis_proj_.get(),
                         vis_qkv_.get(), vis_mlp_.get(), vis_merg_.get(), vis_idx_.get(), vis_w_.get(),
                         vis_merged_.get()};
    V.forward(image.grid_h, image.grid_w, vis_idx_.get(), vis_w_.get(), s, stream_);
    // The merged rows replace the image_token_id embeddings in all H streams (the embedding kernel wrote
    // each token's row into every stream; the scatter overwrites the same layout).
    kernels::vision_embedding_scatter(x_.get(), vis_rows_.get(), (int64_t)rows.size(), vis_merged_.get(), D.d,
                                      (int64_t)ids.size(), D.H, D.H * D.d, act, stream_);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), fn, ": tower timing sync");
    const int64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count();
    ++vision_images_, vision_patches_ += n, vision_merged_ += (int64_t)rows.size(), vision_ns_ += elapsed_ns;
    for (int64_t m = vision_max_ns_.load(); elapsed_ns > m && !vision_max_ns_.compare_exchange_weak(m, elapsed_ns);) {
    }
    std::fprintf(stderr, "vision tower: grid %lldx%lld, %lld patches, %lld merged rows, %.2f s\n",
                 (long long)image.grid_h, (long long)image.grid_w, (long long)n, (long long)rows.size(),
                 (double)elapsed_ns * 1e-9);
    std::fflush(stderr);
}

std::vector<float> Qwen4ExpSession::forward(const std::vector<int32_t> &ids, int64_t n_logits,
                                            const ImageInput *image, const Qwen4ExpProbe &probe) {
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward: an earlier call failed midway; reset() the session first");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward: a forward_verify awaits keep_verify / drop_verify");
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    STRIX_TRACE_RANGE(std::string(gdn_to_spare_ ? "verify" : ids.size() == 1 ? "decode" : "prefill") + " T=" +
                      std::to_string(ids.size()));
    STRIX_TRACE_STAGE(stage);
    STRIX_TRACE_SET(stage, "embed");
    // Dense Q4 / Q8 projection: on the matrix units for large enough forwards when switched on.
    auto lin = [&](const void *x, const QWeightView &w, void *y, int64_t M, Act out_act) {
        if (wmma && M >= kWmmaMinTokens) kernels::linear_qw_wmma(x, w, y, M, act, out_act, stream_);
        else kernels::linear_qw(x, w, y, M, act, out_act, stream_);
    };
    const kernels::MoeMath moe_math = wmma ? kernels::MoeMath::WmmaBf16 : kernels::MoeMath::F32;
    // Expert slots per token: top-k + the shared expert's when it's stacked.
    const int64_t T = (int64_t)ids.size(), n4 = D.H * D.d, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::forward: ", T, " tokens, expected 1..", max_tokens_);
    STRIX_CHECK(pos_ + T <= capacity_, "Qwen4ExpSession::forward: positions [", pos_, ", ", pos_ + T,
                ") exceed the capacity ", capacity_);
    STRIX_CHECK(n_logits >= 0 && n_logits <= std::min(T, kMaxLogits), "Qwen4ExpSession::forward: n_logits = ",
                n_logits, ", expected 0..", std::min(T, kMaxLogits));
    const int64_t cand_n_valid = cand_n_valid_;  // want_candidates() covers this one forward, whatever happens
    const int64_t mask_rows = mask_rows_, mask_words = mask_words_;
    cand_n_valid_ = 0, mask_rows_ = 0, mask_words_ = 0;
    STRIX_CHECK(mask_rows == 0 || (cand_n_valid > 0 && mask_rows == n_logits), "Qwen4ExpSession::forward: ",
                mask_rows, " structured-output mask rows for a forward of ", n_logits, " logits rows");
    // Image forward (vision): every check that can fail without touching session state runs here, before
    // the pending positions are consumed. run_tower_and_scatter checks the rest (the row count, against ids).
    const VisionTower *tower = nullptr;
    if (image != nullptr) {
        const char *ifn = "Qwen4ExpSession::forward(image)";
        STRIX_CHECK(m_.has_vision(), ifn, ": the model was loaded without a vision tower - construct "
                    "Qwen4ExpModel with the vision.strixw path to enable image input");
        tower = &m_.vision();
        STRIX_CHECK(image->grid_t == 1, ifn, ": grid_t = ", image->grid_t, ", expected 1 (a single image; "
                    "the tower runs one temporal segment)");
        STRIX_CHECK(image->grid_h >= tower->dims().merge && image->grid_w >= tower->dims().merge &&
                        image->grid_h % tower->dims().merge == 0 && image->grid_w % tower->dims().merge == 0,
                    ifn, ": grid ", image->grid_h, "x", image->grid_w, ", expected multiples of the merge "
                    "side ", tower->dims().merge);
        const int64_t np = image->grid_h * image->grid_w;
        STRIX_CHECK(np <= kernels::kMaxVisionSeq, ifn, ": ", np, " patches exceeds the dense-attention cap ",
                    kernels::kMaxVisionSeq);
        STRIX_CHECK((int64_t)image->pixel_values.size() == np * tower->dims().patch_in, ifn, ": ",
                    image->pixel_values.size(), " pixel values, expected ", np, " patches x ",
                    tower->dims().patch_in, " columns");
        STRIX_CHECK(!pending3_.empty(), ifn, ": an image forward carries its set_positions triples (the "
                    "image rows' (t,h,w) section positions) - set them first (Phase A's walk rule)");
        STRIX_CHECK((int64_t)pending3_.size() / 3 == T, ifn, ": set_positions holds ", pending3_.size() / 3,
                    " rows, this forward runs ", T);
        const int64_t n_img = std::count(ids.begin(), ids.end(), kVisionImageTokenId);
        STRIX_CHECK(n_img == np / (tower->dims().merge * tower->dims().merge), ifn, ": ids carry ", n_img,
                    " image_token_id rows (", kVisionImageTokenId, "), the image produces ", np, " patches / ",
                    tower->dims().merge * tower->dims().merge, " = ",
                    np / (tower->dims().merge * tower->dims().merge), " merged rows - expand the placeholder "
                    "to the merged token count before the forward");
    }
    // mRoPE: consume this forward's pending triples into the history (set_positions checked the continuation at
    // set time; pos_ cannot have moved - restore / import refuse a pending span). Once interleaved, every
    // forward's rows join the history - scalar rows materialized at their rope_pos - so it stays contiguous and
    // set_positions can keep checking continuations. Text-only forwards: both no-ops, no upload, no allocation.
    // After broken_ = true: a mismatching span (rows != T) takes the session down cleanly instead of leaving
    // a half-consumed history.
    broken_ = true;
    if (!pending3_.empty()) {
        STRIX_CHECK((int64_t)pending3_.size() / 3 == T, "Qwen4ExpSession::forward: set_positions holds ",
                    pending3_.size() / 3, " rows, this forward runs ", T, " - the triples cover exactly one forward");
        // The history holds rows [pos3_start_, pos_); a text-only prefix needs no rows (their triples are the
        // scalar rope_pos values), so the first interleaved forward starts the history at its own row 0.
        if (pos3_host_.empty()) pos3_start_ = pos_;
        pos3_host_.insert(pos3_host_.end(), pending3_.begin(), pending3_.end());
        pending3_.clear();
        history_ever_interleaved_ = true;
    } else if (!pos3_host_.empty()) {
        for (int64_t r = 0; r < T; ++r) {
            const int64_t v = rope_pos(pos_ + r);
            pos3_host_.insert(pos3_host_.end(), {v, v, v});
        }
    }
    // rope_delta: with an interleaved prefix, every later scalar row's angle sits at (the continuation of
    // the history) - (the sequence index). Recomputed only when the next row is a scalar one: a span ending
    // on a uniform row steps by 1, one ending at a complete image block row continues at the block's max
    // position + 1; a span ending mid-block-row (an image split across forwards) has image rows next - the
    // delta stays at its last block-end value (unused until the image completes, then recomputed).
    if (history_ever_interleaved_ && !pos3_host_.empty()) {  // (a session resumed after the image has the delta
        const Cont next = continuation_of(pos3_host_.data() + (size_t)((pos3Hend() - pos3_start_ - 1) * 3),  // but
                                          pos3_host_.data(), (int64_t)pos3_host_.size() / 3);  // no history -
        if (next.kind == ContKind::kUniform || next.kind == ContKind::kEndOrImage)  // its rows are all scalar
            rope_delta_ = next.exact[0] - (pos_ + T);
    }
    prepare_positions(pos_, T);
    const int64_t rope0 = mk_jobs_.pos3 == nullptr ? rope_pos(pos_) : pos_;  // scalar-path rope base (Phase A:
    // interleaved calls pass the kernels their positions, so pos0 is unused there)
    const bool fresh = pos_ == 0;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    const ProbeType pt = act == Act::F32 ? ProbeType::F32 : ProbeType::BF16;
    auto show = [&](const std::string &name, const void *dev, int64_t cols, ProbeType type) {
        if (!probe) return;
        STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "before probe '", name, "'");
        probe(name, dev, T, cols, type);
    };
    // inv_ready: inv_ already holds X's inverse RMS (the preceding injection was hc_inject_inv). xb: the streams
    // to mix, x_ unless given (the MTP layer's mtp_x_).
    auto hc_mix = [&](const Qwen4ExpModel::Hc &hc, bool inject, bool inv_ready = false, uint8_t *xb = nullptr) {
        float *w_in = inject ? w_in_.get() : nullptr;
        uint8_t *X = xb ? xb : x_.get();
        STRIX_CHECK((hc.down.bits == 4 || hc.down.bits == 8) && (hc.up.bits == 4 || hc.up.bits == 8),
                    "Qwen4ExpSession: HC weights not loaded (down Q", hc.down.bits, ", up Q", hc.up.bits, ")");
        if (wmma && T >= kWmmaMinTokens) {  // the prefill path on the matrix units (kernels/hc_wmma)
            float *ws = hc_ws_.get();
            if (hc.down.bits == 8)
                kernels::hc_mix_down_wmma(X, hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in,
                                          inv_.get(), ws, hc_ws_bytes_, act, stream_, inv_ready);
            else
                kernels::hc_mix_down_wmma(X, hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in,
                                          inv_.get(), ws, hc_ws_bytes_, act, stream_, inv_ready);
            if (hc.up.bits == 8)
                kernels::hc_mix_up_wmma(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act,
                                        stream_);
            else
                kernels::hc_mix_up_wmma(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act,
                                        stream_);
            return;
        }
        if (hc.down.bits == 8)
            kernels::hc_mix_down(X, hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act,
                                 stream_);
        else
            kernels::hc_mix_down(X, hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act,
                                 stream_);
        if (hc.up.bits == 8)
            kernels::hc_mix_up(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
    };

    // PLE rows for this forward: the ids advance the n-gram history now; the rows come from set_lookahead's gather
    // when it was for exactly these ids and history, else a gather starts now on a worker - overlapping the
    // embedding and the layers before the PLE layer. Joined there (a gather failure fails the forward).
    int ple_buf = 0;
    std::future<void> ple_ready;
    std::vector<int32_t> lookahead = std::move(ple_lookahead_);
    ple_lookahead_.clear();
    if (D.ple_layer >= 0) {
        const PleHistory before = ple_hist_;
        std::vector<int64_t> tok(ids.begin(), ids.end()), rows((size_t)(T * 16));
        ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), T, ple_hist_, rows.data());
        if (ple_pending_.active && ple_pending_.ids == ids && ple_pending_.hist.prev == before.prev) {
            ple_buf = ple_pending_.buf, ple_ready = std::move(ple_pending_.done);
            ple_pending_ = PleGather{};
        } else {
            drop_ple_pending();  // a stale hint: let it finish before its buffer is reused
            // Small forwards (decode, MTP verify): on the persistent worker, overlapping the embedding and layer 0.
            ple_ready = start_ple_gather(std::move(rows), T, ple_buf,
                                         T >= kWmmaMinTokens ? PleRun::Async : PleRun::Worker);
        }
    }

    // Injection; on the WMMA mix path fused with the next mix's inverse RMS (hc_inject_inv). Returns whether inv_
    // is ready for that mix.
    const bool fuse_inv = wmma && T >= kWmmaMinTokens;
    auto inject_y = [&]() {
        if (fuse_inv) kernels::hc_inject_inv(x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, D.eps, inv_.get(), act, stream_);
        else kernels::hc_inject(x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);
        return fuse_inv;
    };
    bool inv_ready = false;  // X changed since inv_ was computed (the embedding)

    // Embedding, written straight into all 4 streams (one launch).
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), ids.data(), (size_t)T * 4, hipMemcpyHostToDevice, stream_),
                    "uploading ", T, " token ids");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    if (m_.is_embed_q8()) {
        kernels::embedding_lookup(ids_.get(), T, m_.embed_q8(), x_.get(), act, emb_err_.get(), stream_, D.H, n4);
    } else {
        kernels::embedding_lookup(ids_.get(), T, m_.embed(), D.vocab, D.d, x_.get(), act, emb_err_.get(), stream_, D.H, n4);
    }
    // The image (if any): the tower runs on this forward's stream after the embedding launch, and its
    // merged rows overwrite the image_token_id rows in every stream before layer 0 reads x_ (and before
    // the probe below - a probed embed_streams is after the scatter).
    if (image != nullptr) run_tower_and_scatter(ids, *image);
    show("embed_streams", x_.get(), n4, pt);  // [T, H * d]: the embedding in each of the H streams

    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen4ExpModel::Layer &l = m_.layer(i);
        const std::string L = "L" + std::to_string(i) + ".";
        if (i == D.ple_layer) STRIX_TRACE_SET(stage, "L" + std::to_string(i) + " ple");
        if (i == D.ple_layer) {
            // n-gram rows (gathered on the host since the forward started), key|value projection, gate + conv into X.
            show(L + "ple_in", x_.get(), n4, pt);
            {
                const auto t0 = std::chrono::steady_clock::now();
                ple_ready.get();
                const int64_t ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                ++ple_waits_, ple_wait_ns_ += ns;
                for (int64_t m = ple_wait_max_ns_.load(); ns > m && !ple_wait_max_ns_.compare_exchange_weak(m, ns);) {
                }
            }
            STRIX_HIP_CHECK(hipMemcpyAsync(ple_e_.get(), ple_host_[ple_buf].get(), (size_t)(T * D.ple_e) * e,
                                           hipMemcpyHostToDevice, stream_),
                            "uploading PLE rows");
            // The next forward's rows (set_lookahead) into the other buffer, during this forward's GPU work: that
            // buffer's last upload was in an earlier forward, which ended with a stream sync.
            if (!lookahead.empty()) {
                std::vector<int64_t> tok(lookahead.begin(), lookahead.end()), rows(lookahead.size() * 16);
                PleHistory h = ple_hist_;
                ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), (int64_t)tok.size(), h, rows.data());
                ple_pending_.buf = 1 - ple_buf;
                ple_pending_.done = start_ple_gather(std::move(rows), (int64_t)tok.size(), ple_pending_.buf,
                                                     PleRun::Async);
                ple_pending_.ids = std::move(lookahead), ple_pending_.hist = ple_hist_, ple_pending_.active = true;
            }
            lin(ple_e_.get(), l.ple_kv, ple_kv_.get(), T, act);
            const kernels::PleShape sh{T, D.H, D.d, D.ple_taps, D.ple_dil, D.ple_ld};
            kernels::ple_gate(ple_kv_.get(), x_.get(), l.ple_norm_key, l.ple_norm_query, l.ple_norm_conv,
                              ple_state_.get(), sh, D.eps, ple_hist_buf_.get(), ple_sigma_.get(), act, stream_);
            kernels::ple_conv(ple_hist_buf_.get(), ple_sigma_.get(), ple_kv_.get(), l.ple_conv, sh, x_.get(),
                              ple_state_.get(), act, stream_);
            show(L + "ple_out", x_.get(), n4, pt);
        }

        if (i == D.ple_layer) inv_ready = false;  // the PLE changed X after the last injection
        STRIX_TRACE_SET(stage, "L" + std::to_string(i) + (D.is_attention[(size_t)i] ? " attn" : " gdn"));
        hc_mix(l.hc_attn, true, inv_ready);
        show(L + "attn_in", u_.get(), D.d, pt);
        show(L + "attn_w_in", w_in_.get(), D.H, ProbeType::F32);
        if (!D.is_attention[(size_t)i]) {
            // A verify writes its in_proj rows straight into the layer's verify_in_ buffer (keep_verify_prefix may
            // replay them) instead of proj_ and a copy.
            uint8_t *gp = gdn_to_spare_ ? verify_in_[(size_t)i].get() : proj_.get();
            lin(u_.get(), l.in_proj, gp, T, act);
            kernels::gdn_front(gp, {T, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                               l.conv_w, conv_state_[(size_t)i].get(),
                               (gdn_to_spare_ ? conv_spare_ : conv_state_)[(size_t)i].get(), fresh, l.A_log,
                               l.dt_bias, qkv_.get(), beta_.get(), g_.get(), act, stream_);
            // Prefill-sized WMMA forwards: the chunked delta rule on the matrix units; decode / MTP verify per token.
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {T, D.gk, D.gv, D.conv_c}, rec_state_[(size_t)i].get(),
                                (gdn_to_spare_ ? rec_spare_ : rec_state_)[(size_t)i].get(), fresh, core_.get(), act,
                                stream_,
                                wmma && T >= kWmmaMinTokens ? kernels::DeltaRulePath::ChunkedWmma
                                                            : kernels::DeltaRulePath::Auto,
                                precision_.gdn);
            // z (gv heads of kHD a token) read in place from the in_proj rows.
            kernels::rmsnorm_gated(core_.get(), gp + (size_t)D.conv_c * e, l.gdn_norm, gnorm_.get(), T * D.gv, kHD, D.eps,
                                   kernels::GateAct::Sigmoid, act, stream_, D.gv, D.gstride);
            show(L + "gdn_core", gnorm_.get(), D.gz, pt);
            lin(gnorm_.get(), l.out_proj, y_.get(), T, act);
        } else {
            // Plain K/V caches (empty with KvFormat::Q8 / Q8H).
            uint8_t *kc = k_cache_[(size_t)i].size() ? k_cache_[(size_t)i].get() : nullptr,
                    *vc = v_cache_[(size_t)i].size() ? v_cache_[(size_t)i].get() : nullptr;
            // 8-bit K/V: the new rows go to a staging area first ([T][K | V], a verify's into its saved rows for
            // keep_verify_prefix), then kv8_commit quantizes the groups they complete and writes the new tail.
            const bool q8 = is_q8_format(kv_format_);
            const bool hadamard = kv_format_ == KvFormat::Q8H;
            const int64_t W = D.hkv * hd;
            uint8_t *kv_rows = !q8 ? nullptr
                               : gdn_to_spare_ ? verify_in_[(size_t)i].get() + (size_t)(kMaxLogits * D.idx_d) * e
                                               : kv_new_.get();
            uint8_t *k_out = q8 ? kv_rows : kc + (size_t)(pos_ * W) * e, *v_out = q8 ? kv_rows + (size_t)W * e
                                                                                       : vc + (size_t)(pos_ * W) * e;
            const int64_t kv_ts = q8 ? 2 * W : W;
            lin(u_.get(), l.qkv, proj_.get(), T, act);
            // One launch: q normed + roped in place, k into the K cache, v copied into the V cache, past the dense
            // limit the indexer queries (normed + roped in place), and in a verify the raw indexer keys copied into
            // the layer's verify_in_ (keep_verify_prefix may replay them) - kernels/attention.hpp qk_norm_rope_jobs.
            // With q8h, q and k heads are rotated by the orthonormal 256-point Fast Walsh-Hadamard Transform.
            const bool qsa_layer = pos_ + T > D.dense_key_limit();
            kernels::NormRopeJob nr[5] = {
                {proj_.get(), D.astride, 2 * hd, proj_.get(), D.astride, 2 * hd, D.hq, hd, l.q_norm, hadamard, 0.0625f},
                {at(proj_, D.hq * 2 * hd), D.astride, hd, k_out, kv_ts, hd, D.hkv, hd, l.k_norm, hadamard, 0.0625f},
                {at(proj_, D.hq * 2 * hd + D.hkv * hd), D.astride, hd, v_out, kv_ts, hd, D.hkv, hd, nullptr, hadamard, 0.0625f},
                {at(proj_, D.idx_col), D.astride, D.idx_d, at(proj_, D.idx_col), D.astride, D.idx_d, D.idx_h, D.idx_d,
                 l.idx_q_norm, false, 0.0f}};
            int njobs = 3;
            if (qsa_layer) ++njobs;
            else nr[3] = {};
            if (gdn_to_spare_)
                nr[njobs++] = {at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, D.idx_d, verify_in_[(size_t)i].get(),
                               D.idx_d, D.idx_d, 1, D.idx_d, nullptr};
            kernels::qk_norm_rope_jobs(nr, njobs, T, D.eps, D.rot, rope0, m_.inv_freq(), act, stream_, m_.rope_scale(),
                                       mk_jobs_);
            // Indexer block keys (every completed block, this call's included) - kept current for when selection
            // starts to bite; the queries and scores are only needed past the dense limit. pos0 stays the
            // sequence index (the kernel derives block / tail array indices from it); with a rope_delta the
            // window's triples carry the shifted angles (a shifted pos0 would shift the block partition too).
            kernels::qsa_block_keys(at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, T, pos_,
                                    pos_ % kernels::kQsaBlock ? tail_[tail_cur_][(size_t)i].get() : nullptr,
                                    tail_[1 - tail_cur_][(size_t)i].get(), D.idx_d, l.idx_k_norm, D.eps, m_.inv_freq(),
                                    D.rot, block_keys_[(size_t)i].get(), cap_blocks_, act, stream_, m_.rope_scale(),
                                    mk_block_);
            const kernels::AttentionShape as{T, pos_, D.hq, D.hkv, hd, D.astride, 2 * hd};
            const float scale = 1.0f / std::sqrt((float)hd);
            kernels::Kv8View kv8;
            if (q8) {
                const size_t li = (size_t)i;
                kernels::kv8_commit(k_out, v_out, kv_ts, T, pos_,
                                    pos_ % kernels::kKv8Group ? kv8_tail_[tail_cur_][li].get() : nullptr,
                                    kv8_tail_[1 - tail_cur_][li].get(), kv8_store_[li].get(), kv8_cap_, W, act,
                                    kv8_err_.get(), stream_);
                kv8 = {kv8_store_[li].get(), kv8_cap_, W,
                       pos_ % kernels::kKv8Group ? kv8_tail_[tail_cur_][li].get() : nullptr, k_out, v_out, kv_ts, pos_};
                kv8.hadamard = hadamard;
            }
            // Prefill-sized forwards with BF16 activations: the attention on the matrix units (kernels/attention.hpp).
            const bool wmma_attn = wmma && act == Act::BF16 && T >= kWmmaMinTokens;
            // Dequantize the rows below the last query's exact_from once, then plain loads (Kv8View staged_k), when
            // that is cheaper than unpacking in the loop (kv8_stage_pays).
            const auto stage_kv8 = [&](int64_t keys_per_query) {
                const int64_t rows = kernels::kv8_exact_from(pos_ + T - 1);
                if (!(wmma_attn && q8 && rows > 0 && kv8_stage_pays(T, keys_per_query, rows))) return;
                STRIX_CHECK(kv8_stage_.size() >= (size_t)(2 * kv8_cap_ * W) * 2, "Qwen4ExpSession::forward: ",
                            "8-bit staging buffer of ", kv8_stage_.size(), " bytes for a ", T, "-token forward");
                uint8_t *sk = kv8_stage_.get(), *sv = sk + (size_t)(kv8_cap_ * W) * 2;
                kernels::kv8_dequant(kv8.store, kv8_cap_, W, 0, rows, sk, sv, act, stream_);
                kv8.staged_k = sk, kv8.staged_v = sv, kv8.staged_rows = rows;
            };
            if (!qsa_layer) {
                stage_kv8(pos_ + T);  // a dense query reads every row up to its own
                if (wmma_attn && q8)
                    kernels::attention_wmma(proj_.get(), at(proj_, hd), kv8, as, scale, core_.get(), attn_ws_.get(),
                                            attn_ws_bytes_, act, stream_, precision_.attention);
                else if (wmma_attn)
                    kernels::attention_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(),
                                            attn_ws_.get(), attn_ws_bytes_, act, stream_, precision_.attention);
                else if (q8)
                    kernels::attention(proj_.get(), at(proj_, hd), kv8, as, scale, core_.get(), attn_ws_.get(),
                                       attn_ws_bytes_, act, stream_);
                else
                    kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(),
                                       attn_ws_.get(), attn_ws_bytes_, act, stream_);
            } else {
                // QSA: indexer queries (normed + roped in place), block scores, the top budget / 4 blocks, gathered
                // attention. Queries that still see every key keep all their blocks - the dense set.
                const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
                void *iq = at(proj_, D.idx_col);  // normed + roped by the qk_norm_rope_jobs above
                kernels::qsa_scores(iq, D.astride, D.idx_d, T, pos_, D.idx_h, D.idx_d, block_keys_[(size_t)i].get(),
                                    cap_blocks_, qsa_scores_.get(), cap_blocks_, act, stream_);
                kernels::qsa_topk(qsa_scores_.get(), cap_blocks_, T, pos_, qsa_k, qsa_sel_.get(), qsa_nsel_.get(),
                                  qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
                show(L + "qsa_sel", qsa_sel_.get(), qsa_k, ProbeType::I32);
                show(L + "qsa_nsel", qsa_nsel_.get(), 1, ProbeType::I32);
                stage_kv8(std::min<int64_t>(D.qsa_budget, pos_ + T));
                if (wmma_attn && q8)
                    kernels::attention_gathered_wmma(proj_.get(), at(proj_, hd), kv8, as, qsa_sel_.get(),
                                                     qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                                     attn_ws_bytes_, attn_err_.get(), act, stream_, precision_.attention);
                else if (wmma_attn)
                    kernels::attention_gathered_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                                     qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                                     attn_ws_bytes_, attn_err_.get(), act, stream_, precision_.attention);
                else if (q8)
                    kernels::attention_gathered(proj_.get(), at(proj_, hd), kv8, as, qsa_sel_.get(), qsa_nsel_.get(),
                                                qsa_k, scale, core_.get(), attn_ws_.get(), attn_ws_bytes_,
                                                attn_err_.get(), act, stream_);
                else
                    kernels::attention_gathered(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                                qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                                attn_ws_bytes_, attn_err_.get(), act, stream_);
            }
            show(L + "attn_core", core_.get(), D.hq * D.hd, pt);
            lin(core_.get(), l.o_proj, y_.get(), T, act);
        }
        show(L + "mixer", y_.get(), D.d, pt);
        inv_ready = inject_y();

        STRIX_TRACE_SET(stage, "L" + std::to_string(i) + " moe");
        hc_mix(l.hc_mlp, true, inv_ready);
        show(L + "mlp_in", u_.get(), D.d, pt);
        show(L + "mlp_w_in", w_in_.get(), D.H, ProbeType::F32);
        // Router: logits FP32, 512 routed + the shared gate in one [T, 513] output; prefill-sized
        // forwards on the matrix units (exact BF16 products, FP32 sums - only the summation order differs).
        if (wmma && T >= kWmmaMinTokens)
            kernels::linear_bf16w_wmma(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32,
                                       stream_);
        else
            kernels::linear_bf16w(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32,
                                  stream_);
        show(L + "router_logits", router_logits_.get(), D.experts + 1, ProbeType::F32);
        kernels::moe_router(router_logits_.get(), D.experts + 1, T, D.experts, D.top_k,
                            sep ? nullptr : router_logits_.get() + D.experts, D.experts + 1, (int32_t)D.experts,
                            route_ids_.get(), route_coef_.get(), Act::F32, router_err_.get(), stream_);
        show(L + "router_ids", route_ids_.get(), A, ProbeType::I32);
        show(L + "router_coef", route_coef_.get(), A, ProbeType::F32);
        const int64_t Estack = D.experts + (sep ? 0 : 1);
        if (T >= kGroupedMinTokens) {
            // Q4 experts on the matrix units take F16; Q5 stays on BF16 scaled codes
            // until its kernel moves over.
            const kernels::MoeMath em = moe_math == kernels::MoeMath::WmmaBf16 && l.gate_up.bits == 4 &&
                                                l.down.bits == 4
                                            ? kernels::MoeMath::WmmaF16
                                            : moe_math;
            kernels::moe_group_routes(route_ids_.get(), T, A, Estack, group_ws_.get(), group_ws_bytes_,
                                      expert_err_.get(), stream_);
            if (em == kernels::MoeMath::WmmaF16 && l.gate_up.bits == 4) {  // SwiGLU in the gather's epilogue
                kernels::linear_q4_experts_gather_grouped_swiglu(u_.get(), l.gate_up.q4, Estack, group_ws_.get(),
                                                                 group_ws_bytes_, T, A, hh_.get(), act, em, stream_);
            } else {
                kernels::linear_qw_experts_gather_grouped(u_.get(), l.gate_up, Estack, group_ws_.get(), group_ws_bytes_,
                                                          T, A, gu_.get(), act, em, stream_);
                kernels::swiglu(gu_.get(), hh_.get(), T * A, D.inter, act, stream_);
            }
            kernels::linear_qw_experts_combine_grouped(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(),
                                                       group_ws_.get(), group_ws_bytes_, group_partial_.get(), T, A,
                                                       y_.get(), act, em, stream_);
        } else {
            kernels::linear_qw_experts_gather_swiglu(u_.get(), l.gate_up, Estack, route_ids_.get(), T, A, hh_.get(),
                                                     act, expert_err_.get(), stream_);
            kernels::linear_qw_experts_combine(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(), T, A,
                                               y_.get(), act, expert_err_.get(), stream_);
        }
        show(L + "experts_h", hh_.get(), A * D.inter, pt);  // the routed experts' down input, T rows of A slots
        if (sep) {  // the shared expert as dense projections, then y += sigmoid(its gate logit) * its output
            lin(u_.get(), l.shared_gate_up, gu_.get(), T, act);
            kernels::swiglu(gu_.get(), hh_.get(), T, D.inter, act, stream_);
            show(L + "shared_h", hh_.get(), D.inter, pt);
            lin(hh_.get(), l.shared_down, sh_y_.get(), T, act);
        }
        if (sep && !fuse_inv) {  // decode / verify: the gated add and the injection in one launch (bit-identical)
            kernels::moe_shared_add_inject(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                           D.d, (uint32_t)D.experts, x_.get(), w_in_.get(), D.H, act,
                                           router_err_.get(), stream_);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = false;
        } else {
            if (sep)
                kernels::moe_shared_add(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                        D.d, (uint32_t)D.experts, act, router_err_.get(), stream_);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = inject_y();
        }
        show(L + "out", x_.get(), n4, pt);
    }
    // Indexer tails: every attention layer wrote tail_out = the other buffer this call.
    tail_cur_ = 1 - tail_cur_;

    STRIX_TRACE_SET(stage, "head");
    hc_mix(m_.final_mix(), false, inv_ready);
    show("final_mixed", u_.get(), D.d, pt);
    if (n_logits > 0)
        lin(at(u_, (T - n_logits) * D.d), m_.lm_head(), logits_.get(), n_logits, Act::F32);
    const bool cands = cand_n_valid > 0 && n_logits > 0;
    if (cands) {
        auto *cd = reinterpret_cast<kernels::LogitCand *>(cand_dev_.get());
        if (mask_rows > 0) {
            STRIX_HIP_CHECK(hipMemcpyAsync(mask_dev_.get(), mask_host_.get(), (size_t)(mask_rows * mask_words) * 4,
                                           hipMemcpyHostToDevice, stream_),
                            "Qwen4ExpSession::forward: structured-output masks upload");
            STRIX_HIP_CHECK(hipEventRecord(mask_uploaded_, stream_), "Qwen4ExpSession::forward: mask upload event");
            mask_upload_recorded_ = true;
        }
        kernels::logits_topk(logits_.get(), n_logits, D.vocab, cand_n_valid, cd,
                             reinterpret_cast<uint32_t *>(cd + n_logits * kernels::kLogitCands), cand_ws_.get(),
                             cand_ws_bytes_, stream_, mask_rows > 0 ? mask_dev_.get() : nullptr,
                             mask_rows > 0 ? mask_words : 0);
    }

    // MTP: the layer's K / V (and indexer keys) for these positions, each from its token and the trunk's streams
    // after the position before - what a later draft attends to. The rest of the
    // layer only matters for a draft's own row, so it's skipped here.
    if (mtp_) {
        const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
        mtp_input(T, mtp_prev_.get());
        hc_mix(ml.hc_attn, true, false, mtp_x_.get());
        lin(u_.get(), ml.qkv, proj_.get(), T, act);
        const kernels::NormRopeJob nr[2] = {
            {at(proj_, D.hq * 2 * hd), D.astride, hd, k_cache_mtp_.get() + (size_t)(pos_ * D.hkv * hd) * e, D.hkv * hd, hd,
             D.hkv, hd, ml.k_norm},
            {at(proj_, D.hq * 2 * hd + D.hkv * hd), D.astride, hd, v_cache_mtp_.get() + (size_t)(pos_ * D.hkv * hd) * e,
             D.hkv * hd, hd, D.hkv, hd, nullptr}};
        kernels::qk_norm_rope_jobs(nr, 2, T, D.eps, D.rot, rope0, m_.inv_freq(), act, stream_, m_.rope_scale(),
                                   mk_jobs_);  // k, v into the MTP cache
        kernels::qsa_block_keys(at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, T, pos_,
                                pos_ % kernels::kQsaBlock ? tail_mtp_[mtp_tail_cur_].get() : nullptr,
                                tail_mtp_[1 - mtp_tail_cur_].get(), D.idx_d, ml.idx_k_norm, D.eps, m_.inv_freq(),
                                D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale(), mk_block_);
        mtp_tail_cur_ = 1 - mtp_tail_cur_;
        STRIX_HIP_CHECK(hipMemcpyAsync(mtp_prev_.get(), at(x_, (T - 1) * n4), (size_t)n4 * e, hipMemcpyDeviceToDevice,
                                       stream_),
                        "MTP previous streams");
    }
    std::vector<float> logits = cands ? read_back(0, "", n_logits) : read_back((size_t)(n_logits * D.vocab), "");
    pos_ += T;
    broken_ = false;
    mtp_chain_step_ = -1;
    return logits;
}

std::vector<float> Qwen4ExpSession::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits,
                                                   const std::vector<int64_t> &pos3) {
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward_verify: the last one awaits keep_verify / drop_verify");
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward_verify: an earlier call failed midway; reset() the session first");
    // mRoPE: the verify's rows go through the same pending path as a forward's (set_positions validates them);
    // the forward below appends them (and materializes the scalar ones) to the history, and keep / drop
    // finalize or truncate it. A scalar verify with an interleaved span pending would silently drop it - refused.
    STRIX_CHECK(pos3.empty() || (int64_t)pos3.size() / 3 == (int64_t)ids.size(),
                "Qwen4ExpSession::forward_verify: pos3 has ", pos3.size() / 3, " rows, the verify runs ",
                ids.size(), " - one triple per row");
    if (!pos3.empty()) set_positions(pos3);
    else STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::forward_verify: scalar verify while set_positions holds ",
                     pending3_.size() / 3, " interleaved rows - pass them as pos3, or clear first");
    verify_hist_before_ = pos3_host_;              // the forward below appends the rows; keep/drop decide
    verify_hist_start_before_ = pos3_start_;
    verify_delta_before_ = rope_delta_;
    verify_ever_before_ = history_ever_interleaved_;
    if (conv_spare_.empty()) {
        for (size_t l = 0; l < conv_state_.size(); ++l) {
            const std::string tag = "spare layer " + std::to_string(l);
            conv_spare_.emplace_back(conv_state_[l].size() ? DeviceBuffer<uint8_t>(conv_state_[l].size(), "conv state " + tag)
                                                           : DeviceBuffer<uint8_t>());
            rec_spare_.emplace_back(rec_state_[l].size() ? DeviceBuffer<float>(rec_state_[l].size(), "recurrent state " + tag)
                                                         : DeviceBuffer<float>());
        }
        if (ple_state_.size()) ple_state_kept_ = DeviceBuffer<uint8_t>(ple_state_.size(), "kept PLE conv state");
        if (mtp_) mtp_prev_kept_ = DeviceBuffer<uint8_t>(mtp_prev_.size(), "kept MTP previous streams");
        const Qwen4ExpDims &D = m_.dims();
        const size_t e = es(m_.act());
        for (size_t l = 0; l < conv_state_.size(); ++l)
            verify_in_.emplace_back(
                (size_t)(kMaxLogits * (D.is_attention[l] ? D.idx_d + (is_q8_format(kv_format_) ? 2 * D.hkv * D.hd : 0)
                                                         : D.gstride)) * e,
                "verify rows, layer " + std::to_string(l));
    }
    // What the forward changes in place (small): kept for drop_verify. The GDN states go to the spare set.
    copy_state(ple_state_, ple_state_kept_, stream_, "PLE conv state");
    copy_state(mtp_prev_, mtp_prev_kept_, stream_, "MTP previous streams");
    STRIX_CHECK((int64_t)ids.size() <= kMaxLogits, "Qwen4ExpSession::forward_verify: ", ids.size(), " tokens, expected <= ",
                kMaxLogits, " (the saved rows for keep_verify_prefix)");
    verify_pos_ = pos_, verify_tail_cur_ = tail_cur_, verify_mtp_tail_cur_ = mtp_tail_cur_, verify_hist_ = ple_hist_;
    verify_ids_ = ids;
    gdn_to_spare_ = true;
    try {
        std::vector<float> logits = forward(ids, n_logits);
        gdn_to_spare_ = false;
        verify_pending_ = true;
        return logits;
    } catch (...) {
        gdn_to_spare_ = false;
        throw;
    }
}

void Qwen4ExpSession::keep_verify() {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::keep_verify: no forward_verify to keep");
    std::swap(conv_state_, conv_spare_);
    std::swap(rec_state_, rec_spare_);
    // mRoPE: the verify's rows are kept - the history / delta already include them (forward appended them).
    verify_hist_before_.clear();
    verify_pending_ = false;
}

void Qwen4ExpSession::drop_verify() {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::drop_verify: no forward_verify to drop");
    copy_state(ple_state_kept_, ple_state_, stream_, "PLE conv state");
    copy_state(mtp_prev_kept_, mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::drop_verify");
    pos_ = verify_pos_, tail_cur_ = verify_tail_cur_, mtp_tail_cur_ = verify_mtp_tail_cur_, ple_hist_ = verify_hist_;
    // mRoPE: the verify's rows never happened - history and rope_delta revert to before it.
    pos3_host_ = verify_hist_before_, rope_delta_ = verify_delta_before_;
    pos3_start_ = verify_hist_start_before_, history_ever_interleaved_ = verify_ever_before_;
    verify_hist_before_.clear();
    mtp_chain_step_ = -1;
    rewind_to_.push_back(verify_pos_);  // rows past it were overwritten: snapshots beyond it no longer match
    verify_pending_ = false;
}

void Qwen4ExpSession::keep_verify_prefix(int64_t rows) {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::keep_verify_prefix: no forward_verify pending");
    const int64_t T = (int64_t)verify_ids_.size();
    STRIX_CHECK(rows >= 1 && rows < T, "Qwen4ExpSession::keep_verify_prefix: keep ", rows, " rows of a ", T,
                "-row verify, expected 1..", T - 1, " (all of them: keep_verify; none: drop_verify)");
    STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::keep_verify_prefix: set_positions holds ", pending3_.size() / 3,
                " interleaved rows while a verify is pending - resolve the verify first");
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t n4 = D.H * D.d, p0 = verify_pos_;
    // mRoPE: the verify's rows (kept and rejected alike) are already in the history; the block keys of a
    // block completed within the kept rows replay with its first row's triple / the scalar base + delta -
    // exactly like the verify's own forward did.
    prepare_positions(p0, rows);
    broken_ = true;  // until the state is whole again
    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen4ExpModel::Layer &l = m_.layer(i);
        const size_t li = (size_t)i;
        if (!D.is_attention[li]) {
            // GDN: the verify wrote its states to the spare set, so conv_state_ / rec_state_ are still the ones before
            // it - replay rows [0, rows) on them in place, with the verify's own path (Auto: per token), so the state
            // is exactly the verify's after that row (split invariance).
            kernels::gdn_front(verify_in_[li].get(),
                               {rows, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                               l.conv_w, conv_state_[li].get(), conv_state_[li].get(), false, l.A_log, l.dt_bias,
                               qkv_.get(), beta_.get(), g_.get(), act, stream_);
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {rows, D.gk, D.gv, D.conv_c}, rec_state_[li].get(),
                                rec_state_[li].get(), false, core_.get(), act, stream_, kernels::DeltaRulePath::Auto);
        } else {
            // Attention: the tail (and any block completed within the kept rows - same keys the verify wrote) from
            // the tail before the verify, which it only read. Same block-window triples as the verify's forward.
            kernels::qsa_block_keys(verify_in_[li].get(), D.idx_d, rows, p0,
                                    p0 % kernels::kQsaBlock ? tail_[verify_tail_cur_][li].get() : nullptr,
                                    tail_[1 - verify_tail_cur_][li].get(), D.idx_d, l.idx_k_norm, D.eps,
                                    m_.inv_freq(), D.rot, block_keys_[li].get(), cap_blocks_, act, stream_, m_.rope_scale(),
                                    mk_block_);
            // 8-bit K/V: the groups completed within the kept rows and the tail, from the verify's saved rows and
            // the tail before it (its own commit wrote the other tail buffer, which this overwrites).
            if (is_q8_format(kv_format_)) {
                const int64_t W = D.hkv * D.hd;
                const uint8_t *rows_kv = verify_in_[li].get() + (size_t)(kMaxLogits * D.idx_d) * e;
                kernels::kv8_commit(rows_kv, rows_kv + (size_t)W * e, 2 * W, rows, p0,
                                    p0 % kernels::kKv8Group ? kv8_tail_[verify_tail_cur_][li].get() : nullptr,
                                    kv8_tail_[1 - verify_tail_cur_][li].get(), kv8_store_[li].get(), kv8_cap_, W, act,
                                    kv8_err_.get(), stream_);
            }
        }
    }
    if (D.ple_layer >= 0) {
        // PLE: the history held [S state rows][T verify rows]; the state after row rows - 1 is its rows [rows, rows + S).
        const int64_t S = (D.ple_taps - 1) * D.ple_dil;
        STRIX_HIP_CHECK(hipMemcpyAsync(ple_state_.get(), ple_hist_buf_.get() + (size_t)(rows * n4) * e,
                                       (size_t)(S * n4) * e, hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: PLE state");
    }
    if (mtp_) {
        // The MTP layer's projections are the last thing the verify wrote to proj_; its streams are still in x_.
        const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
        kernels::qsa_block_keys(proj_.get() + (size_t)(D.idx_col + D.idx_h * D.idx_d) * e, D.astride, rows, p0,
                                p0 % kernels::kQsaBlock ? tail_mtp_[verify_mtp_tail_cur_].get() : nullptr,
                                tail_mtp_[1 - verify_mtp_tail_cur_].get(), D.idx_d, ml.idx_k_norm, D.eps, m_.inv_freq(),
                                D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale(), mk_block_);
        STRIX_HIP_CHECK(hipMemcpyAsync(mtp_prev_.get(), x_.get() + (size_t)((rows - 1) * n4) * e, (size_t)n4 * e,
                                       hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::keep_verify_prefix");
    // The n-gram history: the one before the verify, advanced over the kept ids.
    PleHistory h = verify_hist_;
    if (D.ple_layer >= 0) {
        std::vector<int64_t> tok(verify_ids_.begin(), verify_ids_.begin() + rows), out((size_t)(rows * 16));
        ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), rows, h, out.data());
    }
    ple_hist_ = h;
    pos_ = p0 + rows;
    // mRoPE: the verify's rows past the kept prefix never happened - truncate the history (the verify's
    // forward appended all of them, so it reaches past pos_); the rejected draft's image rows would
    // otherwise sit at sequence indices the next forward overwrites with different tokens (and triples).
    if (!pos3_host_.empty()) {
        const int64_t keep_len = pos_ - pos3_start_;
        STRIX_CHECK(keep_len >= 0 && keep_len * 3 <= (int64_t)pos3_host_.size(),
                    "Qwen4ExpSession::keep_verify_prefix: keeping ", pos_, " sequence rows from a history of [",
                    pos3_start_, ", ", pos3Hend(), ") - internal mismatch");
        pos3_host_.resize((size_t)(keep_len * 3));
    }
    tail_cur_ = 1 - verify_tail_cur_, mtp_tail_cur_ = 1 - verify_mtp_tail_cur_;
    mtp_chain_step_ = -1;
    rewind_to_.push_back(pos_);  // the verify's rows past pos_ get overwritten: snapshots beyond it no longer match
    verify_pending_ = false;
    broken_ = false;
}

void Qwen4ExpSession::set_mtp_vocab(int64_t n) {
    STRIX_CHECK(n >= 1 && n <= m_.dims().vocab, "Qwen4ExpSession::set_mtp_vocab: ", n, ", expected 1..",
                m_.dims().vocab);
    mtp_vocab_ = n;
}

void Qwen4ExpSession::mtp_input(int64_t T, const void *prev0) {
    STRIX_CHECK(mtp_, "Qwen4ExpSession::mtp_input: MTP is off");
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::mtp_input: ", T, " rows, expected 1..", max_tokens_);
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t n4 = D.H * D.d;
    const Qwen4ExpModel::MtpHead &mtp = m_.mtp();
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    auto lin = [&](const void *x, const QWeightView &w, void *y, int64_t M) {
        if (wmma && M >= kWmmaMinTokens) kernels::linear_qw_wmma(x, w, y, M, act, act, stream_);
        else kernels::linear_qw(x, w, y, M, act, act, stream_);
    };
    // e = rmsnorm_zc(embed(token)) [T, d]; h = the previous streams, each stream normed [T, H, d].
    if (m_.is_embed_q8())
        kernels::embedding_lookup(ids_.get(), T, m_.embed_q8(), mtp_emb_.get(), act, emb_err_.get(), stream_);
    else
        kernels::embedding_lookup(ids_.get(), T, m_.embed(), D.vocab, D.d, mtp_emb_.get(), act, emb_err_.get(), stream_);
    kernels::rmsnorm_zc(mtp_emb_.get(), mtp.pre_fc_norm_embedding, mtp_norm_emb_.get(), T, D.d, D.eps, act, stream_);
    STRIX_CHECK(prev0 != nullptr, "Qwen4ExpSession::mtp_input: null previous streams");
    kernels::group_rmsnorm_zc(prev0, mtp.pre_fc_norm_hidden, mtp_norm_hid_.get(), 1, D.H, D.d, D.eps, act, stream_);
    if (T > 1)
        kernels::group_rmsnorm_zc(x_.get(), mtp.pre_fc_norm_hidden, mtp_norm_hid_.get() + (size_t)n4 * e, T - 1, D.H,
                                  D.d, D.eps, act, stream_);
    // Per stream s: x_s = fc_embedding(e) + fc_hidden(h_s) - fc_hidden over the T * H stream rows ([T, H, d] is
    // [T * H, d]), then fc_embedding's row added to each of its token's streams (an injection with weight 1).
    lin(mtp_norm_emb_.get(), mtp.fc_embedding, mtp_proj_emb_.get(), T);
    lin(mtp_norm_hid_.get(), mtp.fc_hidden, mtp_x_.get(), T * D.H);
    kernels::hc_inject(mtp_x_.get(), mtp_ones_.get(), mtp_proj_emb_.get(), T, D.H, D.d, act, stream_);
}

std::vector<float> Qwen4ExpSession::forward_mtp(int32_t token_id, int64_t step) {
    run_mtp(token_id, step);
    std::vector<float> logits = read_back((size_t)mtp_vocab_, " MTP");
    mtp_done(step);
    return logits;
}

Qwen4ExpSession::MtpTop2 Qwen4ExpSession::forward_mtp_top2(int32_t token_id, int64_t step) {
    run_mtp(token_id, step);
    if (mtp_pick_.size() == 0) mtp_pick_ = DeviceBuffer<uint8_t>(sizeof(kernels::MtpPick), "MTP pick");
    auto *dev = reinterpret_cast<kernels::MtpPick *>(mtp_pick_.get());
    kernels::mtp_pick(logits_.get(), mtp_vocab_, router_err_.get(), expert_err_.get(), attn_err_.get(),
                      emb_err_.get(), dev, stream_);
    // One copy into the pinned area (the logits' part - unused here) behind the forward, one synchronize.
    STRIX_CHECK(readback_.size() >= kReadbackLogitsAt + sizeof(kernels::MtpPick),
                "Qwen4ExpSession::forward_mtp_top2: pinned area of ", readback_.size(), " bytes");
    auto *host = reinterpret_cast<kernels::MtpPick *>(static_cast<uint8_t *>(readback_.get()) + kReadbackLogitsAt);
    STRIX_HIP_CHECK(hipMemcpyAsync(host, dev, sizeof(kernels::MtpPick), hipMemcpyDeviceToHost, stream_),
                    "Qwen4ExpSession::forward_mtp_top2: the pick");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::forward_mtp_top2: end of the draft");
    const kernels::MtpPick p = *host;
    kernels::report_router_error(p.err, "Qwen4ExpSession MTP router");
    kernels::report_expert_error(p.err + 3, "Qwen4ExpSession MTP experts");
    kernels::report_attention_error(p.err + 6, "Qwen4ExpSession MTP QSA attention");
    kernels::report_embedding_error(p.emb, m_.dims().vocab);
    STRIX_CHECK(p.nan || (p.best >= 0 && p.best < mtp_vocab_), "Qwen4ExpSession::forward_mtp_top2: pick ", p.best,
                " outside the draft vocabulary [0, ", mtp_vocab_, ")");
    mtp_done(step);
    return {p.best, p.best_v, p.second_v, p.nan != 0};
}

void Qwen4ExpSession::mtp_done(int64_t step) {
    broken_ = false;
    mtp_chain_step_ = step, mtp_chain_pos_ = pos_;
}

void Qwen4ExpSession::run_mtp(int32_t token_id, int64_t step) {
    STRIX_TRACE_RANGE("draft " + std::to_string(step));
    STRIX_CHECK(mtp_, "Qwen4ExpSession::forward_mtp: the session was made without MTP");
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward_mtp: an earlier call failed midway; reset() the session first");
    const Qwen4ExpDims &D = m_.dims();
    STRIX_CHECK(token_id >= 0 && token_id < D.vocab, "Qwen4ExpSession::forward_mtp: token ", token_id,
                " outside [0, ", D.vocab, ")");
    STRIX_CHECK(step >= 0, "Qwen4ExpSession::forward_mtp: step ", step, ", expected >= 0");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward_mtp: a forward_verify awaits keep_verify / drop_verify");
    STRIX_CHECK(step == 0 || (mtp_chain_pos_ == pos_ && mtp_chain_step_ == step - 1), "Qwen4ExpSession::forward_mtp: step ",
                step, " doesn't continue a chain (last step ", mtp_chain_step_, " at position ", mtp_chain_pos_,
                ", now at ", pos_, ")");
    const int64_t p = pos_ + step;  // this row's sequence position (cache row, dense checks; rope angle below)
    STRIX_CHECK(p < capacity_, "Qwen4ExpSession::forward_mtp: position ", p, " is at the capacity ", capacity_);
    STRIX_CHECK(pending3_.empty(), "Qwen4ExpSession::forward_mtp: set_positions holds ", pending3_.size() / 3,
                " interleaved rows for the next forward - the draft runs at pos() with the trunk's positions; "
                "forward them first");
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const int64_t T = 1, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    // mRoPE: the draft row continues the (scalar) text span, so its angle sits at rope_pos(p) = p +
    // rope_delta; the indexer block window may reach an image tail, so its block keys use the history's
    // triples. History rows past p can't exist (no forward ran since the last keep / drop).
    prepare_positions(p, T);
    const int64_t rope_p = mk_jobs_.pos3 == nullptr ? rope_pos(p) : p;
    const Qwen4ExpModel::MtpHead &mtp = m_.mtp();
    const Qwen4ExpModel::Layer &l = mtp.layer;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    auto hc_mix = [&](const Qwen4ExpModel::Hc &hc, bool inject) {
        float *w_in = inject ? w_in_.get() : nullptr;
        if (hc.down.bits == 8)
            kernels::hc_mix_down(mtp_x_.get(), hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act, stream_);
        else
            kernels::hc_mix_down(mtp_x_.get(), hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act, stream_);
        if (hc.up.bits == 8)
            kernels::hc_mix_up(h_.get(), hc.up.q8, mtp_x_.get(), hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up(h_.get(), hc.up.q4, mtp_x_.get(), hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
    };
    broken_ = true;
    mtp_chain_step_ = -1;
    // Indexer tails: step 0 continues the committed tail into the spare buffer (not flipped); chain steps continue
    // from the step before into their own pair.
    const void *tail_in = step == 0 ? tail_mtp_[mtp_tail_cur_].get()
                          : step == 1 ? tail_mtp_[1 - mtp_tail_cur_].get()
                                      : mtp_chain_tail_[step % 2].get();
    void *tail_out = step == 0 ? tail_mtp_[1 - mtp_tail_cur_].get() : mtp_chain_tail_[(step + 1) % 2].get();
    if (step > 0) {  // the step before's output streams, kept by swapping buffers (mtp_input overwrites mtp_x_)
        STRIX_CHECK(mtp_chain_prev_.size() == mtp_x_.size(), "Qwen4ExpSession: MTP chain buffers ", mtp_chain_prev_.size(),
                    " / ", mtp_x_.size(), " bytes, expected equal (they're swapped)");
        std::swap(mtp_x_, mtp_chain_prev_);
    }
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), &token_id, 4, hipMemcpyHostToDevice, stream_), "uploading the MTP token id");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    mtp_input(T, step == 0 ? mtp_prev_.get() : mtp_chain_prev_.get());

    // The layer, as a trunk full-attention layer at position p. Step 0's KV row and block keys are the ones the next
    // forward's catch-up writes again from the same inputs; a chain step's, until the position runs, its guess.
    hc_mix(l.hc_attn, true);
    uint8_t *kc = k_cache_mtp_.get(), *vc = v_cache_mtp_.get();
    kernels::linear_qw(u_.get(), l.qkv, proj_.get(), T, act, act, stream_);
    // q, k, v and (past the dense limit) the indexer queries in one launch, as in the trunk's layers.
    const bool qsa_layer = p + T > D.dense_key_limit();
    const kernels::NormRopeJob nr[4] = {
        {proj_.get(), D.astride, 2 * hd, proj_.get(), D.astride, 2 * hd, D.hq, hd, l.q_norm},
        {at(proj_, D.hq * 2 * hd), D.astride, hd, kc + (size_t)(p * D.hkv * hd) * e, D.hkv * hd, hd, D.hkv, hd, l.k_norm},
        {at(proj_, D.hq * 2 * hd + D.hkv * hd), D.astride, hd, vc + (size_t)(p * D.hkv * hd) * e, D.hkv * hd, hd, D.hkv,
         hd, nullptr},
        {at(proj_, D.idx_col), D.astride, D.idx_d, at(proj_, D.idx_col), D.astride, D.idx_d, D.idx_h, D.idx_d,
         l.idx_q_norm}};
    kernels::qk_norm_rope_jobs(nr, qsa_layer ? 4 : 3, T, D.eps, D.rot, rope_p, m_.inv_freq(), act, stream_,
                               m_.rope_scale(), mk_jobs_);
    kernels::qsa_block_keys(at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, T, p,
                            p % kernels::kQsaBlock ? tail_in : nullptr, tail_out, D.idx_d, l.idx_k_norm, D.eps, m_.inv_freq(),
                            D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale(), mk_block_);
    const kernels::AttentionShape as{T, p, D.hq, D.hkv, hd, D.astride, 2 * hd};
    const float scale = 1.0f / std::sqrt((float)hd);
    if (!qsa_layer) {
        kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(), attn_ws_.get(),
                           attn_ws_bytes_, act, stream_);
    } else {
        const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
        void *iq = at(proj_, D.idx_col);  // normed + roped by the qk_norm_rope_jobs above
        kernels::qsa_scores(iq, D.astride, D.idx_d, T, p, D.idx_h, D.idx_d, block_keys_mtp_.get(), cap_blocks_,
                            qsa_scores_.get(), cap_blocks_, act, stream_);
        kernels::qsa_topk(qsa_scores_.get(), cap_blocks_, T, p, qsa_k, qsa_sel_.get(), qsa_nsel_.get(),
                          qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
        kernels::attention_gathered(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(), qsa_nsel_.get(),
                                    qsa_k, scale, core_.get(), attn_ws_.get(), attn_ws_bytes_, attn_err_.get(), act,
                                    stream_);
    }
    kernels::linear_qw(core_.get(), l.o_proj, y_.get(), T, act, act, stream_);
    kernels::hc_inject(mtp_x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);

    hc_mix(l.hc_mlp, true);
    kernels::linear_bf16w(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32, stream_);
    kernels::moe_router(router_logits_.get(), D.experts + 1, T, D.experts, D.top_k,
                        sep ? nullptr : router_logits_.get() + D.experts, D.experts + 1, (int32_t)D.experts,
                        route_ids_.get(), route_coef_.get(), Act::F32, router_err_.get(), stream_);
    const int64_t Estack = D.experts + (sep ? 0 : 1);
    kernels::linear_qw_experts_gather_swiglu(u_.get(), l.gate_up, Estack, route_ids_.get(), T, A, hh_.get(), act,
                                             expert_err_.get(), stream_);
    kernels::linear_qw_experts_combine(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(), T, A, y_.get(),
                                       act, expert_err_.get(), stream_);
    if (sep) {
        kernels::linear_qw(u_.get(), l.shared_gate_up, gu_.get(), T, act, act, stream_);
        kernels::swiglu(gu_.get(), hh_.get(), T, D.inter, act, stream_);
        kernels::linear_qw(hh_.get(), l.shared_down, sh_y_.get(), T, act, act, stream_);
        kernels::moe_shared_add_inject(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T, D.d,
                                       (uint32_t)D.experts, mtp_x_.get(), w_in_.get(), D.H, act, router_err_.get(),
                                       stream_);
    } else {
        kernels::hc_inject(mtp_x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);
    }

    // The head's own collapse, then the LM head's first mtp_vocab_ rows (row-major: a prefix view) - from the model's
    // Q4 draft copy when it covers mtp_vocab_ (Qwen4ExpModel::make_draft_head_q4: drafts only, the verify decides),
    // else from the LM head as loaded.
    hc_mix(mtp.hc_mixer, false);
    const QWeightView &draft = m_.draft_head_q4();
    QWeightView head = draft.bits != 0 && mtp_vocab_ <= draft.N() ? draft : m_.lm_head();
    STRIX_CHECK(head.K() == m_.lm_head().K() && mtp_vocab_ <= head.N(), "Qwen4ExpSession::forward_mtp: draft head [",
                head.N(), ", ", head.K(), "] for mtp_vocab ", mtp_vocab_, " and the LM head's K ", m_.lm_head().K());
    switch (head.bits) {
        case 8: head.q8.N = mtp_vocab_; break;
        case 6: head.q6.N = mtp_vocab_; break;
        case 5: head.q5.N = mtp_vocab_; break;
        case 4: head.q4.N = mtp_vocab_; break;
        default: STRIX_FAIL("Qwen4ExpSession::forward_mtp: LM head has ", head.bits, " bits, expected 4, 5, 6 or 8");
    }
    kernels::linear_qw(u_.get(), head, logits_.get(), 1, act, Act::F32, stream_);
}

}  // namespace strix
