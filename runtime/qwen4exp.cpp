#include "runtime/qwen4exp.hpp"

#include "runtime/ngram_table.hpp"

#include "formats/q4_from_q8.hpp"

#include "common/bf16.hpp"
#include "common/hip_check.hpp"
#include "common/log.hpp"
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

#include <atomic>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
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

// The WMMA attention over the 8-bit cache scores in F16 (kernels/kv8_math.hpp f16x2): every q and K value must fit
// F16's range. After the zero-centred RMSNorm a head's values have sum((y_i / (1 + w_i))^2) <= D, so a RoPE pair has
// |y'| <= sqrt(D) max|1 + w| cos_sin_scale - a bound from the weights alone, checked once here (kF16AttnBound
// leaves 2x for BF16 rounding and the 8-bit K's quantization step).
constexpr double kF16AttnBound = 32768.0;
void check_f16_attn_bound(const float *w_dev, int64_t D, float cos_sin_scale, const std::string &name) {
    std::vector<float> w((size_t)D);
    STRIX_HIP_CHECK(hipMemcpy(w.data(), w_dev, (size_t)D * 4, hipMemcpyDeviceToHost), "Qwen4ExpModel: read ", name);
    double m = 0;
    for (float x : w) {
        STRIX_CHECK(std::isfinite(x), "Qwen4ExpModel: ", name, " has a non-finite weight ", x);
        m = std::max(m, std::fabs(1.0 + (double)x));
    }
    const double bound = std::sqrt((double)D) * m * std::max(1.0, (double)cos_sin_scale);
    STRIX_CHECK(bound <= kF16AttnBound, "Qwen4ExpModel: ", name, " bounds the normed values by ", bound,
                " (sqrt(", D, ") x max|1 + w| ", m, " x RoPE scale ", cos_sin_scale, "), over ", kF16AttnBound,
                " - the 8-bit K/V attention scores in F16 (kernels/kv8_math.hpp) and needs them inside F16's range");
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
    dims_ = qwen4exp_dims_from_file(f);  // the layer count and kind pattern, read off the index
    Qwen4ExpDims &D = dims_;
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
    // How many routed experts per layer: the tensors decide (strixite PR #35), like the shared expert's placement.
    D.experts = qwen4exp_routed_experts(f.find(kLm + "layers.0.mlp.experts_gate_up"), D, shared_separate_,
                                        "Qwen4ExpModel: '" + weights + "' " + kLm + "layers.0.mlp.experts_gate_up");
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
        const bool attn = D.is_attention[(size_t)i];  // the kinds (and their one-of check) come from dims_from_file
        Layer l;
        l.hc_attn = load_hc(*w_, L + "attn_hyper_connection.", D, true);
        l.hc_mlp = load_hc(*w_, L + "mlp_hyper_connection.", D, true);
        if (attn) {
            const std::string A = L + "self_attn.";
            l.qkv = w_->qw(A + "qkv", {D.astride, D.d});
            l.o_proj = w_->qw(A + "o_proj.weight", {D.d, D.hq * D.hd});
            l.q_norm = w_->f32(A + "q_norm.weight", {D.hd});
            l.k_norm = w_->f32(A + "k_norm.weight", {D.hd});
            check_f16_attn_bound(l.q_norm, D.hd, D.rope_scale, A + "q_norm.weight");
            check_f16_attn_bound(l.k_norm, D.hd, D.rope_scale, A + "k_norm.weight");
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
        STRIX_CHECK(vision_->dims().out_hidden == D.d, "Qwen4ExpModel: vision tower '", vision, "' outputs ",
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
    STRIX_CHECK(f == KvFormat::Act || f == KvFormat::Q8, "kv_format_name: unknown KvFormat ", (int)f);
    return f == KvFormat::Q8 ? "q8" : "bf16";
}

KvFormat parse_kv_format(const std::string &v) {
    if (v == "bf16") return KvFormat::Act;
    if (v == "q8") return KvFormat::Q8;
    STRIX_FAIL("parse_kv_format: '", v, "', expected bf16 or q8");
}

// --------------------------------------------------------------------------- the unified K/V reservation (L2a)
// The same sizes the session's per-layer buffers are built with (their expressions, this one place).
std::vector<KvArraySpec> kv_slot_specs(const Qwen4ExpDims &D, size_t act_bytes, int64_t capacity, bool mtp,
                                        KvFormat kv) {
    STRIX_CHECK(D.layers >= 1 && (int64_t)D.is_attention.size() == D.layers,
                "kv_slot_specs: dims with ", D.layers, " layers but ", D.is_attention.size(), " kind entries");
    STRIX_CHECK(capacity >= 1, "kv_slot_specs: capacity = ", capacity, ", expected >= 1");
    STRIX_CHECK(act_bytes == 2 || act_bytes == 4, "kv_slot_specs: activation element size ", act_bytes,
                ", expected 2 (BF16) or 4 (FP32)");
    const int64_t cap_blocks = capacity / kernels::kQsaBlock + 1;
    const int64_t kv8_cap = (capacity + kernels::kKv8Group - 1) / kernels::kKv8Group * kernels::kKv8Group;
    const int64_t W = D.hkv * D.hd;
    const bool q8 = kv == KvFormat::Q8;
    std::vector<KvArraySpec> specs;
    for (int64_t i = 0; i < D.layers; ++i) {
        if (!D.is_attention[(size_t)i]) continue;
        const std::string tag = "layer " + std::to_string(i);
        if (q8) specs.push_back({"8-bit K/V " + tag, kernels::kv8_bytes(kv8_cap, W)});
        else {
            specs.push_back({"K cache " + tag, (size_t)(capacity * W) * act_bytes});
            specs.push_back({"V cache " + tag, (size_t)(capacity * W) * act_bytes});
        }
        specs.push_back({"block keys " + tag, (size_t)(cap_blocks * D.idx_d) * act_bytes});
    }
    if (mtp) {  // the head is one full-attention layer: its K / V / keys beside the trunk's
        specs.push_back({"MTP K cache", (size_t)(capacity * D.hkv * D.hd) * act_bytes});
        specs.push_back({"MTP V cache", (size_t)(capacity * D.hkv * D.hd) * act_bytes});
        specs.push_back({"MTP block keys", (size_t)(cap_blocks * D.idx_d) * act_bytes});
    }
    STRIX_CHECK(!specs.empty(), "kv_slot_specs: no attention layers in a ", D.layers, "-layer model");
    return specs;
}

size_t KvReservationLayout::slot_offset(int64_t slot) const {
    STRIX_CHECK(slot >= 0 && slot < slots, "KvReservationLayout::slot_offset: slot ", slot, ", expected 0..",
                slots - 1);
    return (size_t)slot * slot_region_bytes;  // == slot * slot_bytes while generations == 1
}

size_t KvReservationLayout::array_offset(int64_t slot, size_t spec_i) const {
    STRIX_CHECK(spec_i < specs.size(), "KvReservationLayout::array_offset: spec ", spec_i, ", expected < ",
                specs.size());
    return slot_offset(slot) + array_offsets[spec_i];
}

size_t KvReservationLayout::parked_gen_offset(int64_t p) const {
    STRIX_CHECK(p >= 0 && p < parked_generations, "KvReservationLayout::parked_gen_offset: parked generation ",
                p, ", expected 0..", parked_generations - 1);
    return parked_offset + (size_t)p * parked_bytes;
}

KvReservationLayout kv_reservation_layout(const Qwen4ExpDims &D, size_t act_bytes, int64_t capacity, bool mtp,
                                          KvFormat kv, int64_t slots, size_t page_bytes, int64_t generations,
                                          int64_t parked_generations) {
    STRIX_CHECK(page_bytes > 0, "kv_reservation_layout: page of 0 bytes");
    STRIX_CHECK(slots >= 1, "kv_reservation_layout: slots = ", slots, ", expected >= 1");
    STRIX_CHECK(generations >= 1, "kv_reservation_layout: generations = ", generations, ", expected >= 1");
    STRIX_CHECK(generations <= UINT32_MAX, "kv_reservation_layout: generations = ", generations,
                ", past the uint32 generation cursor of PagedDeviceBuffer");
    STRIX_CHECK(parked_generations >= 1, "kv_reservation_layout: parked_generations = ", parked_generations,
                ", expected >= 1");
    STRIX_CHECK(parked_generations <= UINT32_MAX, "kv_reservation_layout: parked_generations = ",
                parked_generations, ", past the uint32 parked-generation cursor");
    KvReservationLayout L;
    L.page_bytes = page_bytes;
    L.specs = kv_slot_specs(D, act_bytes, capacity, mtp, kv);
    L.slots = slots;
    size_t off = 0;
    for (const KvArraySpec &s : L.specs) {
        STRIX_CHECK(s.bytes > 0 && s.bytes <= SIZE_MAX - page_bytes, "kv_reservation_layout: '", s.name,
                    "' capacity ", s.bytes, " B overflows when rounded to ", page_bytes, " B pages");
        L.array_offsets.push_back(off);
        off += (s.bytes + page_bytes - 1) / page_bytes * page_bytes;
    }
    L.slot_bytes = off;  // already page-aligned: every carve is
    STRIX_CHECK((uint64_t)generations <= (uint64_t)SIZE_MAX / off, "kv_reservation_layout: ", generations,
                " generations of ", off, " B overflow the slot region");
    L.generations = generations;
    L.slot_region_bytes = (size_t)generations * off;
    L.parked_offset = (size_t)slots * L.slot_region_bytes;
    L.parked_bytes = L.slot_bytes;  // one conversation's pages, the size a parked slot's K/V region needs
    STRIX_CHECK((uint64_t)parked_generations <= (uint64_t)SIZE_MAX / L.parked_bytes,
                "kv_reservation_layout: ", parked_generations, " parked generations of ", L.parked_bytes,
                " B overflow the parked region");
    L.parked_generations = parked_generations;
    L.parked_region_bytes = (size_t)parked_generations * L.parked_bytes;
    L.total_bytes = L.parked_offset + L.parked_region_bytes;
    STRIX_CHECK(L.total_bytes >= off && (uint64_t)slots * L.slot_region_bytes / L.slot_region_bytes == (uint64_t)slots,
                "kv_reservation_layout: ", slots, " slot regions of ", L.slot_region_bytes,
                " B overflow the reservation size");
    return L;
}

int64_t kv_generations_that_fit(const Qwen4ExpDims &D, size_t act_bytes, int64_t capacity, bool mtp, KvFormat kv,
                                int64_t slots, size_t page_bytes, int64_t at_most, uint64_t ceiling) {
    STRIX_CHECK(at_most >= 1 && at_most <= UINT32_MAX, "kv_generations_that_fit: at_most = ", at_most,
                ", expected 1..2^32-1");
    STRIX_CHECK(ceiling > 0, "kv_generations_that_fit: ceiling of 0 bytes");
    const KvReservationLayout one = kv_reservation_layout(D, act_bytes, capacity, mtp, kv, slots, page_bytes, 1, 1);
    // g generations take (slots + 1) x g carves: every slot region and the parked region hold g each.
    const uint64_t per_gen = (uint64_t)(slots + 1) * one.slot_bytes;
    STRIX_CHECK(per_gen / (uint64_t)(slots + 1) == one.slot_bytes, "kv_generations_that_fit: ", slots + 1,
                " carves of ", one.slot_bytes, " B overflow");
    int64_t g = (int64_t)std::min<uint64_t>((uint64_t)at_most, ceiling / per_gen);
    // The layout is the authority (page rounding lives there): step down until it fits.
    while (g >= 1 && kv_reservation_layout(D, act_bytes, capacity, mtp, kv, slots, page_bytes, g, g).total_bytes > ceiling)
        --g;
    return std::max<int64_t>(g, 0);
}

int64_t qwen4exp_routed_experts(const StrixwTensor *gate_up, const Qwen4ExpDims &D, bool shared_separate,
                                const std::string &where) {
    STRIX_CHECK(D.inter >= 1 && D.d >= 1 && D.top_k >= 1, where, ": dims inter ", D.inter, ", d ", D.d, ", top_k ",
                D.top_k, ", expected each >= 1");
    STRIX_CHECK(gate_up != nullptr, where, ": missing (layer 0's routed experts, needed for the expert count)");
    const auto &sh = gate_up->shape;
    STRIX_CHECK(sh.size() == 2 && sh[1] == D.d && sh[0] > 0 && sh[0] % (2 * D.inter) == 0, where, ": shape [",
                sh.size() > 0 ? sh[0] : -1, ", ", sh.size() > 1 ? sh[1] : -1, "] (rank ", sh.size(),
                "), expected [expert slots x ", 2 * D.inter, ", ", D.d, "]");
    const int64_t slots = sh[0] / (2 * D.inter);
    STRIX_CHECK(gate_up->experts == 0 || gate_up->experts == slots, where, ": records ", gate_up->experts,
                " experts, but its ", sh[0], " rows hold ", slots);
    const int64_t routed = slots - (shared_separate ? 0 : 1);  // a stacked shared expert is the stack's last slot
    STRIX_CHECK(routed >= D.top_k, where, ": ", routed, " routed experts (", slots, " slots, shared expert ",
                shared_separate ? "separate" : "stacked", "), fewer than the ", D.top_k, " each token selects");
    return routed;
}

Qwen4ExpDims qwen4exp_dims_from_file(const StrixwFile &f) {
    Qwen4ExpDims D;
    while (f.find(kLm + "layers." + std::to_string(D.layers) + ".attn_hyper_connection.mix_down")) ++D.layers;
    STRIX_CHECK(D.layers >= 1, "qwen4exp_dims_from_file: no decoder layers in '", f.path(), "'");
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kLm + "layers." + std::to_string(i) + ".";
        const bool attn = f.find(L + "self_attn.qkv") != nullptr;
        STRIX_CHECK(attn != (f.find(L + "linear_attn.in_proj") != nullptr), "qwen4exp_dims_from_file: layer ", i,
                    " must have exactly one of self_attn.qkv / linear_attn.in_proj");
        D.is_attention.push_back(attn);
    }
    return D;
}

// --------------------------------------------------------------------------- one conversation's state: the buffers
// (feat/multi-sequence, the design in the repo docs). The per-layer caches and states at `capacity` positions, the
// MTP head's own K/V + chain buffers with mtp, the PLE conv state; positions start at 0. The per-forward
// workspaces live in Qwen4ExpSession (shared by the slots); the verify's spare state set is allocated by the
// first forward_verify (it needs the session's kMaxLogits row sizing).
// The paged K/V arrays are carved out of the session's one shared reservation at slot's region: every slot's K/V
// and the parked-page region in one address range, so a parked conversation's pages can be mapped into another
// slot without a copy - the hand-off is only correct inside one reservation on gfx1151.
Qwen4ExpSeqState::Qwen4ExpSeqState(const Qwen4ExpModel &model, int64_t capacity, int64_t chain_rows, bool mtp,
                                   KvFormat kv, VmmReservation &kv_res, const KvReservationLayout &layout,
                                   int64_t slot) : mtp(mtp) {
    const Qwen4ExpDims &D = model.dims();
    const size_t e = es(model.act());
    const int64_t W = D.hkv * D.hd;
    const bool q8 = kv == KvFormat::Q8;
    // Attach one of this slot's paged arrays at its page-aligned carve within the shared reservation. The name is
    // both buffers' and layout's; the capacity comes from the spec, so sizes can't drift between the two. With
    // generations (kv_generations >= 2) the array gets its generation cursor: the slot region holds that many
    // generations of the carve, a reset walks the cursor forward (paged_buffer.hpp's one-use addresses).
    auto attach = [&](const std::string &name) {
        size_t i = 0;
        while (i < layout.specs.size() && layout.specs[i].name != name) ++i;
        STRIX_CHECK(i < layout.specs.size(), "Qwen4ExpSeqState: the K/V layout has no array '", name, "' for slot ",
                    slot, " - the reservation's layout and the state's buffers drifted apart (specs: ",
                    layout.specs.size(), " arrays)");
        PagedDeviceBuffer b(kv_res, layout.array_offset(slot, i), layout.specs[i].bytes, name);
        if (layout.generations > 1) b.enable_generations(layout.slot_bytes, (uint32_t)layout.generations);
        return b;
    };
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "layer " + std::to_string(i);
        const bool attn = D.is_attention[(size_t)i];
        k_cache.emplace_back(attn && !q8 ? attach("K cache " + tag) : PagedDeviceBuffer());
        v_cache.emplace_back(attn && !q8 ? attach("V cache " + tag) : PagedDeviceBuffer());
        kv8_store.emplace_back(attn && q8 ? attach("8-bit K/V " + tag) : PagedDeviceBuffer());
        for (int k = 0; k < 2; ++k)
            kv8_tail[k].emplace_back(attn && q8 ? DeviceBuffer<uint8_t>(kernels::kv8_tail_bytes(W, model.act()),
                                                                      "K/V tail " + std::to_string(k) + " " + tag)
                                                 : DeviceBuffer<uint8_t>());
        block_keys.emplace_back(attn ? attach("block keys " + tag) : PagedDeviceBuffer());
        for (int k = 0; k < 2; ++k)
            tail[k].emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "indexer tail " + tag)
                                       : DeviceBuffer<uint8_t>());
        conv_state.emplace_back(!attn ? DeviceBuffer<uint8_t>((size_t)((kernels::kGdnConvWidth - 1) * D.conv_c) * e,
                                                              "conv state " + tag)
                                      : DeviceBuffer<uint8_t>());
        rec_state.emplace_back(!attn ? DeviceBuffer<float>((size_t)(D.gv * kHD * kHD), "recurrent state " + tag)
                                     : DeviceBuffer<float>());
    }
    if (D.ple_layer >= 0) {
        const int64_t S = (D.ple_taps - 1) * D.ple_dil, n4 = D.H * D.d;
        ple_state = DeviceBuffer<uint8_t>((size_t)(S * n4) * e, "PLE conv state");
    }
    if (mtp) {  // the head's K/V / keys / tails, the trunk's streams, and the chain's own buffers
        const int64_t n4 = D.H * D.d;
        this->mtp_prev = DeviceBuffer<uint8_t>((size_t)n4 * e, "MTP previous streams");
        k_cache_mtp = attach("MTP K cache");
        v_cache_mtp = attach("MTP V cache");
        block_keys_mtp = attach("MTP block keys");
        for (int k = 0; k < 2; ++k)
            tail_mtp[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "MTP indexer tail " + std::to_string(k));
        // mtp_x_'s twin (the session's workspace): a chain step swaps the two, so the step before's output is
        // kept without a copy - same size, same activation bytes (chain_rows is the workspace's max forward rows).
        mtp_chain_prev = DeviceBuffer<uint8_t>((size_t)(chain_rows * n4) * e, "MTP chain previous streams");
        for (int k = 0; k < 2; ++k)
            mtp_chain_tail[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e,
                                                       "MTP chain indexer tail " + std::to_string(k));
    }
}

void Qwen4ExpSeqState::reclaim_paged(const Qwen4ExpModel &model, int64_t capacity, KvFormat kv) {
    const Qwen4ExpDims &D = model.dims();
    const size_t e = es(model.act());
    const int64_t cap_blocks = capacity / kernels::kQsaBlock + 1;
    const int64_t kv8_cap = (capacity + kernels::kKv8Group - 1) / kernels::kKv8Group * kernels::kKv8Group;
    const int64_t W = D.hkv * D.hd;
    const bool q8 = kv == KvFormat::Q8;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "layer " + std::to_string(i);
        const bool attn = D.is_attention[(size_t)i];
        if (attn && !q8) {
            k_cache[(size_t)i] = PagedDeviceBuffer((size_t)(capacity * W) * e, kPageBytes, "K cache " + tag);
            v_cache[(size_t)i] = PagedDeviceBuffer((size_t)(capacity * W) * e, kPageBytes, "V cache " + tag);
        }
        if (attn && q8) {
            kv8_store[(size_t)i] = PagedDeviceBuffer(kernels::kv8_bytes(kv8_cap, W), kPageBytes, "8-bit K/V " + tag);
        }
        if (attn) {
            block_keys[(size_t)i] = PagedDeviceBuffer((size_t)(cap_blocks * D.idx_d) * e, kPageBytes, "block keys " + tag);
        }
    }
    if (mtp) {
        k_cache_mtp = PagedDeviceBuffer((size_t)(capacity * D.hkv * D.hd) * e, kPageBytes, "MTP K cache");
        v_cache_mtp = PagedDeviceBuffer((size_t)(capacity * D.hkv * D.hd) * e, kPageBytes, "MTP V cache");
        block_keys_mtp = PagedDeviceBuffer((size_t)(cap_blocks * D.idx_d) * e, kPageBytes, "MTP block keys");
    }
}

// /proc/meminfo's MemAvailable in bytes (the slot allocation guard); 0 if it can't be read (the guard then refuses
// any slot past the first rather than guess).
uint64_t Qwen4ExpSession::mem_available_bytes() {
    FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kib = 0;
    bool found = false;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            found = true;
            break;
        }
    std::fclose(f);
    return found ? (uint64_t)kib * 1024 : 0;
}

void Qwen4ExpSession::for_paged(Qwen4ExpSeqState &sq, int64_t end, const std::function<void(PagedDeviceBuffer &,
                                const std::vector<PagedDeviceBuffer::Range> &)> &f) const {
    STRIX_CHECK(end >= 0 && end <= capacity_, "Qwen4ExpSession::for_paged: end = ", end, ", expected 0..", capacity_);
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    const int64_t W = D.hkv * D.hd;
    // Block keys: blocks [0, end / kQsaBlock] (the block holding row end - 1, and the one a forward of rows up to end
    // starts); the 8-bit K/V: whole groups up to end (a group's rows land in the store when it completes).
    const int64_t blocks = end == 0 ? 0 : std::min(cap_blocks_, end / kernels::kQsaBlock + 1);
    // K / V rows: kOverReadRows past end too - the attention kernels load whole key tiles (16-64 rows) and mask the
    // rows past the query's position afterwards, so a tile ending past end must not reach an unmapped page.
    constexpr int64_t kOverReadRows = 64;
    const int64_t kv_end = end == 0 ? 0 : std::min(capacity_, end + kOverReadRows);
    const int64_t g_end = std::min(kv8_cap_, (kv_end + kernels::kKv8Group - 1) / kernels::kKv8Group * kernels::kKv8Group);
    const PagedDeviceBuffer::Range kv{0, (size_t)(kv_end * W) * e};
    std::vector<PagedDeviceBuffer::Range> keys;  // the block keys' 8-dimension chunks (kernels/qsa.hpp layout)
    for (const auto &[off, bytes] : kernels::qsa_block_keys_used_ranges(cap_blocks_, D.idx_d, e, blocks))
        keys.push_back({off, bytes});
    for (size_t i = 0; i < sq.block_keys.size(); ++i) {
        if (sq.k_cache[i].capacity()) f(sq.k_cache[i], {kv});
        if (sq.v_cache[i].capacity()) f(sq.v_cache[i], {kv});
        if (sq.block_keys[i].capacity()) f(sq.block_keys[i], keys);
        if (sq.kv8_store[i].capacity()) {
            std::vector<PagedDeviceBuffer::Range> r;
            for (const auto &[off, bytes] : kernels::kv8_used_ranges(kv8_cap_, W, g_end)) r.push_back({off, bytes});
            f(sq.kv8_store[i], r);
        }
    }
    if (sq.k_cache_mtp.capacity()) {
        const PagedDeviceBuffer::Range mkv{0, (size_t)(kv_end * D.hkv * D.hd) * e};
        f(sq.k_cache_mtp, {mkv});
        f(sq.v_cache_mtp, {mkv});
        f(sq.block_keys_mtp, keys);
    }
}

void Qwen4ExpSession::ensure_rows(Qwen4ExpSeqState &sq, int64_t end) {
    if (total_capacity_ > 0) {
        int64_t total_tokens = 0;
        for (const auto &s : seqs_) {
            total_tokens += (s.get() == &sq ? end : s->ensured_rows);
        }
        if (total_tokens > total_capacity_) {
            // Evict/reclaim memory from idle slots to make room in the global pool
            for (size_t k = 0; k < seqs_.size(); ++k) {
                if (seqs_[k].get() != &sq && !seqs_[k]->verify_pending && seqs_[k]->ensured_rows > precommitted_rows_) {
                    reset_seq_state(*seqs_[k]);
                    total_tokens = 0;
                    for (const auto &s : seqs_) {
                        total_tokens += (s.get() == &sq ? end : s->ensured_rows);
                    }
                    if (total_tokens <= total_capacity_) break;
                }
            }
        }
        STRIX_CHECK(total_tokens <= total_capacity_,
                    "Qwen4ExpSession: total tokens across all active slots would reach ", total_tokens,
                    ", exceeding total-capacity ", total_capacity_);
    }
    uint64_t missing = 0;
    for_paged(sq, end, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &r) { missing += b.missing_bytes(r); });
    if (missing == 0) {
        sq.ensured_rows = std::max(sq.ensured_rows, end);
        return;
    }
    if (kv_budget_) {
        kv_budget_(missing);  // throws to refuse (nothing mapped yet)
    } else {
        const uint64_t avail = mem_available_bytes();
        STRIX_CHECK(avail >= kSlotFloorBytes + missing, "Qwen4ExpSession: the K/V for positions [0, ", end, ") needs ",
                    missing >> 20, " MiB more, MemAvailable ", avail >> 20, " MiB would leave less than the ",
                    kSlotFloorBytes >> 30, " GiB floor");
    }
    for_paged(sq, end, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &r) { b.ensure(r); });
    sq.ensured_rows = std::max(sq.ensured_rows, end);
}

void Qwen4ExpSession::precommit() {
    int64_t per_slot = capacity_;
    if (total_capacity_ > 0 && (int64_t)seqs_.size() * capacity_ > total_capacity_) {
        per_slot = total_capacity_ / (int64_t)seqs_.size();
    }
    for (size_t k = 0; k < seqs_.size(); ++k) {
        ensure_rows(*seqs_[k], per_slot);
    }
    precommitted_rows_ = per_slot;
}

uint64_t Qwen4ExpSession::kv_full_bytes() const {
    uint64_t n = 0;
    for_paged(*seqs_.front(), capacity_, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &r) {
        n += b.touched_bytes(r);
    });
    return n;
}

uint64_t Qwen4ExpSession::kv_mapped_bytes(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::kv_mapped_bytes: seq ", seq, ", expected 0..",
                n_slots() - 1);
    uint64_t n = 0;
    for_paged(*seqs_[(size_t)seq], 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) { n += b.size(); });
    return n;
}

int64_t Qwen4ExpSession::kv_generation(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::kv_generation: seq ", seq, ", expected 0..",
                n_slots() - 1);
    int64_t gen = -1;
    for_paged(*seqs_[(size_t)seq], 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) {
        if (b.generations_enabled()) gen = (int64_t)b.generation();  // one value across the slot's arrays
    });
    return gen;
}

bool Qwen4ExpSession::kv_generation_fresh(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::kv_generation_fresh: seq ", seq, ", expected 0..",
                n_slots() - 1);
    bool fresh = true;
    for_paged(*seqs_[(size_t)seq], 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) {
        if (b.generations_enabled() && !b.generation_fresh()) fresh = false;
    });
    return fresh;
}

namespace {
// One slot array's base within its region (a slot generation or a parked generation - the parked region is a
// run of slot-sized carves, so the intra-region offsets are the layout's). Found by name: the name is the
// buffer's and the spec's, and the ParkedKv record carries names.
size_t kv_array_within(const KvReservationLayout &L, const std::string &name) {
    for (size_t i = 0; i < L.specs.size(); ++i)
        if (L.specs[i].name == name) return L.array_offsets[i];
    STRIX_CHECK(false, "the K/V layout has no array '", name, "' - a parked record and the layout drifted apart");
    return 0;  // unreachable
}
// The layout's index of the array with this name (state_paged_ranges names arrays by spec index).
int64_t kv_spec_index(const KvReservationLayout &L, const std::string &name) {
    for (size_t i = 0; i < L.specs.size(); ++i)
        if (L.specs[i].name == name) return (int64_t)i;
    STRIX_CHECK(false, "the K/V layout has no array '", name, "' - the state's paged ranges and the layout "
                "drifted apart");
    return -1;  // unreachable
}
// The slot's live paged array parked under this name (the for_paged set; a resume pairs record to buffer).
PagedDeviceBuffer &kv_paged_by_name(Qwen4ExpSeqState &sq, const std::string &name, const char *fn) {
    PagedDeviceBuffer *found = nullptr;
    std::vector<PagedDeviceBuffer *> all;
    for (auto *v : {&sq.k_cache, &sq.v_cache, &sq.kv8_store, &sq.block_keys})
        for (PagedDeviceBuffer &b : *v) all.push_back(&b);
    if (sq.mtp) {  // without the head these are default-constructed (no capacity, no reservation)
        all.push_back(&sq.k_cache_mtp);
        all.push_back(&sq.v_cache_mtp);
        all.push_back(&sq.block_keys_mtp);
    }
    for (PagedDeviceBuffer *b : all)
        if (b->capacity() > 0 && b->name() == name) { found = b; break; }
    STRIX_CHECK(found != nullptr, "Qwen4ExpSession::", fn, ": the slot has no paged array '", name,
                "' to pair the parked pages with - the layout and the state's buffers drifted apart");
    return *found;
}
}  // namespace

// Parked ids carry a process-wide session epoch in their high bits: a parked generation number is per-session,
// so after a session rebuild (none in serving today, but the design allows one) an id of the old session could
// otherwise name a LIVE record of the new one and resume the wrong conversation. With the epoch, a stale id
// refuses by name. The low bits are the parked generation itself, which is never refunded - so an id is unique
// for the process's life, which is what makes it safe to hand out to the cache.
namespace {
std::atomic<uint64_t> g_parked_epoch{0};
constexpr int kParkedEpochShift = 20;                        // 1 Mi parked generations per session (budget: 1024)
constexpr int64_t kParkedGenBits = (1ll << kParkedEpochShift) - 1;
}  // namespace

// The live record behind a parked id (see park_kv): the loud refusal for one from another or an earlier
// session, for a generation never taken or whose park failed, or that was already resumed or evicted - a
// parked conversation is handed back exactly once.
const ParkedKv &Qwen4ExpSession::parked(int64_t parked_id) const {
    STRIX_CHECK(uint64_t(parked_id >> kParkedEpochShift) == parked_epoch_,
                "Qwen4ExpSession::parked: parked id ", parked_id, " is not from this session (its epoch is ",
                parked_id >> kParkedEpochShift, ", this session is ", parked_epoch_, ") - a parked record belongs "
                "to the session that carved its pages, and after a session rebuild every id of the old one is "
                "dead (hazards \"VMM\")");
    const int64_t gen = parked_id & kParkedGenBits;
    STRIX_CHECK(gen >= 0 && gen < (int64_t)parked_by_gen_.size(), "Qwen4ExpSession::parked: parked generation ",
                gen, " of id ", parked_id, ", expected 0..", (int64_t)parked_by_gen_.size() - 1);
    const ParkedKv &kv = parked_by_gen_[(size_t)gen];
    STRIX_CHECK(kv.live(), "Qwen4ExpSession::parked: parked id ", parked_id, " is not a live parked record - it "
                "was never taken, its park failed, or it was already resumed or evicted (a parked conversation "
                "is handed back exactly once)");
    return kv;
}

int64_t Qwen4ExpSession::park_kv(int seq, std::string tally_category) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::park_kv: seq ", seq, ", expected 0..",
                n_slots() - 1);
    STRIX_CHECK(kv_layout_.generations > 1, "Qwen4ExpSession::park_kv: parking needs kv-generations >= 2 (the "
                "parked pages ride the generation budgets, and a kv_generations = 1 session hands its pages to "
                "nobody); this session has ", kv_layout_.generations);
    STRIX_CHECK(!broken_, "Qwen4ExpSession::park_kv: seq ", seq, ": an earlier call failed midway - reset() first");
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(!sq.verify_pending, "Qwen4ExpSession::park_kv: seq ", seq, " has a verify pending - resolve it "
                "first (its rows live in the pages this would park)");
    STRIX_CHECK(!tally_category.empty(), "Qwen4ExpSession::park_kv: empty tally category - the parked bytes are "
                "charged to the entry's category at hand-off");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::park_kv: stream");
    // Refuse BEFORE touching anything (the exhaustion-before-teardown rule):
    // the shared parked budget first, then the slot's next generation, then something worth parking at all.
    STRIX_CHECK_KV_BUDGET(parked_used_ < kv_layout_.parked_generations,
                "Qwen4ExpSession::park_kv: parked generations exhausted - ", parked_used_, " parks of the ",
                kv_layout_.parked_generations, " the reservation holds (the shared parked budget: every park "
                "spends one parked generation and neither resume nor eviction refunds it - a clean "
                "refusal, only a restart reclaims; the gauge is strix_parked_generation_used)");
    const int64_t gen = kv_generation(seq);
    STRIX_CHECK_KV_BUDGET(gen + 1 < kv_layout_.generations, "Qwen4ExpSession::park_kv: seq ", seq, " is at K/V generation ",
                gen, " of ", kv_layout_.generations, " - parking advances the cursor like a reset does, and "
                "generation ", gen + 1, " is the exhaustion refusal");
    const uint64_t mapped = kv_mapped_bytes(seq);
    STRIX_CHECK(mapped > 0, "Qwen4ExpSession::park_kv: seq ", seq, " has no K/V pages mapped - nothing to park");
    const int64_t p = parked_used_++;  // spent when taken: a park that fails part-way spent it too (its
                                        // addresses, like every released generation's, never come back)
    ParkedKv out;
    out.reservation = kv_res_.get();
    out.parked_gen = p;
    out.source_seq = seq;
    out.tally_category = std::move(tally_category);
    char *res_base = static_cast<char *>(kv_res_->base());
    size_t mapped_arrays = 0;  // arrays whose parked-side mapping is up (map_pages is all-or-nothing per array)
    try {
        for_paged(sq, 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) {
            out.arrays.push_back({b.name(), b.park_and_next_generation()});
        });
        for (const ParkedArrayKv &a : out.arrays) out.bytes += a.pages.size() * kv_res_->page_bytes();
        STRIX_CHECK(out.bytes == mapped, "Qwen4ExpSession::park_kv: parked ", out.bytes, " bytes of handles "
                    "against ", mapped, " mapped - the page bookkeeping drifted");
        for (const ParkedArrayKv &a : out.arrays) {
            // host access for the parked mapping (probe check 2: CPU read needs host; the flag's ~4.2 us/page
            // is paid at park, off the request path - the design pass' choice, revisitable by measurement).
            map_pages(res_base + kv_layout_.parked_gen_offset(p) + kv_array_within(kv_layout_, a.name),
                      kv_res_->page_bytes(), a.pages, /*host_access=*/true, "parked " + a.name);
            ++mapped_arrays;
        }
        // The charge moves to the entry's category only once the pages really live in the parked region (the
        // arrays' own names lost it page by page inside park_and_next_generation).
        GpuMemoryTally::add(out.tally_category, (int64_t)out.bytes);
    } catch (...) {
        // The park failed: the conversation's pages are released (their RAM back; a page that would not
        // release stays charged - it really is still held), the slot half-advanced - broken, like a throwing
        // reset: the request fails clean, there is never a copy-out of the K/V to save it.
        int64_t freed = 0;
        for (size_t k = 0; k < out.arrays.size(); ++k) {
            const std::vector<PageHandle> &pgs = out.arrays[k].pages;
            if (k < mapped_arrays) {  // parked-side mapping is up: unmap it, then release
                int64_t done = 0;
                try {
                    unmap_pages(res_base + kv_layout_.parked_gen_offset(p) +
                                    kv_array_within(kv_layout_, out.arrays[k].name),
                                kv_res_->page_bytes(), pgs, /*release=*/true, "park unwind " + out.arrays[k].name,
                                &done);
                } catch (...) {
                    freed += done;
                    continue;
                }
                freed += done;
            } else {  // never mapped into the parked region: the handle alone holds the page
                for (const PageHandle &ph : pgs)
                    if (hipMemRelease(ph.handle) == hipSuccess) freed += (int64_t)kv_res_->page_bytes();
            }
        }
        if (freed != (int64_t)out.bytes)
            slog(LogLevel::Warning, "kv-parking: slot %d park failed, %zu of %zu page-bytes did not release "
                 "(held until the process ends)", seq, (out.bytes - (size_t)freed) / kv_res_->page_bytes(),
                 out.bytes / kv_res_->page_bytes());
        broken_ = true;
        throw;
    }
    slog(LogLevel::Debug, "kv-parking: slot %d parked its K/V into parked generation %lld of %lld (%zu pages, "
         "%.1f MiB kept by handle; resuming or evicting spends that generation either way)",
         seq, (long long)p, (long long)kv_layout_.parked_generations, out.bytes / kv_res_->page_bytes(),
         out.bytes / 1048576.0);
    parked_by_gen_[(size_t)p] = std::move(out);  // the session keeps the record; the caller holds its id
    // The whole paged K/V of this slot is out of it now, so no snapshot saved from it can be restored any
    // more: can_restore reasons about rewinds (rows rewritten at or below its position), and a park is a rewind
    // to nothing - the rows are gone until the record comes back (and resume_parked pushes its own rewind).
    // Without this a restore after a park would rebuild the small parts and read the slot's new, empty pages
    // as the conversation's K/V: the silent wrong memory hazards "VMM" is about.
    sq.rewind_to.push_back(0);  // (the slot's cursor stands at a fresh generation now - kv_generation_fresh)
    return (int64_t(parked_epoch_) << kParkedEpochShift) | p;
}

int64_t Qwen4ExpSession::park_and_cold(int seq, std::string tally_category) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::park_and_cold: seq ", seq, ", expected 0..",
                n_slots() - 1);
    const int64_t id = park_kv(seq, std::move(tally_category));  // refuses (and touches nothing) before tearing down
    // The slot back to a cold state - the same steps a single-seq reset takes, and the same failure rule: a
    // teardown that throws half way is a broken session, not a slot with a half-cleared state. The page release
    // inside it is skipped because the park left the slot at a fresh generation, so no generation is spent twice.
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    try {
        reset_seq_state(sq);
    } catch (...) {
        broken_ = true;
        // The id never reaches the caller: release the record here, or its pages stay held until the process
        // ends (release_parked does not need an unbroken session - eviction must always be possible).
        try {
            release_parked(id);
        } catch (const std::exception &ex) {
            slog(LogLevel::Error, "kv-parking: releasing parked id %lld after a failed cold reset failed too: %s",
                 (long long)id, ex.what());
        }
        throw;
    }
    slog(LogLevel::Debug, "kv-parking: parked id %lld left slot %d cold (position 0, its snapshots dead)",
         (long long)id, seq);
    return id;
}

void Qwen4ExpSession::resume_parked(int seq, int64_t parked_id) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::resume_parked: seq ", seq, ", expected 0..",
                n_slots() - 1);
    STRIX_CHECK(!broken_, "Qwen4ExpSession::resume_parked: seq ", seq, ": an earlier call failed midway - reset() first");
    // The id checks itself: never taken, a failed park, or already resumed or evicted all refuse here.
    parked(parked_id);  // the id checks itself: never taken, a failed park, already resumed or evicted
    // Pages only ever map back into the reservation they were carved from (hazards "VMM", probe: a
    // second-reservation read gets wrong memory) - parked() proved this record belongs to this session and is live.
    ParkedKv &parked = parked_by_gen_[(size_t)(parked_id & kParkedGenBits)];
    STRIX_CHECK(kv_layout_.generations > 1, "Qwen4ExpSession::resume_parked: this session has no generations - "
                "nothing was ever parkable in it");
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(!sq.verify_pending, "Qwen4ExpSession::resume_parked: seq ", seq, " has a verify pending - resolve it "
                "first (the slot's pages are mid-batch)");
    // Where the kept handles go: a slot whose current generation was never used (a park just left it, or a
    // reset with nothing to map since) takes them right there - no generation spent, which is what keeps a
    // switch at one generation (the park's) instead of two. A slot whose generation holds a conversation
    // releases it and advances first, like a reset. Refuse BEFORE touching: the record stays resumable.
    const bool advance = !kv_generation_fresh(seq);
    const int64_t gen = kv_generation(seq);
    STRIX_CHECK_KV_BUDGET(!advance || gen + 1 < kv_layout_.generations, "Qwen4ExpSession::resume_parked: seq ", seq,
                " is at K/V generation ", gen, " of ", kv_layout_.generations, " and holds a conversation - "
                "resuming releases it and advances the cursor like a reset does, and generation ", gen + 1,
                " is the exhaustion refusal (only a restart reclaims)");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::resume_parked: stream");
    const int64_t p = parked.parked_gen;
    char *res_base = static_cast<char *>(kv_res_->base());
    for (const ParkedArrayKv &a : parked.arrays)  // pair every array before the first step (a mismatch then
        (void)kv_paged_by_name(sq, a.name, "resume_parked");  // leaves the slot and the record untouched)
    size_t adopted = 0;
    bool advancing = advance;
    try {
        if (advance)
            for_paged(sq, 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) {
                b.release_and_next_generation();  // the conversation the slot runs now is over (the caller
            });                                   // decided it): its pages go the way a reset's do
        advancing = false;
        for (const ParkedArrayKv &a : parked.arrays) {
            kv_paged_by_name(sq, a.name, "resume_parked").adopt_pages(a.pages);  // device access; the record still
            ++adopted;                                                      // names the handles
        }
    } catch (...) {
        if (advancing) broken_ = true;  // stepped part-way through the slot's buffers: only reset() / restart sorts that out
        else  // hand the adopted arrays' pages back (the handles stay in the record, whose parked-side mapping
              // is still up: the conversation is intact, the slot's cursor is one generation further and empty)
            for (size_t k = 0; k < adopted; ++k)
                kv_paged_by_name(sq, parked.arrays[k].name, "resume_parked").discard_current_pages();
        throw;
    }
    // The slot owns the pages now (its cursor and buffers carry them forward): the parked-side mapping comes
    // down, handles kept. A failure here leaves the pages mapped twice - the record must not live on.
    try {
        for (const ParkedArrayKv &a : parked.arrays)
            unmap_pages(res_base + kv_layout_.parked_gen_offset(p) + kv_array_within(kv_layout_, a.name),
                        kv_res_->page_bytes(), a.pages, /*release=*/false, "resumed " + a.name);
    } catch (...) {
        broken_ = true;  // the slot reads the same pages the parked region still maps: trust nothing after
        throw;
    }
    GpuMemoryTally::add(parked.tally_category, -(int64_t)parked.bytes);  // the buffers' names took the charge (adopt_pages)
    slog(LogLevel::Debug, "kv-parking: slot %d resumed its K/V from parked generation %lld of %lld (%zu pages, "
         "%.1f MiB; that generation stays spent)", seq, (long long)p, (long long)kv_layout_.parked_generations,
         parked.bytes / kv_res_->page_bytes(), parked.bytes / 1048576.0);
    parked = ParkedKv();
}

void Qwen4ExpSession::release_parked(int64_t parked_id) {
    parked(parked_id);  // the id checks itself: never taken, a failed park, already resumed or evicted
    ParkedKv &parked = parked_by_gen_[(size_t)(parked_id & kParkedGenBits)];  // this session's own record, live: unmap + release
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::release_parked: stream");
    const int64_t p = parked.parked_gen;
    const std::string category = parked.tally_category;
    char *res_base = static_cast<char *>(kv_res_->base());
    int64_t freed = 0;
    bool failed = false;
    for (const ParkedArrayKv &a : parked.arrays) {  // every page gets its attempt first (a half-torn-down
        int64_t done = 0;                            // eviction is still an eviction: the record dies either way)
        try {
            unmap_pages(res_base + kv_layout_.parked_gen_offset(p) + kv_array_within(kv_layout_, a.name),
                        kv_res_->page_bytes(), a.pages, /*release=*/true, "evicted " + a.name, &done);
        } catch (...) {
            failed = true;
        }
        freed += done;
    }
    GpuMemoryTally::add(category, -freed);  // pages that failed to release stay charged - their RAM really is still held
    const size_t pages = parked.bytes / kv_res_->page_bytes();
    const size_t freed_pages = (size_t)freed / kv_res_->page_bytes();
    const int64_t src = parked.source_seq;
    const double total_mib = parked.bytes / 1048576.0;
    parked = ParkedKv();  // spent either way: the parked generation's addresses never come back
    slog(LogLevel::Debug, "kv-parking: evicted parked generation %lld of %lld (slot %d's): %.1f of %.1f MiB "
         "back to RAM", (long long)p, (long long)kv_layout_.parked_generations, (int)src, freed / 1048576.0,
         total_mib);
    STRIX_CHECK(!failed, "Qwen4ExpSession::release_parked: parked generation ", p, " (slot ", src,
                "'s) had pages that would not unmap or release - ", pages - freed_pages,
                " pages stay held until the process ends (counted under '", category, "')");
}

Qwen4ExpSession::Qwen4ExpSession(const Qwen4ExpModel &model, int64_t capacity, int64_t max_tokens,
                                 PrefillMath prefill_math, bool mtp, KvFormat kv, int64_t n_slots,
                                 int64_t kv_generations)
    : m_(model), capacity_(capacity), max_tokens_(max_tokens), prefill_math_(prefill_math), kv_format_(kv),
      mtp_(mtp) {
    STRIX_CHECK(kv == KvFormat::Act || kv == KvFormat::Q8, "Qwen4ExpSession: unknown KV format ", (int)kv);
    STRIX_CHECK(prefill_math == PrefillMath::F32 || prefill_math == PrefillMath::WmmaBf16,
                "Qwen4ExpSession: unknown prefill math ", (int)prefill_math);
    STRIX_CHECK(!mtp || model.has_mtp(), "Qwen4ExpSession: MTP asked for, but the loaded weights have no MTP head "
                "(no mtp.fc_embedding.weight)");
    STRIX_CHECK(n_slots >= 1, "Qwen4ExpSession: n_slots = ", n_slots, ", expected >= 1");
    STRIX_CHECK(kv_generations >= 1 || kv_generations == kKvGenerationsAuto, "Qwen4ExpSession: kv_generations = ",
                kv_generations, ", expected >= 1 or ", kKvGenerationsAuto, " (auto)");
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
    const bool q8 = kv == KvFormat::Q8;
    kv8_cap_ = (capacity + kernels::kKv8Group - 1) / kernels::kKv8Group * kernels::kKv8Group;
    const int64_t W = D.hkv * D.hd;
    // Slot 0's state first: everything below and every single-sequence call is per-forward workspace, but st() and
    // the checks after this need it to exist. The other slots' states come last, each only if it fits (below).
    // The unified K/V reservation: all slots' paged K/V and the parked-page region carved out of ONE
    // hipMemAddressReserve range - on gfx1151 a page mapped into a second reservation reads the wrong memory, and
    // parking moves a conversation's pages between slot regions. Address space only: it costs no RAM; pages are
    // mapped as conversations grow (ensure_rows). With kv_generations >= 2 each slot region holds that many
    // generations of the arrays: a reset releases the finished conversation's pages and the next maps fresh ones
    // at addresses this process never used (paged_buffer.hpp; the kv-generations budget of the VA policy).
    // Auto (the served default): the most generations up to kKvGenerationDefault that fit under the ceiling.
    const int64_t fit = kv_generations_that_fit(D, e, capacity, mtp, kv, n_slots, Qwen4ExpSeqState::kPageBytes,
                                                std::max(kKvGenerationDefault, kv_generations));
    if (kv_generations == kKvGenerationsAuto) {
        STRIX_CHECK(fit >= 2, "Qwen4ExpSession: kv-generations auto: not even 2 K/V generations of ", n_slots,
                    " slots at capacity ", capacity, " fit under the ", kKvReservationVaCeiling >> 40,
                    " TiB address-space ceiling (", fit, " fit) - lower total-slots or capacity");
        kv_generations = fit;  // at most kKvGenerationDefault: fit's at_most for auto
    }
    kv_layout_ = kv_reservation_layout(D, e, capacity, mtp, kv, n_slots, Qwen4ExpSeqState::kPageBytes,
                                       kv_generations, kv_generations == 1 ? 1 : kv_generations);
    STRIX_CHECK(kv_layout_.total_bytes <= kKvReservationVaCeiling,
                "Qwen4ExpSession: kv-generations ", kv_generations, " needs ", kv_layout_.total_bytes >> 30,
                " GiB of address space (", n_slots, " slots x ", kv_generations, " generations + ",
                kv_layout_.parked_generations, " parked generations of ", kv_layout_.slot_bytes >> 20,
                " MiB), past the ", kKvReservationVaCeiling >> 40,
                " TiB safety ceiling; at most ", fit, " fit with ", n_slots, " slots at capacity ", capacity,
                " - set kv-generations = 0 (auto) or at most ", fit, ", or lower total-slots or capacity");
    // The parked records, indexed by parked generation (= their id), sized with the budget so an id is in range
    // exactly when it could ever have been taken. Empty arrays cost ~120 B each: 1024 parks = 120 KiB.
    parked_epoch_ = g_parked_epoch.fetch_add(1);  // every id this session hands out carries it (see parked())
    parked_by_gen_.assign((size_t)kv_layout_.parked_generations, ParkedKv());
    kv_res_ = std::make_unique<VmmReservation>(kv_layout_.total_bytes, Qwen4ExpSeqState::kPageBytes, "K/V reservation");
    const uint64_t avail0 = mem_available_bytes();
    seqs_.push_back(std::make_unique<Qwen4ExpSeqState>(m_, capacity, max_tokens, mtp, kv, *kv_res_, kv_layout_,
                                                        /*slot=*/0));
    const uint64_t avail1 = mem_available_bytes();
    const uint64_t slot_cost = avail0 > avail1 ? avail0 - avail1 : 0;  // what one slot's state took from RAM
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
        if (wmma_attn && T >= kWmmaAttnMinTokens) {
            // The matrix-unit attention, dense at any depth below the limit.
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
    // The batched path's logits-row workspace (feat/multi-sequence): the compacted logits rows of a
    // multi-run forward, so the LM head's weights are read once for the batch. (A batched verify's q8 new
    // K | V rows need no such staging - each run writes them straight into its own saved rows, as the
    // single-run verify does.)
    lg_src_ = DeviceBuffer<uint8_t>((size_t)(std::max(M, kMaxLogits) * D.d) * e, "batched logits rows");
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
        std::vector<float> ones((size_t)(M * D.H), 1.0f);
        mtp_ones_ = DeviceBuffer<float>(ones.size(), "MTP ones");
        STRIX_HIP_CHECK(hipMemcpy(mtp_ones_.get(), ones.data(), ones.size() * 4, hipMemcpyHostToDevice), "MTP ones");
    }
    // The other slots' states (each at the full capacity), each only while it leaves MemAvailable above the floor
    // (its cost measured on slot 0's, +10% for what varies): on this machine the GPU allocates from system RAM, and RAM
    // run down this far hard-hangs it - so a total_slots far too high for the machine fails here, cleanly, before the
    // startup memory plan (serve/memory_plan.hpp) checks the full rule (margin + one full-capacity state).
    for (int64_t k = 1; k < n_slots; ++k) {
        const uint64_t avail = mem_available_bytes(), need = kSlotFloorBytes + slot_cost + slot_cost / 10;
        STRIX_CHECK(avail >= need, "Qwen4ExpSession: slot ", k, " of ", n_slots, " needs ~", slot_cost >> 20,
                    " MiB (slot 0's state at capacity ", capacity, " took that), MemAvailable ", avail >> 20,
                    " MiB would leave less than the ", kSlotFloorBytes >> 30, " GiB floor after it - ", k,
                    " slot(s) fit: lower total-slots (or the capacity)");
        seqs_.push_back(std::make_unique<Qwen4ExpSeqState>(m_, capacity, max_tokens, mtp, kv, *kv_res_, kv_layout_,
                                                            /*slot=*/k));
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
    if (bias_uploaded_) (void)hipEventDestroy(bias_uploaded_);
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

void Qwen4ExpSession::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) { prefetch_ple(0, ids, first); }

void Qwen4ExpSession::prefetch_ple(int seq, const std::vector<int32_t> &ids, int64_t first) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::prefetch_ple: seq ", seq, ", expected 0..", n_slots() - 1);
    const Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    const int64_t T = (int64_t)ids.size();
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::prefetch_ple: ", T, " ids, expected 1..", max_tokens_);
    STRIX_CHECK(first >= 0 && first < T, "Qwen4ExpSession::prefetch_ple: first = ", first, ", expected 0..", T - 1);
    if (m_.dims().ple_layer < 0) return;
    // Rows of every id (the history runs through them), then the ones asked for. While a verify awaits its outcome
    // the ids start where the verify did (a rejection's kept prefix + the corrected token).
    std::vector<int64_t> tok(ids.begin(), ids.end()), rows((size_t)(T * 16));
    PleHistory h = sq.verify_pending ? sq.verify_hist : sq.ple_hist;
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
                    slog(LogLevel::Warning, "Qwen4ExpSession::prefetch_ple: prefetch failed (the gather retries): %s",
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
std::future<void> Qwen4ExpSession::start_ple_gather(std::vector<int64_t> rows, int64_t T, PinnedHostBuffer &dst,
                                                    PleRun run) {
    const int64_t pe = m_.dims().ple_e;
    STRIX_CHECK(T >= 1 && T <= max_tokens_ && (int64_t)rows.size() == T * 16, "Qwen4ExpSession: PLE gather of ", T,
                " tokens with ", rows.size(), " row ids, expected 1..", max_tokens_, " tokens x 16");
    STRIX_CHECK(dst.size() >= (size_t)(T * pe) * es(m_.act()), "Qwen4ExpSession: PLE host buffer of ", dst.size(),
                " bytes, needs ", (size_t)(T * pe) * es(m_.act()));
    void *dstp = dst.get();
    auto job = [this, rows = std::move(rows), T, pe, dstp] {
        const auto t0 = std::chrono::steady_clock::now();
        const NgramRowSource &ng = m_.ngram_rows();
        if (m_.act() == Act::BF16) {
            ng.gather(rows.data(), (int64_t)rows.size(), static_cast<uint16_t *>(dstp));
        } else {
            std::vector<uint16_t> bits((size_t)(T * pe));
            ng.gather(rows.data(), (int64_t)rows.size(), bits.data());
            float *f = static_cast<float *>(dstp);
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
        st().pending3.clear();
        return;
    }
    const int64_t rows = (int64_t)pos3.size() / 3;
    STRIX_CHECK(!st().verify_pending, fn, ": a forward_verify awaits keep_verify / drop_verify - the next forward's "
                "rows are undefined until then");
    STRIX_CHECK(st().pending3.empty(), fn, ": positions for the next forward are already set (", st().pending3.size() / 3,
                " rows) - forward them or clear with an empty vector");
    STRIX_CHECK(rows >= 1 && rows <= max_tokens_, fn, ": ", rows, " rows, expected 1..", max_tokens_,
                " (one forward of at most max_tokens rows)");
    STRIX_CHECK(st().pos + rows <= capacity_, fn, ": rows [", st().pos, ", ", st().pos + rows, ") exceed the capacity ",
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
    // refused above, and no other call advances st().pos with a span pending.
    // The span must continue the history exactly (the vision position-walk design note). While the history is
    // empty (a text-only session that never set positions, or one resumed after the image) there is nothing
    // to continue - the first interleaved span can start anywhere valid. Once non-empty it must reach st().pos
    // (every forward extends it), and rows at or past st().pos are the pending span's.
    if (!st().pos3_host.empty()) {
        STRIX_CHECK(st().pos3Hend() == st().pos, fn, ": the triple history covers [", st().pos3_start, ", ", st().pos3Hend(),
                    ") but the position is ", st().pos, " (internal: every forward extends the history to st().pos)");
        const int64_t *last = st().pos3_host.data() + (size_t)((st().pos3Hend() - st().pos3_start - 1) * 3);
        const Cont want = continuation_of(last, st().pos3_host.data(), (int64_t)st().pos3_host.size() / 3);
        STRIX_CHECK(accepts(want, pos3.data()), fn, ": row ", st().pos, " is ", triple_str(pos3.data()), " but the "
                    "history's row ", st().pos - 1, " ", triple_str(last),
                    " continues at " + triple_str(want.exact.data()) +
                        (want.kind == ContKind::kEndOrImage ? " (or an image step at the block edge)"
                                                            : std::string()));
    } else {
        // First interleaved span of a text-only (or after-image resumed) session: its first row continues the
        // uniform prefix, so it sits at the scalar continuation of the current position.
        const int64_t want[3] = {st().pos + st().rope_delta, st().pos + st().rope_delta, st().pos + st().rope_delta};
        STRIX_CHECK(triple_eq(pos3.data(), want), fn, ": row ", st().pos, " is ", triple_str(pos3.data()), " but the "
                    "uniform prefix continues at ", triple_str(want), " (the first interleaved row must sit at "
                    "the walk's current position ", st().pos, " + delta)");
    }
    st().history_ever_interleaved = st().history_ever_interleaved || st().rope_delta != 0 || !st().pos3_host.empty();  // (delta is
    // <= 0 by invariant - nonzero only after an image packed positions ahead of the sequence indices)
    st().pending3 = std::move(pos3);
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
void Qwen4ExpSession::prepare_positions(Qwen4ExpSeqState &sq, int64_t seq, int64_t n) {
    const char *fn = "Qwen4ExpSession::prepare_positions";
    STRIX_CHECK(n >= 1 && seq >= 0 && seq + n <= capacity_, fn, ": span [", seq, ", ", seq + n, ") outside [0, ",
                capacity_, ")");
    STRIX_CHECK(sq.pending3.empty(), fn, ": set_positions rows for the forward are not in the history yet "
                "(forward consumes them first)");
    STRIX_CHECK(sq.rope_delta <= 0, fn, ": sq.rope_delta = ", sq.rope_delta, ", expected <= 0 (internal: an image "
                "block's h*w rows consume max(h,w) positions, so positions only ever pack ahead of sequence "
                "indices)");
    // keep_verify_prefix and run_mtp come through here too, and the draft head's rows can run any number of steps
    // past the last forward: the paged K/V for the span is mapped now.
    ensure_rows(sq, seq + n);
    mk_block_ = kernels::MropePos{}, mk_jobs_ = kernels::MropePos{};
    const int64_t t0 = (seq / kernels::kQsaBlock) * kernels::kQsaBlock;  // the block span's first row
    const int64_t hstart = sq.pos3_start, hend = sq.pos3Hend();  // the history holds rows [hstart, hend)
    bool span_diff = false;  // a row of the forward's span [seq, seq+n) with a non-scalar triple
    bool window_diff = false;  // ... or anywhere in the block window [t0, seq+n)
    for (int64_t r = std::max(t0, hstart); r < std::min(hend, seq + n); ++r) {
        const int64_t *tr = sq.pos3_host.data() + (size_t)((r - hstart) * 3);
        const int64_t v = (r + sq.rope_delta);
        const bool differs = !(tr[0] == v && tr[1] == v && tr[2] == v);
        span_diff = span_diff || (r >= seq && differs);
        window_diff = window_diff || differs;
    }
    const bool uploaded = window_diff || sq.rope_delta != 0;  // (delta is <= 0 once a block packed positions ahead)
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
        const int64_t v = (r + sq.rope_delta);
        stage3_.insert(stage3_.end(), {v, v, v});
    };
    for (int64_t r = t0; r < hs; ++r) push_scalar(r);
    if (hs < he)
        stage3_.insert(stage3_.end(), sq.pos3_host.begin() + (hs - hstart) * 3, sq.pos3_host.begin() + (he - hstart) * 3);
    for (int64_t r = he; r < seq + n; ++r) push_scalar(r);
    STRIX_CHECK((int64_t)stage3_.size() == rows * 3, fn, ": staged ", stage3_.size() / 3, " rows for the span [", t0,
                ", ", seq + n, ")");
    if (pos3_dev_.size() == 0)  // (get() throws on the empty buffer - size() is the allocation test)
        pos3_dev_ = DeviceBuffer<int64_t>((size_t)capacity_ * 3, "mRoPE triples");
    // stage3_ is pageable, not pinned: moving it to a pinned buffer needs an event gate before its next write
    // (the gfx1151 pinned-reuse hazard).
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
    if (kv_format_ == KvFormat::Q8)
        STRIX_HIP_CHECK(hipMemset(kv8_err_.get(), 0, 12), "Qwen4ExpSession::reset: zero 8-bit K/V error");
    // Every state (the slots) back to position 0; the error words / position upload machinery above are per-forward.
    // A reset that throws (a generation-exhausted refusal) leaves some slots stepped and some not: mark the
    // session broken so no later forward runs on that inconsistency (the generation's pages are released).
    // A budget refusal (KvBudgetExhausted) is checked for every slot before any slot resets, and touches nothing:
    // the session stays as it was, not broken.
    for (size_t k = 0; k < seqs_.size(); ++k) kv_budget_for_reset(*seqs_[k]);
    try {
        for (size_t k = 0; k < seqs_.size(); ++k) reset_seq_state(*seqs_[k]);
    } catch (...) {
        broken_ = true;
        throw;
    }
    broken_ = false;
    mk_block_ = kernels::MropePos{}, mk_jobs_ = kernels::MropePos{};
    ple_lookahead_.clear();  // a pending gather stays: the next forward's ids / history decide whether it's used
}

void Qwen4ExpSession::reset(int seq) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::reset: seq ", seq, ", expected 0..", n_slots() - 1);
    STRIX_CHECK(!broken_, "Qwen4ExpSession::reset: seq ", seq, ": an earlier call failed midway - any slot's state may "
                "be half-written, reset() them all");
    STRIX_CHECK(!seqs_[(size_t)seq]->verify_pending, "Qwen4ExpSession::reset: seq ", seq,
                " has a verify pending - resolve it first (the other runs of its batch share the forward)");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::reset(seq)");
    // As in reset(): a throwing reset (generation exhaustion) leaves the slot half-stepped - failed, not usable.
    kv_budget_for_reset(*seqs_[(size_t)seq]);  // the budget refusal touches nothing: not broken
    try {
        reset_seq_state(*seqs_[(size_t)seq]);
    } catch (...) {
        broken_ = true;
        throw;
    }
}

int Qwen4ExpSession::seq_index(const Qwen4ExpSeqState &s) const {
    for (size_t k = 0; k < seqs_.size(); ++k)
        if (seqs_[k].get() == &s) return (int)k;
    STRIX_FAIL("Qwen4ExpSession::seq_index: the state is not one of this session's ", seqs_.size(), " slots");
}

// A reset of a slot whose generation was used releases it and advances the cursor; past the last generation that
// is the budget refusal. Every buffer of the slot advances together, so the slot's cursor is the whole question,
// and asking it touches nothing: refused here, the slot still holds its conversation - which is what lets the
// engine keep serving that conversation (KvBudgetExhausted) instead of resetting into a broken session.
void Qwen4ExpSession::kv_budget_for_reset(const Qwen4ExpSeqState &s) const {
    const int slot = seq_index(s);
    if (kv_layout_.generations <= 1 || kv_generation_fresh(slot)) return;
    const int64_t gen = kv_generation(slot);
    STRIX_CHECK_KV_BUDGET(gen + 1 < kv_layout_.generations, "Qwen4ExpSession: seq ", slot, " is at K/V generation ",
                          gen, " of ", kv_layout_.generations, " and holds a conversation - a reset releases it and "
                          "advances the cursor, and generation ", gen + 1, " is the kv-generations exhaustion refusal "
                          "(one-use addresses: only a restart reclaims)");
}

void Qwen4ExpSession::reset_seq_state(Qwen4ExpSeqState &s) {
    // Paged K/V: with one generation (kv-generations = 1) the slot keeps its pages - the next conversation
    // reuses them (runtime/paged_buffer.hpp - no page is released while the buffer lives). With generations the
    // reset instead releases this conversation's pages and advances the cursor, so the next forward maps new
    // handles at addresses this process has never mapped - the one-use addresses:
    // release-then-remap at one address is the silent wrong read. Exhausting the budget refuses cleanly.
    // A generation nothing was ever mapped in (a park just left the slot there, or nothing ran since the last
    // reset) has nothing to release, and its addresses are still unused: the reset stays in it.
    bool fresh = true;
    for_paged(s, 0, [&](PagedDeviceBuffer &b, const std::vector<PagedDeviceBuffer::Range> &) {
        if (b.generations_enabled() && !b.generation_fresh()) fresh = false;
    });
    const int slot = seq_index(s);
    kv_budget_for_reset(s);  // the budget refusal first, before any buffer steps (the per-buffer check is a drift guard)
    if (!is_precommitted() && kv_layout_.generations > 1 && !fresh) {
        uint32_t gen_now = 0, gen_cap = 0;  // every buffer of the slot advances together; the log names the slot
        auto advance = [&](PagedDeviceBuffer &b) {
            if (b.capacity() == 0) return;  // the other layer kind's default-constructed buffer
            STRIX_CHECK(b.generations_enabled(), "Qwen4ExpSession: the K/V layout has generations but buffer '",
                        b.name(), "' was attached without a generation cursor - layout and state drifted apart");
            b.release_and_next_generation();
            gen_now = b.generation(), gen_cap = b.generation_capacity();
        };
        for (PagedDeviceBuffer &b : s.k_cache) advance(b);
        for (PagedDeviceBuffer &b : s.v_cache) advance(b);
        for (PagedDeviceBuffer &b : s.kv8_store) advance(b);
        for (PagedDeviceBuffer &b : s.block_keys) advance(b);
        if (mtp_) {  // without the head these are default-constructed, no reservation behind them
            advance(s.k_cache_mtp);
            advance(s.v_cache_mtp);
            advance(s.block_keys_mtp);
        }
        slog(LogLevel::Debug, "kv-generations: slot %d released its K/V generation, now at generation %u of %u "
             "(one-use addresses; reaching %u refuses the next switch - only a restart reclaims)",
             slot, gen_now, gen_cap, gen_cap);
    }
    s.ensured_rows = precommitted_rows_;
    {
        s.pos = 0, s.tail_cur = 0, s.mtp_tail_cur = 0, s.mtp_chain_step = -1, s.verify_pending = false;  // GDN states reset on the first call (reset_state); KV overwritten
        s.verify_rows = 0, s.verify_carry = 0, s.verify_row_base = 0, s.verify_ple_wbase = 0;
        s.rope_delta = 0, s.history_ever_interleaved = false, s.verify_delta_before = 0, s.verify_ever_before = false,
            s.verify_hist_start_before = 0;
        s.pos3_start = 0;
        s.pos3_host.clear(), s.pending3.clear(), s.verify_hist_before.clear();
        s.imported_at = -1;
        if (mtp_)  // position 0's MTP row pairs its token with no trunk state: zeros
            STRIX_HIP_CHECK(hipMemset(s.mtp_prev.get(), 0, s.mtp_prev.size()), "Qwen4ExpSession: zero MTP previous streams");
        if (m_.dims().ple_layer >= 0) {
            STRIX_HIP_CHECK(hipMemset(s.ple_state.get(), 0, s.ple_state.size()), "Qwen4ExpSession: zero PLE state");
            s.ple_hist = PleHistory::fresh(m_.ngram_rows().hash_params());
        }
        s.rewind_to.push_back(0);
    }
}

int64_t Qwen4ExpSession::pos(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::pos: seq ", seq, ", expected 0..", n_slots() - 1);
    return seqs_[(size_t)seq]->pos;
}

Qwen4ExpSnapshot Qwen4ExpSession::make_snapshot() const {
    return make_snapshot(0);
}

// A snapshot of one slot's state (feat/multi-sequence): the buffers are that state's; saving / restoring
// one never touches another's (the engine keeps a snapshot set per slot).
Qwen4ExpSnapshot Qwen4ExpSession::make_snapshot(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::make_snapshot: seq ", seq, ", expected 0..",
                n_slots() - 1);
    Qwen4ExpSeqState &st = *seqs_[(size_t)seq];
    const Qwen4ExpDims &D = m_.dims();
    Qwen4ExpSnapshot s;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "snapshot layer " + std::to_string(i);
        const size_t l = (size_t)i;
        s.conv_.emplace_back(st.conv_state[l].size() ? DeviceBuffer<uint8_t>(st.conv_state[l].size(), "conv state " + tag)
                                                    : DeviceBuffer<uint8_t>());
        s.rec_.emplace_back(st.rec_state[l].size() ? DeviceBuffer<float>(st.rec_state[l].size(), "recurrent state " + tag)
                                                  : DeviceBuffer<float>());
        s.tail_.emplace_back(st.tail[0][l].size() ? DeviceBuffer<uint8_t>(st.tail[0][l].size(), "indexer tail " + tag)
                                                 : DeviceBuffer<uint8_t>());
        s.kv8_tail_.emplace_back(st.kv8_tail[0][l].size() ? DeviceBuffer<uint8_t>(st.kv8_tail[0][l].size(), "K/V tail " + tag)
                                                         : DeviceBuffer<uint8_t>());
    }
    if (st.ple_state.size()) s.ple_state_ = DeviceBuffer<uint8_t>(st.ple_state.size(), "snapshot PLE conv state");
    if (mtp_) {
        s.mtp_tail_ = DeviceBuffer<uint8_t>(st.tail_mtp[0].size(), "snapshot MTP indexer tail");
        s.mtp_prev_ = DeviceBuffer<uint8_t>(st.mtp_prev.size(), "snapshot MTP previous streams");
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
    save(0, s);
}

void Qwen4ExpSession::save(int seq, Qwen4ExpSnapshot &s) {
    const char *fn = "Qwen4ExpSession::save";
    STRIX_CHECK(seq >= 0 && seq < n_slots(), fn, ": seq ", seq, ", expected 0..", n_slots() - 1);
    Qwen4ExpSeqState &st = *seqs_[(size_t)seq];
    STRIX_CHECK(!broken_, "Qwen4ExpSession::save: an earlier call failed midway; the state is not worth saving");
    STRIX_CHECK(!st.verify_pending, "Qwen4ExpSession::save: a forward_verify awaits keep_verify / drop_verify");
    STRIX_CHECK(st.pending3.empty(), "Qwen4ExpSession::save: set_positions holds ", st.pending3.size() / 3,
                " interleaved rows the position doesn't include yet - forward them first");
    STRIX_CHECK(s.conv_.size() == st.conv_state.size(), "Qwen4ExpSession::save: snapshot has ", s.conv_.size(),
                " layers, the session ", st.conv_state.size(), " (make it with make_snapshot)");
    for (size_t l = 0; l < st.conv_state.size(); ++l) {
        copy_state(st.conv_state[l], s.conv_[l], stream_, "conv state");
        copy_state(st.rec_state[l], s.rec_[l], stream_, "recurrent state");
        copy_state(st.tail[st.tail_cur][l], s.tail_[l], stream_, "indexer tail");
        copy_state(st.kv8_tail[st.tail_cur][l], s.kv8_tail_[l], stream_, "K/V tail");
    }
    copy_state(st.ple_state, s.ple_state_, stream_, "PLE conv state");
    copy_state(st.tail_mtp[st.mtp_tail_cur], s.mtp_tail_, stream_, "MTP indexer tail");
    copy_state(st.mtp_prev, s.mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::save");
    s.ple_hist_ = st.ple_hist;
    s.pos_ = st.pos;
    s.epoch_ = st.rewind_to.size() - 1;
    // mRoPE: rope_delta and the last three history rows (the continuation rule a resume needs; whole-history
    // resume isn't a thing until image serving refuses less, vision Phase A - and restore
    // refuses an image-row tail outright, so what a snapshot really carries is the delta and the proof the
    // tail is text). Throws when the tail can't derive a continuation (a saved row in the middle of a single
    // image block row with nothing to derive the width from) - guessing is not an option.
    s.rope_delta_ = st.rope_delta;
    const int64_t hlen = (int64_t)st.pos3_host.size() / 3;
    s.ntriples_ = std::min<int64_t>(3, hlen);
    s.triples_.assign(st.pos3_host.end() - (size_t)(s.ntriples_ * 3), st.pos3_host.end());
    s.first_cont_ = hlen > 0
                        ? continuation_value_of(st.pos3_host.data() + (size_t)((hlen - 1) * 3), st.pos3_host.data(),
                                                hlen, "Qwen4ExpSession::save")
                        : std::array<int64_t, 3>{st.pos + st.rope_delta, st.pos + st.rope_delta, st.pos + st.rope_delta};
}

bool Qwen4ExpSession::can_restore(const Qwen4ExpSnapshot &s) const {
    return can_restore(0, s);
}

bool Qwen4ExpSession::can_restore(int seq, const Qwen4ExpSnapshot &s) const {
    if (seq < 0 || seq >= n_slots()) return false;
    const Qwen4ExpSeqState &st = *seqs_[(size_t)seq];
    if (s.pos_ < 0 || s.conv_.size() != st.conv_state.size() || s.epoch_ >= st.rewind_to.size()) return false;
    for (size_t e = s.epoch_ + 1; e < st.rewind_to.size(); ++e)
        if (st.rewind_to[e] < s.pos_) return false;
    // Every rewind since was to >= s.pos_, and forwards only write at or past the position they start from, so
    // KV and block keys below s.pos_ are the ones the snapshot's state was built with.
    return true;
}

void Qwen4ExpSession::restore(const Qwen4ExpSnapshot &s) {
    restore(0, s);
}

// Restoring a slot's state (feat/multi-sequence) - snapshot discipline as the docs above (one snapshot
// belongs to the state it was made against; the epoch check is that state's rewind history).
void Qwen4ExpSession::restore(int seq, const Qwen4ExpSnapshot &s) {
    const char *fn = "Qwen4ExpSession::restore";
    STRIX_CHECK(seq >= 0 && seq < n_slots(), fn, ": seq ", seq, ", expected 0..", n_slots() - 1);
    Qwen4ExpSeqState &st = *seqs_[(size_t)seq];
    STRIX_CHECK(st.pending3.empty(), "Qwen4ExpSession::restore: set_positions holds ", st.pending3.size() / 3,
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
    st.verify_pending = false;  // a restore replaces all state a pending verify could have left
    STRIX_CHECK(can_restore(seq, s), "Qwen4ExpSession::restore: the session rewound below the snapshot's position ", s.pos_,
                " since it was saved, so its KV cache no longer matches");
    for (size_t l = 0; l < st.conv_state.size(); ++l) {
        copy_state(s.conv_[l], st.conv_state[l], stream_, "conv state");
        copy_state(s.rec_[l], st.rec_state[l], stream_, "recurrent state");
        copy_state(s.tail_[l], st.tail[st.tail_cur][l], stream_, "indexer tail");
        copy_state(s.kv8_tail_[l], st.kv8_tail[st.tail_cur][l], stream_, "K/V tail");
    }
    copy_state(s.ple_state_, st.ple_state, stream_, "PLE conv state");
    copy_state(s.mtp_tail_, st.tail_mtp[st.mtp_tail_cur], stream_, "MTP indexer tail");
    copy_state(s.mtp_prev_, st.mtp_prev, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::restore");
    st.ple_hist = s.ple_hist_;
    st.pos = s.pos_;
    // mRoPE: the scalar continuation state; with a needs_tail resume also the last saved rows, so the first
    // forward's block window can rope its tail keys with them (rows below them never re-enter a window).
    st.rope_delta = s.rope_delta_;
    st.history_ever_interleaved = s.has_triples();
    st.pos3_host.clear();
    if (needs_tail) st.pos3_host = s.triples_;
    st.pos3_start = st.pos - (int64_t)st.pos3_host.size() / 3;
    STRIX_CHECK(st.pos3Hend() == st.pos, "Qwen4ExpSession::restore: history [", st.pos3_start, ", ", st.pos3Hend(),
                ") doesn't end at the restored position ", st.pos, " (internal)");
    broken_ = false, st.mtp_chain_step = -1;
    st.rewind_to.push_back(s.pos_);
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
    for (size_t l = 0; l < st().conv_state.size(); ++l) {
        if (D.is_attention[l])
            b += (kv_format_ == KvFormat::Q8 ? kv8_state_bytes(n, from) : kv) + keys + st().tail[0][l].size();
        else
            b += st().conv_state[l].size() + st().rec_state[l].size() * sizeof(float);
    }
    b += st().ple_state.size();
    if (mtp_) b += kv + keys + st().tail_mtp[0].size() + st().mtp_prev.size();
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
    // One copy per 8-dimension chunk, not a 2D copy: the cache is paged, and the pitch's gaps between the chunks' used
    // blocks are unmapped (a 2D copy over them is refused). The host bytes are the same, chunk after chunk.
    for (size_t c = 0; c < rows; ++c) {
        if (to_host)
            STRIX_HIP_CHECK(hipMemcpyAsync(cur + c * row, d + c * pitch, row, hipMemcpyDeviceToHost, stream),
                            "Qwen4ExpSession export: block keys, chunk ", c);
        else
            STRIX_HIP_CHECK(hipMemcpyAsync(d + c * pitch, cur + c * row, row, hipMemcpyHostToDevice, stream),
                            "Qwen4ExpSession import: block keys, chunk ", c);
    }
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
    STRIX_CHECK(kv_format_ == KvFormat::Q8, "Qwen4ExpSession::kv8_state_bytes: the session's K/V is ",
                kv_format_name(kv_format_), ", not q8");
    STRIX_CHECK(from >= 0 && from <= n && n <= kv8_cap_, "Qwen4ExpSession::kv8_state_bytes: rows [", from, ", ", n,
                ") outside the 8-bit capacity ", kv8_cap_);
    const Qwen4ExpDims &D = m_.dims();
    const int64_t W = D.hkv * D.hd, G = kernels::kKv8Group, ga = from / G, gb = n / G, rows = (gb - ga) * G;
    return (size_t)(rows * W) * 2 + (size_t)((gb - ga) * W) * 4 + (size_t)(rows * (W / kernels::kKv8Block)) * 4 +
           kernels::kv8_tail_bytes(W, m_.act());
}

void Qwen4ExpSession::kv8_copy(const Qwen4ExpSeqState &sq, uint8_t *&cur, size_t layer, int64_t n, int64_t from,
                               const DeviceBuffer<uint8_t> &tail, bool to_host,
                               std::vector<std::pair<uint8_t *, size_t>> *unused, bool paged) {
    const Qwen4ExpDims &D = m_.dims();
    const int64_t W = D.hkv * D.hd, G = kernels::kKv8Group, cap = kv8_cap_, ga = from / G, gb = n / G;
    STRIX_CHECK(layer < sq.kv8_store.size() && sq.kv8_store[layer].capacity() >= kernels::kv8_bytes(cap, W),
                "Qwen4ExpSession::kv8_copy: layer ", layer, " has no 8-bit K/V storage");
    STRIX_CHECK(tail.size() == kernels::kv8_tail_bytes(W, m_.act()), "Qwen4ExpSession::kv8_copy: tail of ",
                tail.size(), " bytes, expected ", kernels::kv8_tail_bytes(W, m_.act()));
    uint8_t *store = sq.kv8_store[layer].get();  // (its address; the store is touched only when paged)
    const size_t kc = (size_t)(cap * W), kp = (size_t)((cap / G) * W) * 4, nb = (size_t)(W / kernels::kKv8Block);
    if (paged) {  // not paged = the small half of a parked entry: those four regions ARE its parked pages
        host_copy(cur, store + (size_t)(ga * G * W), (size_t)((gb - ga) * G * W), to_host, stream_, "8-bit K codes");
        host_copy(cur, store + kc + (size_t)(ga * W) * 4, (size_t)((gb - ga) * W) * 4, to_host, stream_,
                  "8-bit K params");
        host_copy(cur, store + kc + kp + (size_t)(ga * G * W), (size_t)((gb - ga) * G * W), to_host, stream_,
                  "8-bit V codes");
        host_copy(cur, store + 2 * kc + kp + (size_t)(ga * G) * nb * 4, (size_t)((gb - ga) * G) * nb * 4, to_host,
                  stream_, "8-bit V params");
    }
    // The tail: rows [gb * G, n) valid in each half (K, V).
    const size_t es_ = es(m_.act()), half = (size_t)(G * W) * es_, valid = (size_t)((n - gb * G) * W) * es_;
    if (unused)
        for (int h = 0; h < 2; ++h) unused->emplace_back(cur + h * half + valid, half - valid);
    host_copy(cur, tail.get(), tail.size(), to_host, stream_, "K/V tail");
}

void Qwen4ExpSession::export_state(const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from) {
    export_state(0, s, out, bytes, from);
}

// The whole-entry pack's paged ranges: exactly the pieces export_state copies straight out of a paged array -
// the K / V rows, the four 8-bit store regions, and the block keys' per-chunk rows (chunk-major: the chunks are
// contiguous on the host but strided in the array, so each is its own range). This arithmetic is the twin of
// that walk; forward_qwen4exp_paged_park_stitch is what keeps the two honest. Whole entries only (a delta's
// rows are a window, not whole arrays).
std::vector<Qwen4ExpSession::StatePagedRange> Qwen4ExpSession::state_paged_ranges(int64_t n) const {
    STRIX_CHECK(n >= 1 && n <= capacity_, "Qwen4ExpSession::state_paged_ranges: n = ", n, ", expected 1..",
                capacity_);
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    const size_t row = (size_t)(D.hkv * D.hd) * e;
    const int64_t W = D.hkv * D.hd, G = kernels::kKv8Group, gb = n / G, rows8 = (size_t)(gb * G);
    const size_t kc = (size_t)(kv8_cap_ * W), kp = (size_t)((kv8_cap_ / G) * W) * 4,
                 nb = (size_t)(W / kernels::kKv8Block);
    const int64_t b1 = n / kernels::kQsaBlock;
    const size_t keys_row = (size_t)(b1 * 8) * e, pitch = (size_t)(cap_blocks_ * 8) * e;
    const int64_t key_chunks = D.idx_d / 8;
    const bool q8 = kv_format_ == KvFormat::Q8;
    std::vector<StatePagedRange> r;
    size_t off = sizeof(Qwen4ExpStateHeader) + sizeof(PleHistory);
    for (size_t l = 0; l < st().conv_state.size(); ++l) {
        if (!D.is_attention[l]) {  // GDN: conv + recurrent, all small
            off += st().conv_state[l].size() + st().rec_state[l].size() * sizeof(float);
            continue;
        }
        if (q8) {
            const int64_t s8 = kv_spec_index(kv_layout_, "8-bit K/V layer " + std::to_string(l));
            r.push_back({s8, 0, off, (size_t)(rows8 * W)});                                // K codes
            r.push_back({s8, kc, off + (size_t)(rows8 * W), (size_t)(gb * W) * 4});        // K params
            r.push_back({s8, kc + kp, off + (size_t)(rows8 * W) + (size_t)(gb * W) * 4,    // V codes
                         (size_t)(rows8 * W)});
            r.push_back({s8, 2 * kc + kp,                                                 // V params
                         off + 2 * (size_t)(rows8 * W) + (size_t)(gb * W) * 4, (size_t)(gb * G) * nb * 4});
            off += kv8_state_bytes(n, 0);  // the four regions plus the K/V tail (small)
        } else {
            r.push_back({kv_spec_index(kv_layout_, "K cache layer " + std::to_string(l)), 0, off,
                         (size_t)n * row});
            r.push_back({kv_spec_index(kv_layout_, "V cache layer " + std::to_string(l)), 0,
                         off + (size_t)n * row, (size_t)n * row});
            off += 2 * (size_t)n * row;
        }
        const int64_t sk = kv_spec_index(kv_layout_, "block keys layer " + std::to_string(l));
        for (int64_t c = 0; c < key_chunks; ++c)  // one range per 8-dimension chunk
            r.push_back({sk, (size_t)(c * pitch), off + (size_t)(c * keys_row), keys_row});
        off += (size_t)(b1 * D.idx_d) * e + st().tail[0][l].size();
    }
    off += st().ple_state.size();
    if (mtp_) {
        r.push_back({kv_spec_index(kv_layout_, "MTP K cache"), 0, off, (size_t)n * row});
        r.push_back({kv_spec_index(kv_layout_, "MTP V cache"), 0, off + (size_t)n * row, (size_t)n * row});
        off += 2 * (size_t)n * row;
        const int64_t sk = kv_spec_index(kv_layout_, "MTP block keys");
        for (int64_t c = 0; c < key_chunks; ++c)
            r.push_back({sk, (size_t)(c * pitch), off + (size_t)(c * keys_row), keys_row});
        off += (size_t)(b1 * D.idx_d) * e;
    }
    return r;
}

size_t Qwen4ExpSession::small_state_bytes(int64_t n) const {
    size_t paged = 0;
    for (const StatePagedRange &r : state_paged_ranges(n)) paged += r.bytes;
    const size_t whole = state_bytes(n, 0);
    STRIX_CHECK(whole >= paged, "Qwen4ExpSession::small_state_bytes: the paged ranges hold ", paged,
                " bytes, more than the whole state's ", whole, " - the split drifted from the walk");
    return whole - paged;
}

const uint8_t *Qwen4ExpSession::parked_host(const ParkedKv &parked, int64_t spec, size_t array_off,
                                            size_t bytes) const {
    STRIX_CHECK(parked.reservation != nullptr, "Qwen4ExpSession::parked_host: empty parked K/V record");
    STRIX_CHECK(parked.reservation == kv_res_.get(), "Qwen4ExpSession::parked_host: this parked K/V belongs to "
                "another session's reservation - only its owner can read it back");
    STRIX_CHECK(spec >= 0 && spec < (int64_t)kv_layout_.specs.size(), "Qwen4ExpSession::parked_host: spec ",
                spec, ", expected 0..", kv_layout_.specs.size() - 1);
    const std::string &name = kv_layout_.specs[(size_t)spec].name;
    const ParkedArrayKv *arr = nullptr;
    for (const ParkedArrayKv &a : parked.arrays)
        if (a.name == name) { arr = &a; break; }
    STRIX_CHECK(arr != nullptr, "Qwen4ExpSession::parked_host: the record has no array '", name, "'");
    STRIX_CHECK(array_off <= kv_layout_.specs[(size_t)spec].bytes &&
                    bytes <= kv_layout_.specs[(size_t)spec].bytes - array_off,
                "Qwen4ExpSession::parked_host: [", array_off, ", +", bytes, " B) of '", name, "' is past its ",
                kv_layout_.specs[(size_t)spec].bytes, " bytes");
    const size_t page = kv_res_->page_bytes();
    for (size_t p = array_off / page; p * page < array_off + bytes &&
                                       p * page < kv_layout_.specs[(size_t)spec].bytes; ++p) {
        bool held = false;
        for (const PageHandle &ph : arr->pages)
            if (ph.page == p) { held = true; break; }
        STRIX_CHECK(held, "Qwen4ExpSession::parked_host: '", name, "' page ", p, " (byte ", p * page,
                    " of the range [", array_off, ", +", bytes, " B)) is not in the parked record - the range "
                    "reaches past what the conversation wrote");
    }
    return static_cast<const uint8_t *>(parked.reservation->base()) +
           kv_layout_.parked_gen_offset(parked.parked_gen) + kv_array_within(kv_layout_, name) + array_off;
}

void Qwen4ExpSession::export_state(int seq, const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::export_state: seq ", seq, ", expected 0..", n_slots() - 1);
    const Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(out != nullptr, "Qwen4ExpSession::export_state: null output");
    STRIX_CHECK(can_restore(seq, s), "Qwen4ExpSession::export_state: the snapshot (position ", s.pos(),
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
    hdr.has_image = sq.history_ever_interleaved ? 1 : 0;  // the snapshot's own flag would miss an after-image
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
    for (size_t l = 0; l < sq.conv_state.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            if (kv_format_ == KvFormat::Q8) {
                kv8_copy(sq, cur, l, n, from, s.kv8_tail_[l], true, &tail_unused);
            } else {
                host_copy(cur, sq.k_cache[l].get() + (size_t)from * row, kv, true, stream_, "K cache");
                host_copy(cur, sq.v_cache[l].get() + (size_t)from * row, kv, true, stream_, "V cache");
            }
            block_keys_copy(cur, sq.block_keys[l].get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
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
        host_copy(cur, sq.k_cache_mtp.get() + (size_t)from * row, kv, true, stream_, "MTP K cache");
        host_copy(cur, sq.v_cache_mtp.get() + (size_t)from * row, kv, true, stream_, "MTP V cache");
        block_keys_copy(cur, sq.block_keys_mtp.get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
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

// The small half of a whole state: export_state's walk with the paged pieces elided (the K / V rows, the four
// 8-bit store regions and the block keys ARE the parked pages; kv8_copy's `paged` flag keeps that one
// definition). forward_qwen4exp_paged_park_stitch stitches this + state_paged_ranges back into export_state's
// bytes, so a paged entry's on-disk file is the same file a HostBuffer entry writes.
PagedState Qwen4ExpSession::paged_state(int64_t parked_id, int64_t n) const {
    const ParkedKv &record = parked(parked_id);  // the session's own live record, named by the id
    PagedState out;
    out.parked_id = parked_id;
    out.page_bytes = record.bytes;
    out.record_name = record.tally_category;
    out.state_bytes = state_bytes(n, 0);
    for (const StatePagedRange &r : state_paged_ranges(n)) {
        // A range with nothing in it (the 8-bit regions below one full group, n < 32) holds no byte of the state:
        // the writer's walk wants only ranges that cover something.
        if (r.bytes == 0) continue;
        // The writer walks the chunks in order and trusts them not to overlap, so they are checked here once,
        // where they are made, rather than at every gather.
        STRIX_CHECK(out.chunks.empty() || r.pack_off >= out.chunks.back().state_off + out.chunks.back().bytes,
                    "Qwen4ExpSession::paged_state: range at ", r.pack_off, " overlaps or precedes the previous "
                    "one (the ranges must tile the state in order)");
        STRIX_CHECK(r.pack_off + r.bytes <= out.state_bytes, "Qwen4ExpSession::paged_state: range [", r.pack_off,
                    ", +", r.bytes, " B) is past the state's ", out.state_bytes, " bytes");
        out.chunks.push_back({r.pack_off, r.bytes, parked_host(record, r.spec, r.array_off, r.bytes)});
    }
    size_t paged = 0;
    for (const PagedChunk &c : out.chunks) paged += c.bytes;
    STRIX_CHECK(!out.chunks.empty(), "Qwen4ExpSession::paged_state: the state at ", n, " has no paged ranges");
    STRIX_CHECK(paged + small_state_bytes(n) == out.state_bytes, "Qwen4ExpSession::paged_state: ", paged,
                " paged + ", small_state_bytes(n), " small bytes, the state is ", out.state_bytes);
    return out;
}

void Qwen4ExpSession::export_small_state(int seq, const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::export_small_state: seq ", seq, ", expected 0..",
                n_slots() - 1);
    const Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(out != nullptr, "Qwen4ExpSession::export_small_state: null output");
    STRIX_CHECK(can_restore(seq, s), "Qwen4ExpSession::export_small_state: the snapshot (position ", s.pos(),
                ") is not valid for this session's KV cache any more");
    const int64_t n = s.pos();
    STRIX_CHECK(bytes == small_state_bytes(n), "Qwen4ExpSession::export_small_state: buffer ", bytes,
                " bytes, the small half of the state at ", n, " is ", small_state_bytes(n));
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = out;
    Qwen4ExpStateHeader hdr;
    hdr.rope_delta = s.rope_delta_;
    hdr.cont_t = s.first_cont_[0], hdr.cont_h = s.first_cont_[1], hdr.cont_w = s.first_cont_[2];
    hdr.has_image = sq.history_ever_interleaved ? 1 : 0;  // as in export_state: the snapshot's own flag would miss
                                                          // an after-image resume
    hdr.ntriples = s.has_triples() ? s.ntriples_ : 0;
    for (int64_t k = 0; k < hdr.ntriples * 3; ++k) hdr.triples[k] = s.triples_[(size_t)k];
    std::memcpy(cur, &hdr, sizeof(hdr));
    cur += sizeof(hdr);
    std::memcpy(cur, &s.ple_hist_, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    const size_t tail_valid = (size_t)((n % kernels::kQsaBlock) * D.idx_d) * e;
    std::vector<std::pair<uint8_t *, size_t>> tail_unused;
    // A drift between state_paged_ranges and this walk must name itself, never run off the buffer (the stitch
    // test is what catches it in practice; these checks are the seat belt).
    const auto room = [&](size_t b) {
        STRIX_CHECK((size_t)(cur - out) + b <= bytes, "Qwen4ExpSession::export_small_state: the small walk needs ",
                    (size_t)(cur - out) + b, " bytes, the buffer holds ", bytes,
                    " - the paged ranges and this walk drifted apart");
    };
    for (size_t l = 0; l < sq.conv_state.size(); ++l) {
        if (D.is_attention[l]) {
            if (kv_format_ == KvFormat::Q8) {  // the four store regions elided, the K/V tail kept
                room(kernels::kv8_tail_bytes(D.hkv * D.hd, m_.act()));
                kv8_copy(sq, cur, l, n, /*from=*/0, s.kv8_tail_[l], true, &tail_unused, /*paged=*/false);
            }
            // (the K / V rows and the block keys are paged: nothing to write, nothing to advance)
            STRIX_CHECK(s.tail_[l].size() >= tail_valid, "Qwen4ExpSession::export_small_state: indexer tail of ",
                        s.tail_[l].size(), " bytes, ", tail_valid, " valid");
            tail_unused.emplace_back(cur + tail_valid, s.tail_[l].size() - tail_valid);
            room(s.tail_[l].size());
            host_copy(cur, s.tail_[l].get(), s.tail_[l].size(), true, stream_, "indexer tail");
        } else {
            room(s.conv_[l].size() + s.rec_[l].size() * sizeof(float));
            host_copy(cur, s.conv_[l].get(), s.conv_[l].size(), true, stream_, "conv state");
            host_copy(cur, s.rec_[l].get(), s.rec_[l].size() * sizeof(float), true, stream_, "recurrent state");
        }
    }
    room(s.ple_state_.size());
    host_copy(cur, s.ple_state_.get(), s.ple_state_.size(), true, stream_, "PLE conv state");
    if (mtp_) {
        room(s.mtp_tail_.size() + s.mtp_prev_.size());
        STRIX_CHECK(s.mtp_tail_.size() >= tail_valid, "Qwen4ExpSession::export_small_state: MTP indexer tail of ",
                    s.mtp_tail_.size(), " bytes, ", tail_valid, " valid");
        tail_unused.emplace_back(cur + tail_valid, s.mtp_tail_.size() - tail_valid);
        host_copy(cur, s.mtp_tail_.get(), s.mtp_tail_.size(), true, stream_, "MTP indexer tail");
        host_copy(cur, s.mtp_prev_.get(), s.mtp_prev_.size(), true, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::export_small_state");
    for (const auto &[at, len] : tail_unused) std::memset(at, 0, len);  // as in export_state: the state's bytes
                                                                        // depend only on its sequence
    STRIX_CHECK((size_t)(cur - out) == bytes, "Qwen4ExpSession::export_small_state: wrote ", cur - out, " of ",
                bytes, " bytes");
}

void Qwen4ExpSession::import_state(const uint8_t *in, size_t bytes, int64_t n, int64_t from) {
    import_state(0, in, bytes, n, from);
}

// The small half written back (the resume side of the same split): import_state's checks and tail bookkeeping
// with the paged pieces elided - resume_parked mapped the parked pages into this slot's generation first, and the
// pages ARE the rows.
void Qwen4ExpSession::import_small_state(int seq, const uint8_t *in, size_t bytes, int64_t n) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::import_small_state: seq ", seq, ", expected 0..",
                n_slots() - 1);
    STRIX_CHECK(!broken_, "Qwen4ExpSession::import_small_state: seq ", seq, ": an earlier call failed midway - "
                "reset() first");
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(in != nullptr, "Qwen4ExpSession::import_small_state: null input");
    STRIX_CHECK(n >= 1 && n < capacity_, "Qwen4ExpSession::import_small_state: n = ", n, ", expected 1..",
                capacity_ - 1);
    STRIX_CHECK(bytes >= sizeof(Qwen4ExpStateHeader), "Qwen4ExpSession::import_small_state: ", bytes,
                " bytes, shorter than a state header (", sizeof(Qwen4ExpStateHeader), ")");
    Qwen4ExpStateHeader hdr;
    std::memcpy(&hdr, in, sizeof(hdr));
    const int64_t max_pos = m_.dims().max_positions();
    STRIX_CHECK(hdr.magic == Qwen4ExpStateHeader::kMagic && hdr.version == Qwen4ExpStateHeader::kVersion,
                "Qwen4ExpSession::import_small_state: blob has no mRoPE state header (magic 0x",
                std::hex, hdr.magic, " version ", std::dec, hdr.version, ")");
    STRIX_CHECK(bytes == small_state_bytes(n), "Qwen4ExpSession::import_small_state: ", bytes,
                " bytes, the small half of the state at ", n, " is ", small_state_bytes(n),
                " (from another model, layout or activation dtype?)");
    STRIX_CHECK(hdr.rope_delta <= 0 && hdr.rope_delta >= -max_pos, "Qwen4ExpSession::import_small_state: header "
                "rope_delta = ", hdr.rope_delta, ", expected ", -max_pos, "..0");
    const int64_t cont[3] = {hdr.cont_t, hdr.cont_h, hdr.cont_w};
    for (int k = 0; k < 3; ++k)
        STRIX_CHECK(cont[k] >= 0 && cont[k] <= max_pos, "Qwen4ExpSession::import_small_state: header continuation ",
                    (k == 0 ? "t" : k == 1 ? "h" : "w"), " = ", cont[k], ", expected 0..", max_pos);
    STRIX_CHECK(hdr.has_image == 0 || hdr.has_image == 1, "Qwen4ExpSession::import_small_state: header "
                "has_image = ", hdr.has_image, ", expected 0 or 1");
    STRIX_CHECK(hdr.ntriples >= 0 && hdr.ntriples <= 3 && (hdr.ntriples > 0 ? hdr.has_image == 1 : true),
                "Qwen4ExpSession::import_small_state: header carries ", hdr.ntriples, " triples with "
                "has_image = ", hdr.has_image);
    const bool cont_uniform = cont[0] == cont[1] && cont[1] == cont[2];
    STRIX_CHECK(cont_uniform && cont[0] == n + hdr.rope_delta,
                "Qwen4ExpSession::import_small_state: row ", n, " continues at ", triple_str(cont),
                ", not uniformly at n + rope_delta = ", n + hdr.rope_delta);
    const bool carry_tail = hdr.has_image == 1 && hdr.ntriples > 0 && !triple_uniform(hdr.triples + (hdr.ntriples - 1) * 3);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_small_state");
    STRIX_CHECK(sq.verify_pending == false, "Qwen4ExpSession::import_small_state: seq ", seq,
                " has a verify pending - resolve it first");
    const Qwen4ExpDims &D = m_.dims();
    uint8_t *cur = const_cast<uint8_t *>(in);
    cur += sizeof(Qwen4ExpStateHeader);  // parsed and validated above, straight off `in`
    PleHistory hist;
    std::memcpy(&hist, cur, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    const auto room = [&](size_t b) {  // as in export_small_state
        STRIX_CHECK((size_t)(cur - in) + b <= bytes, "Qwen4ExpSession::import_small_state: the small walk needs ",
                    (size_t)(cur - in) + b, " bytes, the blob holds ", bytes,
                    " - the paged ranges and this walk drifted apart");
    };
    for (size_t l = 0; l < sq.conv_state.size(); ++l) {
        if (D.is_attention[l]) {
            if (kv_format_ == KvFormat::Q8) {
                room(kernels::kv8_tail_bytes(D.hkv * D.hd, m_.act()));
                kv8_copy(sq, cur, l, n, /*from=*/0, sq.kv8_tail[sq.tail_cur][l], false, nullptr, /*paged=*/false);
            }
            room(sq.tail[sq.tail_cur][l].size());
            host_copy(cur, sq.tail[sq.tail_cur][l].get(), sq.tail[sq.tail_cur][l].size(), false, stream_,
                      "indexer tail");
        } else {
            room(sq.conv_state[l].size() + sq.rec_state[l].size() * sizeof(float));
            host_copy(cur, sq.conv_state[l].get(), sq.conv_state[l].size(), false, stream_, "conv state");
            host_copy(cur, sq.rec_state[l].get(), sq.rec_state[l].size() * sizeof(float), false, stream_,
                      "recurrent state");
        }
    }
    room(sq.ple_state.size());
    host_copy(cur, sq.ple_state.get(), sq.ple_state.size(), false, stream_, "PLE conv state");
    if (mtp_) {
        room(sq.tail_mtp[sq.mtp_tail_cur].size() + sq.mtp_prev.size());
        host_copy(cur, sq.tail_mtp[sq.mtp_tail_cur].get(), sq.tail_mtp[sq.mtp_tail_cur].size(), false, stream_,
                  "MTP indexer tail");
        host_copy(cur, sq.mtp_prev.get(), sq.mtp_prev.size(), false, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_small_state");
    sq.ple_hist = hist;
    sq.rope_delta = hdr.rope_delta;
    sq.history_ever_interleaved = hdr.has_image == 1;
    sq.pos3_host.clear();
    if (carry_tail) sq.pos3_host.assign(hdr.triples, hdr.triples + (size_t)(hdr.ntriples * 3));
    sq.pos3_start = n - (int64_t)sq.pos3_host.size() / 3;
    STRIX_CHECK((size_t)(cur - in) == bytes, "Qwen4ExpSession::import_small_state: read ", cur - in, " of ",
                bytes, " bytes");
    // broken_ is NOT cleared here (import_state's whole-state import does): the small half lands on one slot over
    // pages resume_parked mapped, and a session flag set by another slot's failure is that slot's to clear.
    sq.pos = n, sq.mtp_chain_step = -1, sq.verify_pending = false;
    sq.imported_at = n;
    sq.rewind_to.push_back(0);  // as in import_state: no older snapshot matches the resumed rows
}

void Qwen4ExpSession::import_state(int seq, const uint8_t *in, size_t bytes, int64_t n, int64_t from) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::import_state: seq ", seq, ", expected 0..", n_slots() - 1);
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(in != nullptr, "Qwen4ExpSession::import_state: null input");
    STRIX_CHECK(n >= 1 && n < capacity_, "Qwen4ExpSession::import_state: n = ", n, ", expected 1..", capacity_ - 1);
    STRIX_CHECK(from >= 0 && from < n, "Qwen4ExpSession::import_state: from = ", from, ", expected 0..", n - 1);
    STRIX_CHECK(bytes >= sizeof(Qwen4ExpStateHeader), "Qwen4ExpSession::import_state: ", bytes,
                " bytes, shorter than a state header (", sizeof(Qwen4ExpStateHeader), ")");
    // A delta lands on the base it was cut from: the rows below `from` must be that base's, imported just before.
    STRIX_CHECK(from == 0 || (sq.pos == from && sq.imported_at == from), "Qwen4ExpSession::import_state: a delta from ",
                from, " needs its base imported just before (position ", sq.pos, ", last import ended at ", sq.imported_at,
                ")");
    STRIX_CHECK(sq.pending3.empty(), "Qwen4ExpSession::import_state: set_positions holds ", sq.pending3.size() / 3,
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
    // Paged K/V: maps what rows [0, n) need; a deeper previous conversation's pages stay mapped (never released).
    ensure_rows(sq, n);
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = const_cast<uint8_t *>(in);
    cur += sizeof(Qwen4ExpStateHeader);  // parsed and validated above, straight off `in`
    PleHistory hist;
    std::memcpy(&hist, cur, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    for (size_t l = 0; l < sq.conv_state.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            if (kv_format_ == KvFormat::Q8) {
                kv8_copy(sq, cur, l, n, from, sq.kv8_tail[sq.tail_cur][l], false, nullptr);
            } else {
                host_copy(cur, sq.k_cache[l].get() + (size_t)from * row, kv, false, stream_, "K cache");
                host_copy(cur, sq.v_cache[l].get() + (size_t)from * row, kv, false, stream_, "V cache");
            }
            block_keys_copy(cur, sq.block_keys[l].get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
            host_copy(cur, sq.tail[sq.tail_cur][l].get(), sq.tail[sq.tail_cur][l].size(), false, stream_, "indexer tail");
        } else {
            host_copy(cur, sq.conv_state[l].get(), sq.conv_state[l].size(), false, stream_, "conv state");
            host_copy(cur, sq.rec_state[l].get(), sq.rec_state[l].size() * sizeof(float), false, stream_, "recurrent state");
        }
    }
    host_copy(cur, sq.ple_state.get(), sq.ple_state.size(), false, stream_, "PLE conv state");
    if (mtp_) {
        const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
        host_copy(cur, sq.k_cache_mtp.get() + (size_t)from * row, kv, false, stream_, "MTP K cache");
        host_copy(cur, sq.v_cache_mtp.get() + (size_t)from * row, kv, false, stream_, "MTP V cache");
        block_keys_copy(cur, sq.block_keys_mtp.get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
        host_copy(cur, sq.tail_mtp[sq.mtp_tail_cur].get(), sq.tail_mtp[sq.mtp_tail_cur].size(), false, stream_,
                  "MTP indexer tail");
        host_copy(cur, sq.mtp_prev.get(), sq.mtp_prev.size(), false, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_state");
    sq.ple_hist = hist;
    sq.rope_delta = hdr.rope_delta;
    sq.history_ever_interleaved = hdr.has_image == 1;
    sq.pos3_host.clear();
    if (carry_tail) sq.pos3_host.assign(hdr.triples, hdr.triples + (size_t)(hdr.ntriples * 3));
    sq.pos3_start = n - (int64_t)sq.pos3_host.size() / 3;
    STRIX_CHECK((size_t)(cur - in) == bytes, "Qwen4ExpSession::import_state: read ", cur - in, " of ", bytes, " bytes");
    sq.pos = n, broken_ = false, sq.mtp_chain_step = -1, sq.verify_pending = false;
    sq.imported_at = n;
    sq.rewind_to.push_back(0);  // the KV cache below n was rewritten (or completed): no older snapshot matches it
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
    if (kv_format_ == KvFormat::Q8) d2h(words + 9, kv8_err_.get(), 12, "8-bit K/V error");
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

void Qwen4ExpSession::want_candidates(int64_t n_valid, const uint32_t *masks, int64_t mask_rows, int64_t mask_words,
                                      const kernels::LogitBiasEntry *bias, int64_t bias_n) {
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
    int64_t rows_end = 0;
    if (bias == nullptr) {
        STRIX_CHECK(bias_n == 0, fn, ": bias_n = ", bias_n, " without bias entries (pass 0)");
    } else {
        STRIX_CHECK(bias_n >= 1 && bias_n <= kMaxLogits * vocab, fn, ": bias_n = ", bias_n, ", expected 1..",
                    kMaxLogits * vocab);
        // The order the kernel's race-freedom rests on (no two entries on one logit), checked in one pass.
        for (int64_t i = 0; i < bias_n; ++i) {
            const kernels::LogitBiasEntry &e = bias[i];
            STRIX_CHECK(e.row0 >= 0 && e.rows >= 1 && (int64_t)e.row0 + e.rows <= kMaxLogits && e.id >= 0 &&
                            e.id < vocab && std::isfinite(e.v),
                        fn, ": logit bias entry ", i, " (rows ", e.row0, "+", e.rows, ", id ", e.id, ", bias ", e.v,
                        "), expected rows inside 0..", kMaxLogits, ", id 0..", vocab - 1, ", a finite bias");
            if (i > 0) {
                const kernels::LogitBiasEntry &p = bias[i - 1];
                const bool same_run = e.row0 == p.row0 && e.rows == p.rows;
                STRIX_CHECK(same_run ? e.id > p.id : e.row0 >= p.row0 + p.rows, fn, ": logit bias entry ", i, " (rows ",
                            e.row0, "+", e.rows, ", id ", e.id, ") after (rows ", p.row0, "+", p.rows, ", id ", p.id,
                            "): expected a higher id in the same run, or a run starting past the last one's rows");
            }
            rows_end = std::max<int64_t>(rows_end, (int64_t)e.row0 + e.rows);
        }
        const size_t bytes = (size_t)bias_n * sizeof(kernels::LogitBiasEntry);
        if (bias_host_.size() < bytes) {
            STRIX_CHECK(!bias_upload_recorded_ || hipEventSynchronize(bias_uploaded_) == hipSuccess, fn,
                        ": waiting for the last logit bias upload before replacing its buffer");
            bias_upload_recorded_ = false;
            const size_t capacity = std::max(bytes, 2 * bias_host_.size());
            bias_host_ = PinnedHostBuffer(capacity, "logit bias (host)");
            bias_dev_ = DeviceBuffer<kernels::LogitBiasEntry>(capacity / sizeof(kernels::LogitBiasEntry), "logit bias");
            if (bias_uploaded_ == nullptr)
                STRIX_HIP_CHECK(hipEventCreateWithFlags(&bias_uploaded_, hipEventDisableTiming), fn,
                                ": logit bias upload event");
        }
        if (bias_upload_recorded_) STRIX_HIP_CHECK(hipEventSynchronize(bias_uploaded_), fn, ": last logit bias upload");
        std::memcpy(bias_host_.get(), bias, bytes);
    }
    mask_rows_ = masks == nullptr ? 0 : mask_rows;
    mask_words_ = masks == nullptr ? 0 : mask_words;
    bias_n_ = bias == nullptr ? 0 : bias_n;
    bias_rows_end_ = rows_end;
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
    slog(LogLevel::Debug, "vision tower: grid %lldx%lld, %lld patches, %lld merged rows, %.2f s", (long long)image.grid_h,
         (long long)image.grid_w, (long long)n, (long long)rows.size(), (double)elapsed_ns * 1e-9);
}

std::vector<float> Qwen4ExpSession::forward(const std::vector<int32_t> &ids, int64_t n_logits,
                                            const ImageInput *image, const Qwen4ExpProbe &probe) {
    std::vector<BatchRun> batch{{0, ids, n_logits}};
    return forward_batch(batch, image, probe);
}

// The batched forward (feat/multi-sequence design): one code path over K row-groups, the runs' rows laid
// into the shared workspaces in batch order (the returned logits rows follow it). The stateless sublayers (embedding, HC mixes, dense linears,
// router, MoE) launch once over all N rows - the weights read once for every run, the batch's gain; the
// state-touching sublayers (PLE gate/conv, GDN front + delta rule, the KV writes, the indexer keys / scores /
// attention, the MTP head's K/V catch-up) launch per run against that state's pointers at the run's row
// offset, with unchanged kernels. A batch of one is exactly the old forward(): same kernels, same arguments,
// same launch order - verified by token identity. Multi-run batches are decode-sized and scalar: no image
// rows, no mRoPE state (checked below); the verify mechanism is per state (forward_batch_verify).
std::vector<float> Qwen4ExpSession::forward_batch(std::vector<BatchRun> &batch, const ImageInput *image,
                                                  const Qwen4ExpProbe &probe) {
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward: an earlier call failed midway; reset() the session first");
    STRIX_CHECK(!batch.empty() && batch.size() <= (size_t)n_slots(), "Qwen4ExpSession::forward: ", batch.size(),
                " runs, expected 1..", n_slots());
    // A verify awaiting resolution anywhere blocks the session: its replay state (verify_in / ple_hist_buf_ /
    // the projection rows keep_verify_prefix needs) lives in the shared workspaces a forward overwrites.
    for (size_t k = 0; k < seqs_.size(); ++k)
        STRIX_CHECK(!seqs_[k]->verify_pending, "Qwen4ExpSession::forward: a forward_verify on seq ", k,
                    " awaits keep_verify / drop_verify - resolve every run of the pending batch before the next forward");
    for (size_t k = 0; k < batch.size(); ++k) {
        STRIX_CHECK(batch[k].seq >= 0 && batch[k].seq < n_slots(), "Qwen4ExpSession::forward: run ", k, " on seq ",
                    batch[k].seq, ", expected 0..", n_slots() - 1);
        for (size_t j = 0; j < k; ++j)
            STRIX_CHECK(batch[j].seq != batch[k].seq, "Qwen4ExpSession::forward: seq ", batch[k].seq,
                        " carries two runs (", j, " and ", k, ") - one state runs one row group per forward");
        // keep_verify(seq, .., carry)'s owed token: a run of a state that carries one must start with it (the
        // MTP cycle's rejection sample - the run's rows continue right after the kept prefix with it at row 0).
        const int32_t carry = seqs_[(size_t)batch[k].seq]->verify_carry;
        if (carry != 0)
            STRIX_CHECK(!batch[k].ids.empty() && batch[k].ids[0] == carry, "Qwen4ExpSession::forward: seq ",
                        batch[k].seq, " carries token ", carry, " from keep_verify - its run ", k, " of the next "
                        "forward must start with it (its ids[0] is ",
                        batch[k].ids.empty() ? std::string("nothing)") : std::to_string(batch[k].ids[0]) + ")");
    }
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    STRIX_TRACE_STAGE(stage);
    STRIX_TRACE_SET(stage, "embed");
    // Dense Q4 / Q8 projection: on the matrix units for large enough forwards when switched on.
    auto lin = [&](const void *x, const QWeightView &w, void *y, int64_t M, Act out_act) {
        if (wmma && M >= kWmmaMinTokens) kernels::linear_qw_wmma(x, w, y, M, act, out_act, stream_);
        else kernels::linear_qw(x, w, y, M, act, out_act, stream_);
    };
    const kernels::MoeMath moe_math = wmma ? kernels::MoeMath::WmmaBf16 : kernels::MoeMath::F32;
    // Expert slots per token: top-k + the shared expert's when it's stacked.
    const int64_t n4 = D.H * D.d, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    struct Run {
        int seq;
        Qwen4ExpSeqState *sq;
        int64_t base, Tr, nL, p;  // row base in the batch buffers, rows, logits rows, the run's start position
        bool fresh;               // the run starts at position 0 (the GDN states' reset marker)
        int64_t rope0;            // the scalar-path rope base (the run's pos + its rope_delta)
        std::future<void> ple_ready;  // multi-run: this run's PLE gather (joined at the PLE layer)
    };
    std::vector<Run> runs;
    runs.reserve(batch.size());
    std::vector<int32_t> all_ids;  // the runs' ids concatenated in batch order
    int64_t nL_total = 0;
    for (auto &br : batch) {
        Qwen4ExpSeqState &sq = *seqs_[(size_t)br.seq];
        const int64_t Tr = (int64_t)br.ids.size();
        STRIX_CHECK(Tr >= 1 && (int64_t)all_ids.size() + Tr <= max_tokens_, "Qwen4ExpSession::forward: ", Tr,
                    " tokens in a run of a forward of at most ", max_tokens_, " rows");
        STRIX_CHECK(sq.pos + Tr <= capacity_, "Qwen4ExpSession::forward: positions [", sq.pos, ", ", sq.pos + Tr,
                    ") exceed the capacity ", capacity_);
        STRIX_CHECK(br.n_logits >= 0 && br.n_logits <= std::min(Tr, kMaxLogits),
                    "Qwen4ExpSession::forward: n_logits = ", br.n_logits, ", expected 0..", std::min(Tr, kMaxLogits));
        runs.push_back({br.seq, &sq, (int64_t)all_ids.size(), Tr, br.n_logits, sq.pos, sq.pos == 0, 0, {}});
        all_ids.insert(all_ids.end(), br.ids.begin(), br.ids.end());
        nL_total += br.n_logits;
    }
    const int64_t T = (int64_t)all_ids.size();  // the stateless sublayers' row count = every run's rows
    STRIX_CHECK(nL_total <= kMaxLogits, "Qwen4ExpSession::forward: ", nL_total, " logits rows across ", runs.size(),
                " runs, expected <=", kMaxLogits);
    if (runs.size() > 1) {  // the batch's decode-size row contract (the header's forward_batch: the batch-wide
                            // path choice stays a decode-sized forward, and the kMaxLogits-row logits
                            // workspace (lg_src_) and the saved verify rows cover it)
        STRIX_CHECK(T <= kMaxLogits, "Qwen4ExpSession::forward: ", T, " rows across ", runs.size(),
                    " runs - a multi-run forward is decode-sized, expected <=", kMaxLogits, " total");
        if (D.ple_layer >= 0)  // the PLE history windows are laid end to end in ple_hist_buf_ [S + rows]
            STRIX_CHECK((int64_t)runs.size() * ((D.ple_taps - 1) * D.ple_dil) + T <=
                            (D.ple_taps - 1) * D.ple_dil + max_tokens_,
                        "Qwen4ExpSession::forward: ", runs.size(), " runs x the PLE state window + ", T,
                        " rows exceed the history buffer's ", (D.ple_taps - 1) * D.ple_dil + max_tokens_);
    }
    const int64_t cand_n_valid = cand_n_valid_;  // want_candidates() covers this one forward, whatever happens
    const int64_t mask_rows = mask_rows_, mask_words = mask_words_;
    const int64_t bias_n = bias_n_, bias_rows_end = bias_rows_end_;
    cand_n_valid_ = 0, mask_rows_ = 0, mask_words_ = 0, bias_n_ = 0, bias_rows_end_ = 0;
    STRIX_CHECK(mask_rows == 0 || (cand_n_valid > 0 && mask_rows == nL_total), "Qwen4ExpSession::forward: ",
                mask_rows, " structured-output mask rows for a forward of ", nL_total, " logits rows");
    STRIX_CHECK(bias_n == 0 || (cand_n_valid > 0 && bias_rows_end <= nL_total), "Qwen4ExpSession::forward: ",
                "logit bias rows up to ", bias_rows_end, " for a forward of ", nL_total, " logits rows");
    STRIX_TRACE_RANGE(std::string(batch.size() == 1 && runs[0].sq->gdn_to_spare ? "verify"
                                : T == 1 ? "decode" : "prefill") + " T=" + std::to_string(T) +
                      " runs=" + std::to_string(batch.size()));
    if (runs.size() > 1) {  // the batch's decode-size scalar contract (see the header's forward_batch)
        STRIX_CHECK(image == nullptr, "Qwen4ExpSession::forward: ", runs.size(),
                    " runs and an image forward - image rows run alone (vision's one-forward-one-image contract)");
        for (const Run &r : runs)
            STRIX_CHECK(r.sq->pending3.empty() && !r.sq->history_ever_interleaved && r.sq->rope_delta == 0,
                        "Qwen4ExpSession::forward: run on seq ", r.seq,
                        " carries mRoPE position state - a multi-run forward is scalar positions only");
    }
    // Image forward (vision): every check that can fail without touching session state runs here, before
    // the pending positions are consumed. run_tower_and_scatter checks the rest (the row count, against ids).
    // Single-run by the check above: `ids` is that run's.
    const std::vector<int32_t> &ids = batch[0].ids;
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
        // set_positions addresses slot 0 only (the per-slot form comes with the engine's slots): an image
        // forward on another slot would be checked against slot 0's triples.
        STRIX_CHECK(batch[0].seq == 0, ifn, ": an image forward on seq ", batch[0].seq,
                    ", expected seq 0 (set_positions / mRoPE are slot 0's only so far)");
        STRIX_CHECK(!runs[0].sq->pending3.empty(), ifn, ": an image forward carries its set_positions triples (the "
                    "image rows' (t,h,w) section positions) - set them first (Phase A's walk rule)");
        STRIX_CHECK((int64_t)runs[0].sq->pending3.size() / 3 == T, ifn, ": set_positions holds ",
                    runs[0].sq->pending3.size() / 3, " rows, this forward runs ", T);
        const int64_t n_img = std::count(ids.begin(), ids.end(), kVisionImageTokenId);
        STRIX_CHECK(n_img == np / (tower->dims().merge * tower->dims().merge), ifn, ": ids carry ", n_img,
                    " image_token_id rows (", kVisionImageTokenId, "), the image produces ", np, " patches / ",
                    tower->dims().merge * tower->dims().merge, " = ",
                    np / (tower->dims().merge * tower->dims().merge), " merged rows - expand the placeholder "
                    "to the merged token count before the forward");
    }
    // mRoPE: consume this forward's pending triples into the history (set_positions checked the continuation at
    // set time; st().pos cannot have moved - restore / import refuse a pending span). Once interleaved, every
    // forward's rows join the history - scalar rows materialized at their rope_pos - so it stays contiguous and
    // set_positions can keep checking continuations. Text-only forwards: both no-ops, no upload, no allocation.
    // After broken_ = true: a mismatching span (rows != T) takes the session down cleanly instead of leaving
    // a half-consumed history.
    broken_ = true;
    for (Run &r : runs) {
        Qwen4ExpSeqState &sq = *r.sq;
        if (!sq.pending3.empty()) {
            STRIX_CHECK((int64_t)sq.pending3.size() / 3 == r.Tr, "Qwen4ExpSession::forward: set_positions holds ",
                        sq.pending3.size() / 3, " rows, its run forwards ", r.Tr, " - the triples cover exactly one forward");
            // The history holds rows [sq.pos3_start, sq.pos); a text-only prefix needs no rows (their triples are
            // the scalar rope_pos values), so the first interleaved forward starts the history at its own row 0.
            if (sq.pos3_host.empty()) sq.pos3_start = sq.pos;
            sq.pos3_host.insert(sq.pos3_host.end(), sq.pending3.begin(), sq.pending3.end());
            sq.pending3.clear();
            sq.history_ever_interleaved = true;
        } else if (!sq.pos3_host.empty()) {
            for (int64_t row = 0; row < r.Tr; ++row) {
                const int64_t v = sq.pos + row + sq.rope_delta;
                sq.pos3_host.insert(sq.pos3_host.end(), {v, v, v});
            }
        }
        // rope_delta: with an interleaved prefix, every later scalar row's angle sits at (the continuation of
        // the history) - (the sequence index). Recomputed only when the next row is a scalar one: a span ending
        // on a uniform row steps by 1, one ending at a complete image block row continues at the block's max
        // position + 1; a span ending mid-block-row (an image split across forwards) has image rows next - the
        // delta stays at its last block-end value (unused until the image completes, then recomputed).
        if (sq.history_ever_interleaved && !sq.pos3_host.empty()) {  // (a session resumed after the image has the delta
            const Cont next = continuation_of(sq.pos3_host.data() + (size_t)((sq.pos3Hend() - sq.pos3_start - 1) * 3),
                                              sq.pos3_host.data(), (int64_t)sq.pos3_host.size() / 3);
            if (next.kind == ContKind::kUniform || next.kind == ContKind::kEndOrImage)  // its rows are all scalar
                sq.rope_delta = next.exact[0] - (sq.pos + r.Tr);
        }
        // Paged K/V for this run's rows (a multi-run forward skips prepare_positions below).
        ensure_rows(sq, r.p + r.Tr);
        // The batch's last prepared span is what the kernels get (mk_block_ / mk_jobs_): a single run is today's
        // call; a multi-run forward is scalar-only (checked above) and needs no uploads at all.
        if (runs.size() == 1) {
            prepare_positions(sq, r.p, r.Tr);
            r.rope0 = mk_jobs_.pos3 == nullptr ? r.p + sq.rope_delta : r.p;  // scalar-path rope base (Phase A:
            // interleaved calls pass the kernels their positions, so pos0 is unused there)
        } else {
            r.rope0 = r.p;  // scalar with rope_delta 0 (checked above); a pure-text state's rope_pos(p) == p
            // A multi-run forward is scalar-only (checked above), so it must carry no mRoPE window to the
            // kernels: clear whatever the last single-run forward prepared. Left stale, an image's triples
            // (a non-null mk_block_ / mk_jobs_) would be matched against these runs' positions by
            // qk_norm_rope_jobs / qsa_block_keys - a confusing error, or (p == the stale base) silently
            // roping text rows with the image's (t,h,w) angles.
            mk_block_ = {};
            mk_jobs_ = {};
        }
    }
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

    // PLE rows for this forward: the ids advance each run's own n-gram history now; the rows come from
    // set_lookahead's gather when it was for exactly these ids and history (single-run serving), else a gather
    // starts now on a worker - overlapping the embedding and the layers before the PLE layer. Joined there (a
    // gather failure fails the forward). Multi-run: every run gathers into its own state's pinned staging
    // (the lookahead / ping-pong buffers are the single-sequence path); the host-side gathers queue on the
    // worker, so they overlap the GPU's embedding + layer 0 like today's.
    int ple_buf = 0;
    std::future<void> ple_ready;  // single-run: the batch's gather (the look-ahead's when it matches)
    std::vector<int32_t> lookahead = std::move(ple_lookahead_);
    ple_lookahead_.clear();
    if (D.ple_layer >= 0) {
        for (Run &r : runs) {
            Qwen4ExpSeqState &sq = *r.sq;
            const std::vector<int32_t> &rids = batch[(size_t)(&r - runs.data())].ids;
            std::vector<int64_t> tok(rids.begin(), rids.end()), rows((size_t)(r.Tr * 16));
            const PleHistory before = sq.ple_hist;  // before the advance: what a pending gather was keyed on
            ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), r.Tr, sq.ple_hist, rows.data());
            if (runs.size() > 1) {  // per-state gather into its own staging, joined at the PLE layer
                if (sq.ple_stage.size() == 0)
                    sq.ple_stage = PinnedHostBuffer((size_t)(kMaxLogits * D.ple_e) * e, "batched PLE rows");
                r.ple_ready = start_ple_gather(std::move(rows), r.Tr, sq.ple_stage, PleRun::Worker);
                continue;
            }
            if (ple_pending_.active && ple_pending_.ids == rids && ple_pending_.hist.prev == before.prev) {
                ple_buf = ple_pending_.buf, ple_ready = std::move(ple_pending_.done);
                ple_pending_ = PleGather{};
            } else {
                drop_ple_pending();  // a stale hint: let it finish before its buffer is reused
                // Small forwards (decode, MTP verify): on the persistent worker, overlapping the embedding and layer 0.
                ple_ready = start_ple_gather(std::move(rows), r.Tr, ple_host_[ple_buf],
                                             r.Tr >= kWmmaMinTokens ? PleRun::Async : PleRun::Worker);
            }
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

    // Embedding, written straight into all 4 streams (one launch over every run's rows). all_ids is pageable, not
    // pinned: a pinned buffer here would need an event gate before its next write (the gfx1151 pinned-reuse hazard).
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), all_ids.data(), (size_t)T * 4, hipMemcpyHostToDevice, stream_),
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
            auto join_wait = [&](std::future<void> &f, int64_t rows_at, void *stage) {
                const auto t0 = std::chrono::steady_clock::now();
                f.get();
                const int64_t ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                ++ple_waits_, ple_wait_ns_ += ns;
                for (int64_t m = ple_wait_max_ns_.load(); ns > m && !ple_wait_max_ns_.compare_exchange_weak(m, ns);) {
                }
                (void)rows_at, (void)stage;
            };
            if (runs.size() == 1) {
                join_wait(ple_ready, T, nullptr);
                STRIX_HIP_CHECK(hipMemcpyAsync(ple_e_.get(), ple_host_[ple_buf].get(), (size_t)(T * D.ple_e) * e,
                                               hipMemcpyHostToDevice, stream_),
                                "uploading PLE rows");
                // The next forward's rows (set_lookahead) into the other buffer, during this forward's GPU work: that
                // buffer's last upload was in an earlier forward, which ended with a stream sync. The
                // look-ahead is the single-sequence serving path (the engine's); a multi-run batch drops the hint.
                if (!lookahead.empty()) {
                    std::vector<int64_t> tok(lookahead.begin(), lookahead.end()), rows(lookahead.size() * 16);
                    PleHistory h = runs[0].sq->ple_hist;
                    ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), (int64_t)tok.size(), h, rows.data());
                    ple_pending_.buf = 1 - ple_buf;
                    ple_pending_.done = start_ple_gather(std::move(rows), (int64_t)tok.size(),
                                                         ple_host_[ple_pending_.buf], PleRun::Async);
                    ple_pending_.ids = std::move(lookahead), ple_pending_.hist = runs[0].sq->ple_hist,
                        ple_pending_.active = true;
                }
            } else {
                for (Run &r : runs) {
                    join_wait(r.ple_ready, r.Tr, nullptr);
                    STRIX_HIP_CHECK(hipMemcpyAsync(ple_e_.get() + (size_t)(r.base * D.ple_e) * e,
                                                   r.sq->ple_stage.get(), (size_t)(r.Tr * D.ple_e) * e,
                                                   hipMemcpyHostToDevice, stream_),
                                    "uploading batched PLE rows");
                }
            }
            lin(ple_e_.get(), l.ple_kv, ple_kv_.get(), T, act);
            // The gate + conv per run: its rows out of the shared projection, its own conv state, and its own
            // history window slice [S state rows][its rows] of ple_hist_buf_ (the gate kernel writes the state
            // rows itself; windows are laid end to end - S + run 0's rows, then S + run 1's, ...).
            const int64_t ple_S = (D.ple_taps - 1) * D.ple_dil;
            int64_t wbase = 0;
            for (Run &r : runs) {
                const kernels::PleShape sh{r.Tr, D.H, D.d, D.ple_taps, D.ple_dil, D.ple_ld};
                uint8_t *hist_w = ple_hist_buf_.get() + (size_t)(wbase * n4) * e;
                kernels::ple_gate(ple_kv_.get() + (size_t)(r.base * D.ple_ld) * e, x_.get() + (size_t)(r.base * n4) * e,
                                  l.ple_norm_key, l.ple_norm_query, l.ple_norm_conv, r.sq->ple_state.get(), sh, D.eps,
                                  hist_w, ple_sigma_.get() + (size_t)(r.base * D.H), act, stream_);
                kernels::ple_conv(hist_w, ple_sigma_.get() + (size_t)(r.base * D.H),
                                  ple_kv_.get() + (size_t)(r.base * D.ple_ld) * e, l.ple_conv, sh,
                                  x_.get() + (size_t)(r.base * n4) * e, r.sq->ple_state.get(), act, stream_);
                wbase += ple_S + r.Tr;
            }
            show(L + "ple_out", x_.get(), n4, pt);
        }

        if (i == D.ple_layer) inv_ready = false;  // the PLE changed X after the last injection
        STRIX_TRACE_SET(stage, "L" + std::to_string(i) + (D.is_attention[(size_t)i] ? " attn" : " gdn"));
        hc_mix(l.hc_attn, true, inv_ready);
        show(L + "attn_in", u_.get(), D.d, pt);
        show(L + "attn_w_in", w_in_.get(), D.H, ProbeType::F32);
        if (!D.is_attention[(size_t)i]) {
            // A verify writes its in_proj rows straight into the layer's verify_in buffer (keep_verify_prefix may
            // replay them) instead of proj_ and a copy - single-run as today; a batch projects all N rows to
            // proj_ and copies each verifying run's slice (the same bytes the direct write would have stored).
            const bool direct_verify_rows = runs.size() == 1 && runs[0].sq->gdn_to_spare;
            uint8_t *gp = direct_verify_rows ? runs[0].sq->verify_in[(size_t)i].get() : proj_.get();
            lin(u_.get(), l.in_proj, gp, T, act);
            if (!direct_verify_rows)
                for (Run &r : runs)
                    if (r.sq->gdn_to_spare)
                        STRIX_HIP_CHECK(hipMemcpyAsync(r.sq->verify_in[(size_t)i].get(),
                                                       gp + (size_t)(r.base * D.gstride) * e,
                                                       (size_t)(r.Tr * D.gstride) * e, hipMemcpyDeviceToDevice, stream_),
                                        "Qwen4ExpSession::forward: the batched verify's in_proj rows");
            for (Run &r : runs) {
                Qwen4ExpSeqState &sq = *r.sq;
                const int64_t qkv_off = (size_t)(r.base * D.conv_c) * e;
                kernels::gdn_front(gp + (size_t)(r.base * D.gstride) * e,
                                   {r.Tr, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                                   l.conv_w, sq.conv_state[(size_t)i].get(),
                                   (sq.gdn_to_spare ? sq.conv_spare : sq.conv_state)[(size_t)i].get(), r.fresh, l.A_log,
                                   l.dt_bias, qkv_.get() + qkv_off, beta_.get() + (size_t)(r.base * D.gv),
                                   g_.get() + (size_t)(r.base * D.gv), act, stream_);
                // Prefill-sized WMMA forwards: the chunked delta rule on the matrix units; decode / MTP verify per token.
                kernels::delta_rule(qkv_.get() + qkv_off, beta_.get() + (size_t)(r.base * D.gv),
                                    g_.get() + (size_t)(r.base * D.gv), {r.Tr, D.gk, D.gv, D.conv_c},
                                    sq.rec_state[(size_t)i].get(),
                                    (sq.gdn_to_spare ? sq.rec_spare : sq.rec_state)[(size_t)i].get(), r.fresh,
                                    core_.get() + (size_t)(r.base * D.gz) * e, act, stream_,
                                    wmma && r.Tr >= kWmmaMinTokens ? kernels::DeltaRulePath::ChunkedWmma
                                                                   : kernels::DeltaRulePath::Auto,
                                    precision_.gdn);
            }
            // z (gv heads of kHD a token) read in place from the in_proj rows (gp: all rows; the verifying
            // runs' rows are in proj_ too, the bytes their verify_in copies).
            kernels::rmsnorm_gated(core_.get(),
                                   (direct_verify_rows ? gp : proj_.get()) + (size_t)D.conv_c * e, l.gdn_norm,
                                   gnorm_.get(), T * D.gv, kHD, D.eps,
                                   kernels::GateAct::Sigmoid, act, stream_, D.gv, D.gstride);
            show(L + "gdn_core", gnorm_.get(), D.gz, pt);
            lin(gnorm_.get(), l.out_proj, y_.get(), T, act);
        } else {
            // 8-bit K/V (the served format; the plain caches are empty then): the new rows go to a staging
            // area first - a run's into kv_new_, or (verify, any batch size) straight into its own saved
            // rows for keep_verify_prefix - and kv8_commit per run then quantizes the groups it completes
            // and writes that state's new tail.
            const bool q8 = kv_format_ == KvFormat::Q8;
            const int64_t W = D.hkv * hd;
            lin(u_.get(), l.qkv, proj_.get(), T, act);
            for (const Run &r : runs) {
                Qwen4ExpSeqState &sq = *r.sq;
                uint8_t *kc = sq.k_cache[(size_t)i].capacity() ? sq.k_cache[(size_t)i].get() : nullptr,
                        *vc = sq.v_cache[(size_t)i].capacity() ? sq.v_cache[(size_t)i].get() : nullptr;
                uint8_t *kv_rows = !q8 ? nullptr
                                   : sq.gdn_to_spare ? sq.verify_in[(size_t)i].get() +
                                                             (size_t)(kMaxLogits * D.idx_d) * e
                                                     : kv_new_.get() + (size_t)(r.base * 2 * W) * e;
                uint8_t *k_out = q8 ? kv_rows : kc + (size_t)(r.p * W) * e,
                        *v_out = q8 ? kv_rows + (size_t)W * e : vc + (size_t)(r.p * W) * e;
                const int64_t kv_ts = q8 ? 2 * W : W;
                // One launch per run: q normed + roped in place, k into the K cache, v copied into the V
                // cache, past the dense limit the indexer queries (normed + roped in place), and in a verify
                // the raw indexer keys copied into the layer's verify_in (keep_verify_prefix may replay
                // them) - kernels/attention.hpp qk_norm_rope_jobs.
                const bool qsa_layer = r.p + r.Tr > D.dense_key_limit();
                kernels::NormRopeJob nr[5] = {
                    {at(proj_, r.base * D.astride), D.astride, 2 * hd, at(proj_, r.base * D.astride), D.astride,
                     2 * hd, D.hq, hd, l.q_norm},
                    {at(proj_, r.base * D.astride + D.hq * 2 * hd), D.astride, hd, k_out, kv_ts, hd, D.hkv, hd,
                     l.k_norm},
                    {at(proj_, r.base * D.astride + D.hq * 2 * hd + D.hkv * hd), D.astride, hd, v_out, kv_ts, hd,
                     D.hkv, hd, nullptr},
                    {at(proj_, r.base * D.astride + D.idx_col), D.astride, D.idx_d,
                     at(proj_, r.base * D.astride + D.idx_col), D.astride, D.idx_d, D.idx_h, D.idx_d,
                     l.idx_q_norm}};
                int njobs = 3;
                if (qsa_layer) ++njobs;
                else nr[3] = {};
                if (sq.gdn_to_spare)
                    nr[njobs++] = {at(proj_, r.base * D.astride + D.idx_col + D.idx_h * D.idx_d), D.astride, D.idx_d,
                                   sq.verify_in[(size_t)i].get(), D.idx_d, D.idx_d, 1, D.idx_d, nullptr};
                kernels::qk_norm_rope_jobs(nr, njobs, r.Tr, D.eps, D.rot, r.rope0, m_.inv_freq(), act, stream_,
                                           m_.rope_scale(), mk_jobs_);
                // Indexer block keys (every completed block, this run's included) - kept current for when
                // selection starts to bite; the queries and scores are only needed past the dense limit.
                // pos0 stays the sequence index (the kernel derives block / tail array indices from it); with
                // a rope_delta the window's triples carry the shifted angles (a shifted pos0 would shift the
                // block partition too).
                kernels::qsa_block_keys(at(proj_, r.base * D.astride + D.idx_col + D.idx_h * D.idx_d), D.astride,
                                        r.Tr, r.p, r.p % kernels::kQsaBlock ? sq.tail[sq.tail_cur][(size_t)i].get() : nullptr,
                                        sq.tail[1 - sq.tail_cur][(size_t)i].get(), D.idx_d, l.idx_k_norm, D.eps,
                                        m_.inv_freq(), D.rot, sq.block_keys[(size_t)i].get(), cap_blocks_, act, stream_,
                                        m_.rope_scale(), mk_block_);
                const kernels::AttentionShape as{r.Tr, r.p, D.hq, D.hkv, hd, D.astride, 2 * hd};
                const float scale = 1.0f / std::sqrt((float)hd);
                kernels::Kv8View kv8;
                if (q8) {
                    const size_t li = (size_t)i;
                    kernels::kv8_commit(k_out, v_out, kv_ts, r.Tr, r.p,
                                        r.p % kernels::kKv8Group ? sq.kv8_tail[sq.tail_cur][li].get() : nullptr,
                                        sq.kv8_tail[1 - sq.tail_cur][li].get(), sq.kv8_store[li].get(), kv8_cap_, W, act,
                                        kv8_err_.get(), stream_);
                    kv8 = {sq.kv8_store[li].get(), kv8_cap_, W,
                           r.p % kernels::kKv8Group ? sq.kv8_tail[sq.tail_cur][li].get() : nullptr, k_out, v_out, kv_ts,
                           r.p};
                }
                uint8_t *q_r = at(proj_, r.base * D.astride), *core_r = core_.get() + (size_t)(r.base * D.hq * hd) * e;
                // BF16 activations: the attention on the matrix units (kernels/attention.hpp, kWmmaAttnMinTokens),
                // chosen per run by its own rows; a run reads its queries / writes its outputs at its row base
                // (forward_qwen4exp_batch_vs_solo: bit-identical to that run alone).
                const bool wmma_attn = wmma && act == Act::BF16 && r.Tr >= kWmmaAttnMinTokens;
                // Dequantize the rows below the last query's exact_from once, then plain loads (Kv8View
                // staged_k), when that is cheaper than unpacking in the loop (kv8_stage_pays).
                const auto stage_kv8 = [&](int64_t keys_per_query) {
                    const int64_t rows = kernels::kv8_exact_from(r.p + r.Tr - 1);
                    if (!(wmma_attn && q8 && rows > 0 && kv8_stage_pays(r.Tr, keys_per_query, rows))) return;
                    STRIX_CHECK(kv8_stage_.size() >= (size_t)(2 * kv8_cap_ * W) * 2, "Qwen4ExpSession::forward: ",
                                "8-bit staging buffer of ", kv8_stage_.size(), " bytes for a ", r.Tr, "-token forward");
                    uint8_t *sk = kv8_stage_.get(), *sv = sk + (size_t)(kv8_cap_ * W) * 2;
                    kernels::kv8_dequant(kv8.store, kv8_cap_, W, 0, rows, sk, sv, act, true, stream_);  // F16 K
                    kv8.staged_k = sk, kv8.staged_v = sv, kv8.staged_rows = rows;
                };
                if (!qsa_layer) {
                    stage_kv8(r.p + r.Tr);  // a dense query reads every row up to its own
                    if (wmma_attn && q8)
                        kernels::attention_wmma(q_r, q_r + (size_t)hd * e, kv8, as, scale, core_r, attn_ws_.get(),
                                                attn_ws_bytes_, act, stream_, precision_.attention);
                    else if (wmma_attn)
                        kernels::attention_wmma(q_r, q_r + (size_t)hd * e, kc, vc, capacity_, as, scale, core_r,
                                                attn_ws_.get(), attn_ws_bytes_, act, stream_, precision_.attention);
                    else if (q8)
                        kernels::attention(q_r, q_r + (size_t)hd * e, kv8, as, scale, core_r, attn_ws_.get(),
                                           attn_ws_bytes_, act, stream_);
                    else
                        kernels::attention(q_r, q_r + (size_t)hd * e, kc, vc, capacity_, as, scale, core_r,
                                           attn_ws_.get(), attn_ws_bytes_, act, stream_);
                } else {
                    // QSA: indexer queries (normed + roped in place), block scores, the top budget / 4
                    // blocks, gathered attention. Queries that still see every key keep all their blocks -
                    // the dense set. Scores / selection are row-indexed: each run gets its slice.
                    const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
                    void *iq = at(proj_, r.base * D.astride + D.idx_col);  // normed + roped by the jobs above
                    kernels::qsa_scores(iq, D.astride, D.idx_d, r.Tr, r.p, D.idx_h, D.idx_d,
                                        sq.block_keys[(size_t)i].get(), cap_blocks_,
                                        qsa_scores_.get() + (size_t)(r.base * cap_blocks_), cap_blocks_, act, stream_);
                    kernels::qsa_topk(qsa_scores_.get() + (size_t)(r.base * cap_blocks_), cap_blocks_, r.Tr, r.p, qsa_k,
                                      qsa_sel_.get() + (size_t)(r.base * qsa_k), qsa_nsel_.get() + r.base,
                                      qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
                    show(L + "qsa_sel", qsa_sel_.get() + (size_t)(r.base * qsa_k), qsa_k, ProbeType::I32);
                    show(L + "qsa_nsel", qsa_nsel_.get() + r.base, 1, ProbeType::I32);
                    stage_kv8(std::min<int64_t>(D.qsa_budget, r.p + r.Tr));
                    if (wmma_attn && q8)
                        kernels::attention_gathered_wmma(q_r, q_r + (size_t)hd * e, kv8, as,
                                                         qsa_sel_.get() + (size_t)(r.base * qsa_k),
                                                         qsa_nsel_.get() + r.base, qsa_k, scale, core_r, attn_ws_.get(),
                                                         attn_ws_bytes_, attn_err_.get(), act, stream_, precision_.attention);
                    else if (wmma_attn)
                        kernels::attention_gathered_wmma(q_r, q_r + (size_t)hd * e, kc, vc, capacity_, as,
                                                         qsa_sel_.get() + (size_t)(r.base * qsa_k),
                                                         qsa_nsel_.get() + r.base, qsa_k, scale, core_r, attn_ws_.get(),
                                                         attn_ws_bytes_, attn_err_.get(), act, stream_, precision_.attention);
                    else if (q8)
                        kernels::attention_gathered(q_r, q_r + (size_t)hd * e, kv8, as,
                                                    qsa_sel_.get() + (size_t)(r.base * qsa_k), qsa_nsel_.get() + r.base,
                                                    qsa_k, scale, core_r, attn_ws_.get(), attn_ws_bytes_, attn_err_.get(),
                                                    act, stream_);
                    else
                        kernels::attention_gathered(q_r, q_r + (size_t)hd * e, kc, vc, capacity_, as,
                                                    qsa_sel_.get() + (size_t)(r.base * qsa_k), qsa_nsel_.get() + r.base,
                                                    qsa_k, scale, core_r, attn_ws_.get(), attn_ws_bytes_, attn_err_.get(),
                                                    act, stream_);
                }
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
        // The shared expert as dense projections (its h through gu_ / hh_ before the routed experts reuse them).
        auto shared_expert = [&] {
            lin(u_.get(), l.shared_gate_up, gu_.get(), T, act);
            kernels::swiglu(gu_.get(), hh_.get(), T, D.inter, act, stream_);
            show(L + "shared_h", hh_.get(), D.inter, pt);
            lin(hh_.get(), l.shared_down, sh_y_.get(), T, act);
        };
        // Prefill (the grouped combine, then the injection's fused form): the shared expert's gated add rides in the
        // combine's reduce (kernels::MoeSharedAdd, bit-identical to moe_shared_add after it), so it runs first.
        const bool fold_shared = sep && fuse_inv && T >= kGroupedMinTokens;
        if (fold_shared) shared_expert();
        const kernels::MoeSharedAdd shared_add{sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1,
                                               (uint32_t)D.experts, router_err_.get()};
        bool h_f16 = false;
        if (T >= kGroupedMinTokens) {
            // Q4 experts on the matrix units take F16; Q5 stays on BF16 scaled codes
            // until its kernel moves over.
            const kernels::MoeMath em = moe_math == kernels::MoeMath::WmmaBf16 && l.gate_up.bits == 4 &&
                                                l.down.bits == 4
                                            ? kernels::MoeMath::WmmaF16
                                            : moe_math;
            kernels::moe_group_routes(route_ids_.get(), T, A, Estack, group_ws_.get(), group_ws_bytes_,
                                      expert_err_.get(), stream_);
            const bool fused = em == kernels::MoeMath::WmmaF16 && l.gate_up.bits == 4;
            // h as F16 bits between the fused gather and a Q4 down: the down's x conversion done once (bit-identical)
            h_f16 = fused && l.down.bits == 4 && act == Act::BF16;
            if (fused) {  // SwiGLU in the gather's epilogue
                kernels::linear_q4_experts_gather_grouped_swiglu(u_.get(), l.gate_up.q4, Estack, group_ws_.get(),
                                                                 group_ws_bytes_, T, A, hh_.get(), act, em, stream_,
                                                                 h_f16);
            } else {
                kernels::linear_qw_experts_gather_grouped(u_.get(), l.gate_up, Estack, group_ws_.get(), group_ws_bytes_,
                                                          T, A, gu_.get(), act, em, stream_);
                kernels::swiglu(gu_.get(), hh_.get(), T * A, D.inter, act, stream_);
            }
            kernels::linear_qw_experts_combine_grouped(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(),
                                                       group_ws_.get(), group_ws_bytes_, group_partial_.get(), T, A,
                                                       y_.get(), act, em, stream_, fold_shared ? &shared_add : nullptr,
                                                       h_f16);
        } else {
            kernels::linear_qw_experts_gather_swiglu(u_.get(), l.gate_up, Estack, route_ids_.get(), T, A, hh_.get(),
                                                     act, expert_err_.get(), stream_);
            kernels::linear_qw_experts_combine(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(), T, A,
                                               y_.get(), act, expert_err_.get(), stream_);
        }
        show(L + "experts_h", hh_.get(), A * D.inter, h_f16 ? ProbeType::F16 : pt);  // the routed experts' down input, T rows of A slots
        if (sep && !fold_shared) shared_expert();  // then y += sigmoid(its gate logit) * its output
        if (sep && !fuse_inv) {  // decode / verify: the gated add and the injection in one launch (bit-identical)
            kernels::moe_shared_add_inject(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                           D.d, (uint32_t)D.experts, x_.get(), w_in_.get(), D.H, act,
                                           router_err_.get(), stream_);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = false;
        } else {
            if (sep && !fold_shared)
                kernels::moe_shared_add(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                        D.d, (uint32_t)D.experts, act, router_err_.get(), stream_);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = inject_y();
        }
        show(L + "out", x_.get(), n4, pt);
    }
    // Indexer tails: every attention layer wrote tail_out = the other buffer this call, per run.
    for (Run &r : runs) r.sq->tail_cur = 1 - r.sq->tail_cur;

    STRIX_TRACE_SET(stage, "head");
    hc_mix(m_.final_mix(), false, inv_ready);
    show("final_mixed", u_.get(), D.d, pt);
    // The logits rows: each run's last nL rows. One run whose rows end the batch (today's serving, and every
    // batch of one) reads them straight out of the streams; a multi-run batch first compacts the wanted rows
    // into lg_src_, so the LM head's weights are read once for the whole batch - the batch's point (the head is
    // ~640 MB). The returned rows are run 0's, then run 1's - the order candidates() comes back in.
    int64_t lrow = 0;
    if (runs.size() == 1) {
        if (runs[0].nL > 0)
            lin(at(u_, (T - runs[0].nL) * D.d), m_.lm_head(), logits_.get(), runs[0].nL, Act::F32);
        lrow = runs[0].nL;
    } else if (nL_total > 0) {
        for (const Run &r : runs) {
            if (r.nL == 0) continue;
            STRIX_HIP_CHECK(hipMemcpyAsync(lg_src_.get() + (size_t)(lrow * D.d) * e,
                                           at(u_, (r.base + r.Tr - r.nL) * D.d), (size_t)(r.nL * D.d) * e,
                                           hipMemcpyDeviceToDevice, stream_),
                            "Qwen4ExpSession::forward: logits-row compaction");
            lrow += r.nL;
        }
        lin(lg_src_.get(), m_.lm_head(), logits_.get(), nL_total, Act::F32);
    }
    const int64_t n_logits = lrow;  // the batch's returned logits rows
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
        if (bias_n > 0) {
            STRIX_HIP_CHECK(hipMemcpyAsync(bias_dev_.get(), bias_host_.get(),
                                           (size_t)bias_n * sizeof(kernels::LogitBiasEntry), hipMemcpyHostToDevice,
                                           stream_),
                            "Qwen4ExpSession::forward: logit bias upload");
            STRIX_HIP_CHECK(hipEventRecord(bias_uploaded_, stream_), "Qwen4ExpSession::forward: logit bias upload event");
            bias_upload_recorded_ = true;
            kernels::logits_bias(logits_.get(), n_logits, D.vocab, bias_dev_.get(), bias_n, stream_);
        }
        kernels::logits_topk(logits_.get(), n_logits, D.vocab, cand_n_valid, cd,
                             reinterpret_cast<uint32_t *>(cd + n_logits * kernels::kLogitCands), cand_ws_.get(),
                             cand_ws_bytes_, stream_, mask_rows > 0 ? mask_dev_.get() : nullptr,
                             mask_rows > 0 ? mask_words : 0);
    }

    // MTP: each run gets the head layer's K / V (and indexer keys) for its positions, each row from its
    // token and the trunk's streams after the position before - what a later draft attends to. The rest of the
    // head layer only matters for a draft's own row, so it's skipped here. The head's own projections / mixes
    // launch per run (its weights are a few MB; batching the catch-up across runs is a later small idea).
    if (mtp_) {
        const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
        // The head's input rows per run (its own mtp_prev streams + the trunk's x_ rows before each of its
        // rows) into the head workspaces' row slices, then the head's own mix / qkv projection once over all
        // N rows (stateless - the catch-up shares the batch's weight reads too). mtp_input can only mix rows
        // of one state: run_mtp swaps mtp_x_ against that state's chain buffer mid-draft.
        for (Run &r : runs) mtp_input(r.base, r.Tr, r.sq->mtp_prev.get());
        hc_mix(ml.hc_attn, true, false, mtp_x_.get());
        lin(u_.get(), ml.qkv, proj_.get(), T, act);
        for (Run &r : runs) {
            Qwen4ExpSeqState &sq = *r.sq;
            const kernels::NormRopeJob nr[2] = {
                {at(proj_, r.base * D.astride + D.hq * 2 * hd), D.astride, hd,
                 sq.k_cache_mtp.get() + (size_t)(r.p * D.hkv * hd) * e, D.hkv * hd, hd, D.hkv, hd, ml.k_norm},
                {at(proj_, r.base * D.astride + D.hq * 2 * hd + D.hkv * hd), D.astride, hd,
                 sq.v_cache_mtp.get() + (size_t)(r.p * D.hkv * hd) * e, D.hkv * hd, hd, D.hkv, hd, nullptr}};
            kernels::qk_norm_rope_jobs(nr, 2, r.Tr, D.eps, D.rot, r.rope0, m_.inv_freq(), act, stream_,
                                       m_.rope_scale(), mk_jobs_);  // k, v into the MTP cache (at row base)
            kernels::qsa_block_keys(at(proj_, r.base * D.astride + D.idx_col + D.idx_h * D.idx_d), D.astride,
                                    r.Tr, r.p,
                                    r.p % kernels::kQsaBlock ? sq.tail_mtp[sq.mtp_tail_cur].get() : nullptr,
                                    sq.tail_mtp[1 - sq.mtp_tail_cur].get(), D.idx_d, ml.idx_k_norm, D.eps,
                                    m_.inv_freq(), D.rot, sq.block_keys_mtp.get(), cap_blocks_, act, stream_,
                                    m_.rope_scale(), mk_block_);
            sq.mtp_tail_cur = 1 - sq.mtp_tail_cur;
            STRIX_HIP_CHECK(hipMemcpyAsync(sq.mtp_prev.get(), at(x_, (r.base + r.Tr - 1) * n4), (size_t)n4 * e,
                                           hipMemcpyDeviceToDevice, stream_),
                            "MTP previous streams");
        }
    }
    std::vector<float> logits = cands ? read_back(0, "", n_logits) : read_back((size_t)(n_logits * D.vocab), "");
    for (Run &r : runs) r.sq->pos += r.Tr, r.sq->verify_carry = 0;  // the run's rows consumed its carried token
    broken_ = false;
    for (Run &r : runs) r.sq->mtp_chain_step = -1;
    return logits;
}std::vector<float> Qwen4ExpSession::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits,
                                                   const std::vector<int64_t> &pos3) {
    const int64_t pending = pending_verify_seq();
    STRIX_CHECK(pending < 0, "Qwen4ExpSession::forward_verify: a forward_verify on seq ", pending,
                " awaits keep_verify / drop_verify");
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward_verify: an earlier call failed midway; reset() the session first");
    // mRoPE: the verify's rows go through the same pending path as a forward's (set_positions validates them);
    // the forward below appends them (and materializes the scalar ones) to the history, and keep / drop
    // finalize or truncate it. A scalar verify with an interleaved span pending would silently drop it - refused.
    STRIX_CHECK(pos3.empty() || (int64_t)pos3.size() / 3 == (int64_t)ids.size(),
                "Qwen4ExpSession::forward_verify: pos3 has ", pos3.size() / 3, " rows, the verify runs ",
                ids.size(), " - one triple per row");
    if (!pos3.empty()) set_positions(pos3);
    else STRIX_CHECK(st().pending3.empty(), "Qwen4ExpSession::forward_verify: scalar verify while set_positions holds ",
                     st().pending3.size() / 3, " interleaved rows - pass them as pos3, or clear first");
    STRIX_CHECK((int64_t)ids.size() <= kMaxLogits, "Qwen4ExpSession::forward_verify: ", ids.size(), " tokens, expected <= ",
                kMaxLogits, " (the saved rows for keep_verify_prefix)");
    verify_prep(st());
    st().verify_ids = ids;
    st().verify_rows = (int64_t)ids.size(), st().verify_carry = 0;
    st().gdn_to_spare = true;
    try {
        std::vector<float> logits = forward(ids, n_logits);
        st().gdn_to_spare = false;
        st().verify_pending = true;
        return logits;
    } catch (...) {
        st().gdn_to_spare = false;
        st().verify_pending = false, st().verify_rows = 0;
        throw;
    }
}

// The seq of the first state with a pending verify (-1: none). While one awaits its outcome the session's shared
// forward / replay buffers are the verify's, so every other call that touches them refuses - with more slots the
// pending verify must not block reads, only forwards / replays / drafts are gated on it.
int64_t Qwen4ExpSession::pending_verify_seq() const {
    for (size_t k = 0; k < seqs_.size(); ++k)
        if (seqs_[k]->verify_pending) return (int64_t)k;
    return -1;
}

// One state's verify preparation, shared by the single-sequence and batched paths: the mRoPE before-state (the
// forward appends the verify's rows; keep / drop decide), the spare set (allocated on the first use - it needs
// the session's kMaxLogits row sizing), the kept small states, and the position / history bookkeeping. The
// caller then records the verify's ids, sets gdn_to_spare and runs the forward.
void Qwen4ExpSession::verify_prep(Qwen4ExpSeqState &sq) {
    sq.verify_hist_before = sq.pos3_host;
    sq.verify_hist_start_before = sq.pos3_start;
    sq.verify_delta_before = sq.rope_delta;
    sq.verify_ever_before = sq.history_ever_interleaved;
    if (sq.conv_spare.empty()) {
        for (size_t l = 0; l < sq.conv_state.size(); ++l) {
            const std::string tag = "spare layer " + std::to_string(l);
            sq.conv_spare.emplace_back(sq.conv_state[l].size() ? DeviceBuffer<uint8_t>(sq.conv_state[l].size(), "conv state " + tag)
                                                        : DeviceBuffer<uint8_t>());
            sq.rec_spare.emplace_back(sq.rec_state[l].size() ? DeviceBuffer<float>(sq.rec_state[l].size(), "recurrent state " + tag)
                                                     : DeviceBuffer<float>());
        }
        if (sq.ple_state.size()) sq.ple_state_kept = DeviceBuffer<uint8_t>(sq.ple_state.size(), "kept PLE conv state");
        if (mtp_) sq.mtp_prev_kept = DeviceBuffer<uint8_t>(sq.mtp_prev.size(), "kept MTP previous streams");
        const Qwen4ExpDims &D = m_.dims();
        const size_t e = es(m_.act());
        for (size_t l = 0; l < sq.conv_state.size(); ++l)
            sq.verify_in.emplace_back(
                (size_t)(kMaxLogits * (D.is_attention[l] ? D.idx_d + (kv_format_ == KvFormat::Q8 ? 2 * D.hkv * D.hd : 0)
                                                         : D.gstride)) * e,
                "verify rows, layer " + std::to_string(l));
    }
    // What the forward changes in place (small): kept for drop_verify. The GDN states go to the spare set.
    copy_state(sq.ple_state, sq.ple_state_kept, stream_, "PLE conv state");
    copy_state(sq.mtp_prev, sq.mtp_prev_kept, stream_, "MTP previous streams");
    sq.verify_pos = sq.pos, sq.verify_tail_cur = sq.tail_cur, sq.verify_mtp_tail_cur = sq.mtp_tail_cur,
        sq.verify_hist = sq.ple_hist;
    // The batch row / PLE window bases: 0 for a single-sequence verify (forward_batch_verify sets the batched
    // ones, mirroring the order forward_batch lays the runs in).
    sq.verify_row_base = 0, sq.verify_ple_wbase = 0;
}

// forward_batch with the verify mechanism (the batched MTP verify): every run's state is prepared like
// forward_verify's, the batched forward writes each verifying state's GDN states into its spare set and saves
// its rows for the prefix replay, and the batch's logits come back run-concatenated. Every run resolves with
// keep_verify(seq, ..) / drop_verify(seq) before anything else runs on the session; the acceptance decisions
// differ per run, so the engine resolves them one by one.
std::vector<float> Qwen4ExpSession::forward_batch_verify(std::vector<BatchRun> &runs) {
    const char *fn = "Qwen4ExpSession::forward_batch_verify";
    const int64_t pending = pending_verify_seq();
    STRIX_CHECK(pending < 0, fn, ": a forward_verify on seq ", pending, " awaits keep_verify / drop_verify - "
                "resolve the pending batch first");
    STRIX_CHECK(!broken_, fn, ": an earlier call failed midway; reset() the session first");
    STRIX_CHECK(!runs.empty() && runs.size() <= (size_t)n_slots(), fn, ": ", runs.size(), " runs, expected 1..",
                n_slots());
    for (size_t k = 0; k < runs.size(); ++k) {
        STRIX_CHECK(runs[k].seq >= 0 && runs[k].seq < n_slots(), fn, ": run ", k, " on seq ", runs[k].seq,
                    ", expected 0..", n_slots() - 1);
        for (size_t j = 0; j < k; ++j)
            STRIX_CHECK(runs[j].seq != runs[k].seq, fn, ": seq ", runs[k].seq, " carries two runs (", j,
                        " and ", k, ")");
        STRIX_CHECK((int64_t)runs[k].ids.size() <= kMaxLogits, fn, ": run ", k, " has ", runs[k].ids.size(),
                    " rows, expected <=", kMaxLogits, " (the saved rows for keep_verify)");
        // (forward_batch checks the row budget, the capacity, and the scalar / no-image multi-run contract)
    }
    // Each run's bases in the batch's shared row buffers mirror the order forward_batch lays the runs in
    // (the batch vector's): trunk/MTP activation rows at the id prefix sum, PLE history windows laid end to
    // end (S state rows + the run's rows each). keep_verify_prefix_seq's replay reads them back per state.
    const int64_t ple_S = m_.dims().ple_layer >= 0 ? (m_.dims().ple_taps - 1) * m_.dims().ple_dil : 0;
    int64_t base = 0, wbase = 0;
    for (const BatchRun &br : runs) {
        Qwen4ExpSeqState &sq = *seqs_[(size_t)br.seq];
        verify_prep(sq);
        sq.verify_ids = br.ids;
        sq.verify_rows = (int64_t)br.ids.size(), sq.verify_carry = 0;
        sq.verify_row_base = base, sq.verify_ple_wbase = wbase;
        sq.gdn_to_spare = true;
        base += (int64_t)br.ids.size();
        if (ple_S) wbase += ple_S + (int64_t)br.ids.size();
    }
    std::vector<float> logits;
    try {
        logits = forward_batch(runs, nullptr, {});
    } catch (...) {
        for (const BatchRun &br : runs) {
            Qwen4ExpSeqState &sq = *seqs_[(size_t)br.seq];
            sq.gdn_to_spare = false, sq.verify_pending = false, sq.verify_rows = 0;
        }
        throw;
    }
    for (const BatchRun &br : runs) {
        Qwen4ExpSeqState &sq = *seqs_[(size_t)br.seq];
        sq.gdn_to_spare = false;
        sq.verify_pending = true;
    }
    return logits;
}

void Qwen4ExpSession::keep_verify() {
    STRIX_CHECK(st().verify_pending, "Qwen4ExpSession::keep_verify: no forward_verify to keep");
    keep_verify_full(st());
}

// A kept whole verify: the spare set becomes current (as if the verify had been a forward()). The rows past
// the kept prefix (keep_verify_prefix_seq) are the verify's rows the next forward rewrites.
void Qwen4ExpSession::keep_verify_full(Qwen4ExpSeqState &sq) {
    std::swap(sq.conv_state, sq.conv_spare);
    std::swap(sq.rec_state, sq.rec_spare);
    // mRoPE: the verify's rows are kept - the history / delta already include them (forward appended them).
    sq.verify_hist_before.clear();
    sq.verify_pending = false, sq.verify_rows = 0, sq.verify_carry = 0;
}

void Qwen4ExpSession::drop_verify() {
    STRIX_CHECK(st().verify_pending, "Qwen4ExpSession::drop_verify: no forward_verify to drop");
    drop_verify_seq(st());
}

// The batched resolution's none-kept: seq's rows never happened, its states roll back to before the verify
// (the other runs of the batch resolve on their own - their acceptance differs).
void Qwen4ExpSession::drop_verify(int seq) {
    STRIX_CHECK(seq >= 0 && seq < n_slots(), "Qwen4ExpSession::drop_verify: seq ", seq, ", expected 0..",
                n_slots() - 1);
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(sq.verify_pending, "Qwen4ExpSession::drop_verify: seq ", seq, " has no pending verify");
    drop_verify_seq(sq);
}

void Qwen4ExpSession::drop_verify_seq(Qwen4ExpSeqState &sq) {
    copy_state(sq.ple_state_kept, sq.ple_state, stream_, "PLE conv state");
    copy_state(sq.mtp_prev_kept, sq.mtp_prev, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::drop_verify");
    sq.pos = sq.verify_pos, sq.tail_cur = sq.verify_tail_cur, sq.mtp_tail_cur = sq.verify_mtp_tail_cur,
        sq.ple_hist = sq.verify_hist;
    // mRoPE: the verify's rows never happened - history and rope_delta revert to before it.
    sq.pos3_host = sq.verify_hist_before, sq.rope_delta = sq.verify_delta_before;
    sq.pos3_start = sq.verify_hist_start_before, sq.history_ever_interleaved = sq.verify_ever_before;
    sq.verify_hist_before.clear();
    sq.mtp_chain_step = -1;
    sq.rewind_to.push_back(sq.verify_pos);  // rows past it were overwritten: snapshots beyond it no longer match
    sq.verify_pending = false;
    sq.verify_rows = 0, sq.verify_carry = 0;
}

// The batched resolution: end seq's pending verify keeping its first `rows` rows - the whole verify
// (rows == its verify_rows: today's keep_verify()) or its prefix (today's keep_verify_prefix(rows)), chosen by
// the count. carry: the token the caller feeds next as the run's row 0 (the MTP rejection's corrected sample,
// or with the whole verify kept the token sampled from its last row) - recorded on the state and consumed
// (checked and cleared) by the run's next forward_batch / forward_batch_verify: exactly like the
// single-sequence engine after keep_verify_prefix, whose corrected token becomes the next verify's row 0,
// nothing is advanced or replayed for the carry here (its trunk rows would take the full forward to write -
// the run's PLE rows reach it through prefetch_ple as today). The last resolution frees the session.
void Qwen4ExpSession::keep_verify(int seq, int64_t rows, int32_t carry) {
    const char *fn = "Qwen4ExpSession::keep_verify";
    STRIX_CHECK(seq >= 0 && seq < n_slots(), fn, ": seq ", seq, ", expected 0..", n_slots() - 1);
    STRIX_CHECK(carry >= 0 && carry < m_.dims().vocab, fn, ": carried token ", carry, ", expected 0..",
                m_.dims().vocab - 1, " (0: the caller forwards nothing)");
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(sq.verify_pending, fn, ": seq ", seq, " has no pending verify");
    const int64_t R = sq.verify_rows;
    STRIX_CHECK(R >= 1, fn, ": seq ", seq, "'s pending verify recorded ", R, " rows (internal)");
    STRIX_CHECK(rows >= 1 && rows <= R, fn, ": keep ", rows, " rows of seq ", seq, "'s ", R,
                "-row verify, expected 1..", R, " (none: drop_verify)");
    STRIX_CHECK(sq.pending3.empty(), fn, ": seq ", seq, "'s set_positions holds ", sq.pending3.size() / 3,
                " interleaved rows while a verify is pending - resolve the verify first");
    if (rows < R)
        keep_verify_prefix_seq(sq, rows);
    else
        keep_verify_full(sq);
    sq.verify_carry = carry;
}

void Qwen4ExpSession::keep_verify_prefix(int64_t rows) {
    STRIX_CHECK(st().verify_pending, "Qwen4ExpSession::keep_verify_prefix: no forward_verify pending");
    const int64_t T = (int64_t)st().verify_ids.size();
    STRIX_CHECK(rows >= 1 && rows < T, "Qwen4ExpSession::keep_verify_prefix: keep ", rows, " rows of a ", T,
                "-row verify, expected 1..", T - 1, " (all of them: keep_verify; none: drop_verify)");
    STRIX_CHECK(st().pending3.empty(), "Qwen4ExpSession::keep_verify_prefix: set_positions holds ", st().pending3.size() / 3,
                " interleaved rows while a verify is pending - resolve the verify first");
    keep_verify_prefix_seq(st(), rows);
}

// The prefix body of a state's pending verify (1 <= rows < its verify_rows; the callers check): the state as
// if the verify had been a forward() over its first rows, replayed from the states before the verify (which
// the verify left intact) over its saved rows - the engine then continues at this position (with keep_verify's
// carry token, or with the next verify's row 0).
void Qwen4ExpSession::keep_verify_prefix_seq(Qwen4ExpSeqState &sq, int64_t rows) {
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t n4 = D.H * D.d, p0 = sq.verify_pos;
    // mRoPE: the verify's rows (kept and rejected alike) are already in the history; the block keys of a
    // block completed within the kept rows replay with its first row's triple / the scalar base + delta -
    // exactly like the verify's own forward did.
    prepare_positions(sq, p0, rows);
    broken_ = true;  // until the state is whole again
    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen4ExpModel::Layer &l = m_.layer(i);
        const size_t li = (size_t)i;
        if (!D.is_attention[li]) {
            // GDN: the verify wrote its states to the spare set, so sq.conv_state / sq.rec_state are still the ones
            // before it - replay rows [0, rows) on them in place, with the verify's own path (Auto: per token), so
            // the state is exactly the verify's after that row (split invariance).
            kernels::gdn_front(sq.verify_in[li].get(),
                               {rows, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                               l.conv_w, sq.conv_state[li].get(), sq.conv_state[li].get(), false, l.A_log, l.dt_bias,
                               qkv_.get(), beta_.get(), g_.get(), act, stream_);
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {rows, D.gk, D.gv, D.conv_c}, sq.rec_state[li].get(),
                                sq.rec_state[li].get(), false, core_.get(), act, stream_, kernels::DeltaRulePath::Auto);
        } else {
            // Attention: the tail (and any block completed within the kept rows - same keys the verify wrote) from
            // the tail before the verify, which it only read. Same block-window triples as the verify's forward.
            kernels::qsa_block_keys(sq.verify_in[li].get(), D.idx_d, rows, p0,
                                    p0 % kernels::kQsaBlock ? sq.tail[sq.verify_tail_cur][li].get() : nullptr,
                                    sq.tail[1 - sq.verify_tail_cur][li].get(), D.idx_d, l.idx_k_norm, D.eps,
                                    m_.inv_freq(), D.rot, sq.block_keys[li].get(), cap_blocks_, act, stream_, m_.rope_scale(),
                                    mk_block_);
            // 8-bit K/V: the groups completed within the kept rows and the tail, from the verify's saved rows and
            // the tail before it (its own commit wrote the other tail buffer, which this overwrites).
            if (kv_format_ == KvFormat::Q8) {
                const int64_t W = D.hkv * D.hd;
                const uint8_t *rows_kv = sq.verify_in[li].get() + (size_t)(kMaxLogits * D.idx_d) * e;
                kernels::kv8_commit(rows_kv, rows_kv + (size_t)W * e, 2 * W, rows, p0,
                                    p0 % kernels::kKv8Group ? sq.kv8_tail[sq.verify_tail_cur][li].get() : nullptr,
                                    sq.kv8_tail[1 - sq.verify_tail_cur][li].get(), sq.kv8_store[li].get(), kv8_cap_, W, act,
                                    kv8_err_.get(), stream_);
            }
        }
    }
    if (D.ple_layer >= 0) {
        // PLE: the history window held [S state rows][T verify rows]; a multi-run verify lays its states' windows
        // end to end in ple_hist_buf_ (the batched forward's order), so sq.verify_ple_wbase is this state's window
        // start (0 for a single-sequence verify). The state after row rows - 1 is its window rows [rows, rows + S).
        const int64_t S = (D.ple_taps - 1) * D.ple_dil;
        STRIX_HIP_CHECK(hipMemcpyAsync(sq.ple_state.get(),
                                       ple_hist_buf_.get() + (size_t)((sq.verify_ple_wbase + rows) * n4) * e,
                                       (size_t)(S * n4) * e, hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: PLE state");
    }
    if (mtp_) {
        // The MTP layer's projections of this state's rows are in proj_ at sq.verify_row_base (0 for a
        // single-sequence verify, the run's row base for a batched one); its streams are still in x_.
        const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
        const int64_t rbase = sq.verify_row_base;
        kernels::qsa_block_keys(proj_.get() + (size_t)(rbase * D.astride + D.idx_col + D.idx_h * D.idx_d) * e,
                                D.astride, rows, p0,
                                p0 % kernels::kQsaBlock ? sq.tail_mtp[sq.verify_mtp_tail_cur].get() : nullptr,
                                sq.tail_mtp[1 - sq.verify_mtp_tail_cur].get(), D.idx_d, ml.idx_k_norm, D.eps, m_.inv_freq(),
                                D.rot, sq.block_keys_mtp.get(), cap_blocks_, act, stream_, m_.rope_scale(), mk_block_);
        STRIX_HIP_CHECK(hipMemcpyAsync(sq.mtp_prev.get(), x_.get() + (size_t)((rbase + rows - 1) * n4) * e,
                                       (size_t)n4 * e, hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::keep_verify_prefix");
    // The n-gram history: the one before the verify, advanced over the kept ids.
    PleHistory h = sq.verify_hist;
    if (D.ple_layer >= 0) {
        std::vector<int64_t> tok(sq.verify_ids.begin(), sq.verify_ids.begin() + rows), out((size_t)(rows * 16));
        ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), rows, h, out.data());
    }
    sq.ple_hist = h;
    sq.pos = p0 + rows;
    // mRoPE: the verify's rows past the kept prefix never happened - truncate the history (the verify's
    // forward appended all of them, so it reaches past sq.pos); the rejected draft's image rows would
    // otherwise sit at sequence indices the next forward overwrites with different tokens (and triples).
    if (!sq.pos3_host.empty()) {
        const int64_t keep_len = sq.pos - sq.pos3_start;
        STRIX_CHECK(keep_len >= 0 && keep_len * 3 <= (int64_t)sq.pos3_host.size(),
                    "Qwen4ExpSession::keep_verify_prefix: keeping ", sq.pos, " sequence rows from a history of [",
                    sq.pos3_start, ", ", sq.pos3Hend(), ") - internal mismatch");
        sq.pos3_host.resize((size_t)(keep_len * 3));
    }
    sq.tail_cur = 1 - sq.verify_tail_cur, sq.mtp_tail_cur = 1 - sq.verify_mtp_tail_cur;
    sq.mtp_chain_step = -1;
    sq.rewind_to.push_back(sq.pos);  // the verify's rows past sq.pos get overwritten: snapshots beyond it no longer match
    sq.verify_pending = false, sq.verify_rows = 0, sq.verify_carry = 0;
    broken_ = false;
}

void Qwen4ExpSession::set_mtp_vocab(int64_t n) {
    STRIX_CHECK(n >= 1 && n <= m_.dims().vocab, "Qwen4ExpSession::set_mtp_vocab: ", n, ", expected 1..",
                m_.dims().vocab);
    mtp_vocab_ = n;
}

// The MTP head's input rows at row_base.. of the forward's row buffers: ids_[row_base + t] with the streams
// after the position before (prev0 for t = 0, x_ row row_base + t - 1 after it) - into the head workspaces'
// rows [row_base, row_base + T), so the head's mix and projections run over the whole batch at once. One
// call's rows must all belong to one state (their prev0 is that state's): the drafts and the catch-up of a
// multi-run batch call it once per run at the run's row base.
void Qwen4ExpSession::mtp_input(int64_t row_base, int64_t T, const void *prev0) {
    STRIX_CHECK(mtp_, "Qwen4ExpSession::mtp_input: MTP is off");
    STRIX_CHECK(T >= 1 && row_base >= 0 && row_base + T <= max_tokens_, "Qwen4ExpSession::mtp_input: rows [",
                row_base, ", ", row_base + T, "), expected inside the workspace's ", max_tokens_);
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
    const size_t de = (size_t)(row_base * D.d) * e, hde = (size_t)(row_base * n4) * e;
    if (m_.is_embed_q8())
        kernels::embedding_lookup(ids_.get() + row_base, T, m_.embed_q8(), mtp_emb_.get() + de, act, emb_err_.get(),
                                  stream_);
    else
        kernels::embedding_lookup(ids_.get() + row_base, T, m_.embed(), D.vocab, D.d, mtp_emb_.get() + de, act,
                                  emb_err_.get(), stream_);
    kernels::rmsnorm_zc(mtp_emb_.get() + de, mtp.pre_fc_norm_embedding, mtp_norm_emb_.get() + de, T, D.d, D.eps, act,
                        stream_);
    STRIX_CHECK(prev0 != nullptr, "Qwen4ExpSession::mtp_input: null previous streams");
    kernels::group_rmsnorm_zc(prev0, mtp.pre_fc_norm_hidden, mtp_norm_hid_.get() + hde, 1, D.H, D.d, D.eps, act,
                              stream_);
    if (T > 1)
        kernels::group_rmsnorm_zc(x_.get() + hde, mtp.pre_fc_norm_hidden, mtp_norm_hid_.get() + hde + (size_t)n4 * e,
                                  T - 1, D.H, D.d, D.eps, act, stream_);
    // Per stream s: x_s = fc_embedding(e) + fc_hidden(h_s) - fc_hidden over the T * H stream rows ([T, H, d] is
    // [T * H, d]), then fc_embedding's row added to each of its token's streams (an injection with weight 1).
    lin(mtp_norm_emb_.get() + de, mtp.fc_embedding, mtp_proj_emb_.get() + de, T);
    lin(mtp_norm_hid_.get() + hde, mtp.fc_hidden, mtp_x_.get() + hde, T * D.H);
    kernels::hc_inject(mtp_x_.get() + hde, mtp_ones_.get() + row_base * D.H, mtp_proj_emb_.get() + de, T, D.H, D.d,
                       act, stream_);
    ++mtp_x_version_;  // the head's rows were rewritten (the draft chain's interruption stamp, see the field)
}

std::vector<float> Qwen4ExpSession::forward_mtp(int32_t token_id, int64_t step) {
    return forward_mtp_seq(0, token_id, step);
}

std::vector<float> Qwen4ExpSession::forward_mtp_seq(int seq, int32_t token_id, int64_t step) {
    run_mtp(seq, token_id, step);
    std::vector<float> logits = read_back((size_t)mtp_vocab_, " MTP");
    mtp_done(seq, step);
    return logits;
}

Qwen4ExpSession::MtpTop2 Qwen4ExpSession::forward_mtp_top2(int32_t token_id, int64_t step) {
    return forward_mtp_top2_seq(0, token_id, step);
}

Qwen4ExpSession::MtpTop2 Qwen4ExpSession::forward_mtp_top2_seq(int seq, int32_t token_id, int64_t step) {
    run_mtp(seq, token_id, step);
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
    mtp_done(seq, step);
    return {p.best, p.best_v, p.second_v, p.nan != 0};
}

void Qwen4ExpSession::mtp_done(int seq, int64_t step) {
    broken_ = false;
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    sq.mtp_chain_step = step, sq.mtp_chain_pos = sq.pos;
    sq.mtp_chain_xver = mtp_x_version_;  // this step's streams are in mtp_x_ up to this version (run_mtp checks)
}

void Qwen4ExpSession::run_mtp(int seq, int32_t token_id, int64_t step) {
    const char *fn = "Qwen4ExpSession::forward_mtp";
    STRIX_TRACE_RANGE("draft " + std::to_string(step));
    STRIX_CHECK(mtp_, fn, ": the session was made without MTP");
    STRIX_CHECK(seq >= 0 && seq < n_slots(), fn, ": seq ", seq, ", expected 0..", n_slots() - 1);
    Qwen4ExpSeqState &sq = *seqs_[(size_t)seq];
    STRIX_CHECK(!broken_, fn, ": an earlier call failed midway; reset() the session first");
    const Qwen4ExpDims &D = m_.dims();
    STRIX_CHECK(token_id >= 0 && token_id < D.vocab, fn, ": token ", token_id, " outside [0, ", D.vocab, ")");
    STRIX_CHECK(step >= 0, fn, ": step ", step, ", expected >= 0");
    const int64_t pending = pending_verify_seq();
    STRIX_CHECK(pending < 0, fn, ": a forward_verify on seq ", pending,
                " awaits keep_verify / drop_verify - resolve the pending batch before drafting");
    // The batched MTP cycle's ordering (design "MTP cycle in the batched world"): a slot drafts its whole
    // chain before the next slot's starts (the batched verify then resolves them together). The draft's
    // streams live in the shared mtp_x_ (swapped against the state's chain buffer), so the ONE thing that
    // must not happen is a chain left half-written while another draft (or forward) overwrites mtp_x_: the
    // version stamp below catches exactly that. A finished, not-yet-verified chain does not block another
    // slot's fresh chain (its last step's streams are not read again - the trunk verify re-derives them),
    // and no verify may be pending (checked above - the shared workspaces are the verify's).
    STRIX_CHECK(step == 0 || (sq.mtp_chain_pos == sq.pos && sq.mtp_chain_step == step - 1), fn, ": step ",
                step, " doesn't continue a chain (last step ", sq.mtp_chain_step, " at position ", sq.mtp_chain_pos,
                ", now at ", sq.pos, ")");
    if (step > 0)
        STRIX_CHECK(sq.mtp_chain_xver == mtp_x_version_, fn, ": seq ", seq, "'s draft chain (step ", step,
                    ") was interrupted - mtp_x_ was written ", mtp_x_version_ - (uint64_t)sq.mtp_chain_xver,
                    " time(s) since its step ", step - 1, " (another draft or a forward ran mid-chain) - finish"
                    " a chain before drafting another slot's");
    const int64_t p = sq.pos + step;  // this row's sequence position (cache row, dense checks; rope angle below)
    STRIX_CHECK(p < capacity_, fn, ": position ", p, " is at the capacity ", capacity_);
    STRIX_CHECK(sq.pending3.empty(), fn, ": set_positions holds ", sq.pending3.size() / 3,
                " interleaved rows for the next forward - the draft runs at pos() with the trunk's positions; "
                "forward them first");
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const int64_t T = 1, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    // mRoPE: the draft row continues the (scalar) text span, so its angle sits at rope_pos(p) = p +
    // rope_delta; the indexer block window may reach an image tail, so its block keys use the history's
    // triples. History rows past p can't exist (no forward ran since the last keep / drop).
    prepare_positions(sq, p, T);
    const int64_t rope_p = mk_jobs_.pos3 == nullptr ? p + sq.rope_delta : p;
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
    sq.mtp_chain_step = -1;
    // Indexer tails: step 0 continues the committed tail into the spare buffer (not flipped); chain steps continue
    // from the step before into their own pair.
    const void *tail_in = step == 0 ? sq.tail_mtp[sq.mtp_tail_cur].get()
                          : step == 1 ? sq.tail_mtp[1 - sq.mtp_tail_cur].get()
                                      : sq.mtp_chain_tail[step % 2].get();
    void *tail_out = step == 0 ? sq.tail_mtp[1 - sq.mtp_tail_cur].get() : sq.mtp_chain_tail[(step + 1) % 2].get();
    if (step > 0) {  // the step before's output streams, kept by swapping buffers (mtp_input overwrites mtp_x_)
        STRIX_CHECK(sq.mtp_chain_prev.size() == mtp_x_.size(), "Qwen4ExpSession: MTP chain buffers ", sq.mtp_chain_prev.size(),
                    " / ", mtp_x_.size(), " bytes, expected equal (they're swapped)");
        std::swap(mtp_x_, sq.mtp_chain_prev);
    }
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), &token_id, 4, hipMemcpyHostToDevice, stream_), "uploading the MTP token id");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    mtp_input(0, T, step == 0 ? sq.mtp_prev.get() : sq.mtp_chain_prev.get());

    // The layer, as a trunk full-attention layer at position p. Step 0's KV row and block keys are the ones the next
    // forward's catch-up writes again from the same inputs; a chain step's, until the position runs, its guess.
    hc_mix(l.hc_attn, true);
    uint8_t *kc = sq.k_cache_mtp.get(), *vc = sq.v_cache_mtp.get();
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
                            D.rot, sq.block_keys_mtp.get(), cap_blocks_, act, stream_, m_.rope_scale(), mk_block_);
    const kernels::AttentionShape as{T, p, D.hq, D.hkv, hd, D.astride, 2 * hd};
    const float scale = 1.0f / std::sqrt((float)hd);
    // The head's own layer, as a trunk full-attention layer: the same dispatch, including the matrix-unit
    // attention from one row (kWmmaAttnMinTokens; the trunk's switch, kernels/attention.hpp). The head's K / V
    // stay BF16 (the 8-bit head cache is the kv-cache-8bit follow-up). Summation order changes -> draft ids and
    // acceptance move - measured as such (docs "MTP head attention on the matrix units").
    const bool wmma_attn = prefill_math_ == PrefillMath::WmmaBf16 && act == Act::BF16 && T >= kWmmaAttnMinTokens;
    if (!qsa_layer) {
        if (wmma_attn)
            kernels::attention_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(),
                                    attn_ws_.get(), attn_ws_bytes_, act, stream_, precision_.attention);
        else
            kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(), attn_ws_.get(),
                               attn_ws_bytes_, act, stream_);
    } else {
        const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
        void *iq = at(proj_, D.idx_col);  // normed + roped by the qk_norm_rope_jobs above
        kernels::qsa_scores(iq, D.astride, D.idx_d, T, p, D.idx_h, D.idx_d, sq.block_keys_mtp.get(), cap_blocks_,
                            qsa_scores_.get(), cap_blocks_, act, stream_);
        kernels::qsa_topk(qsa_scores_.get(), cap_blocks_, T, p, qsa_k, qsa_sel_.get(), qsa_nsel_.get(),
                          qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
        if (wmma_attn)
            kernels::attention_gathered_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                              qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                              attn_ws_bytes_, attn_err_.get(), act, stream_, precision_.attention);
        else
            kernels::attention_gathered(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                        qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(), attn_ws_bytes_,
                                        attn_err_.get(), act, stream_);
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
