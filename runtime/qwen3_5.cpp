#include "runtime/qwen3_5.hpp"

#include "common/hip_check.hpp"
#include "kernels/attention.hpp"
#include "kernels/gdn.hpp"
#include "kernels/linear.hpp"
#include "kernels/residual.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strix {

using kernels::Act;

namespace {

const std::string kP = "model.language_model.";
constexpr int64_t kHD = kernels::kGdnHeadDim;
constexpr int64_t kConvW = kernels::kGdnConvWidth;

size_t es(Act a) { return a == Act::F32 ? 4 : 2; }

const TensorInfo &get(const SafetensorsFile &f, const std::string &name, Dtype dtype, std::vector<int64_t> shape) {
    const TensorInfo &t = f.get(name);
    STRIX_CHECK(t.dtype == dtype, "Qwen35Model: '", name, "' in '", f.path(), "' is ", dtype_name(t.dtype),
                ", expected ", dtype_name(dtype));
    STRIX_CHECK(t.shape == shape, "Qwen35Model: '", name, "' in '", f.path(), "' has shape ", list_str(t.shape),
                ", expected ", list_str(shape));
    return t;
}

// Row-concatenated BF16 weights, all [*, K], uploaded as stored (formats/merge_plan's groups).
DeviceBuffer<uint16_t> bf16_rows(const SafetensorsFile &f, const std::vector<std::string> &names, int64_t K,
                                 std::vector<int64_t> rows) {
    STRIX_CHECK(names.size() == rows.size() && !names.empty(), "bf16_rows: ", names.size(), " names, ", rows.size(),
                " row counts");
    std::vector<uint16_t> bits;
    for (size_t i = 0; i < names.size(); ++i) {
        const TensorInfo &t = get(f, names[i], Dtype::BF16, {rows[i], K});
        const size_t at = bits.size();
        bits.resize(at + (size_t)t.numel());
        std::memcpy(bits.data() + at, f.data(t), t.byte_length);
    }
    return DeviceBuffer<uint16_t>::from_host(bits, names[0] + (names.size() > 1 ? " (merged)" : ""));
}

// A BF16 or F32 checkpoint tensor of n elements (any shape), as FP32 on the device.
DeviceBuffer<float> f32(const SafetensorsFile &f, const std::string &name, int64_t n) {
    const TensorInfo &t = f.get(name);
    STRIX_CHECK(t.numel() == n, "Qwen35Model: '", name, "' has ", t.numel(), " elements (shape ", list_str(t.shape),
                "), expected ", n);
    std::vector<float> v((size_t)n);
    if (t.dtype == Dtype::F32) std::memcpy(v.data(), f.data(t), t.byte_length);
    else {
        STRIX_CHECK(t.dtype == Dtype::BF16, "Qwen35Model: '", name, "' is ", dtype_name(t.dtype), ", expected BF16 or F32");
        const auto *b = static_cast<const uint16_t *>(f.data(t));
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t u = (uint32_t)b[i] << 16;
            std::memcpy(&v[(size_t)i], &u, 4);
        }
    }
    return DeviceBuffer<float>::from_host(v, name);
}

}  // namespace

Qwen35Model::Qwen35Model(const std::string &path, Act act) : act_(act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "Qwen35Model: unsupported activation dtype ", (int)act);
    SafetensorsFile f(path);
    Qwen35Dims &D = dims_;
    const TensorInfo &emb = f.get(kP + "embed_tokens.weight");
    STRIX_CHECK(emb.dtype == Dtype::BF16 && emb.shape.size() == 2, "Qwen35Model: embed_tokens is ",
                dtype_name(emb.dtype), " ", list_str(emb.shape), ", expected BF16 [vocab, d]");
    D.vocab = emb.shape[0];
    D.d = emb.shape[1];
    while (f.find(kP + "layers." + std::to_string(D.layers) + ".input_layernorm.weight")) ++D.layers;
    STRIX_CHECK(D.layers >= 1, "Qwen35Model: no decoder layers in '", path, "'");
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kP + "layers." + std::to_string(i) + ".";
        const bool attn = f.find(L + "self_attn.q_proj.weight") != nullptr;
        STRIX_CHECK(attn != (f.find(L + "linear_attn.A_log") != nullptr), "Qwen35Model: layer ", i,
                    " must have exactly one of self_attn / linear_attn");
        D.is_attention.push_back(attn);
    }
    D.inter = f.get(kP + "layers.0.mlp.gate_proj.weight").shape.at(0);
    // Shapes of the first layer of each kind; every other layer must match (checked while loading).
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kP + "layers." + std::to_string(i) + ".";
        if (D.is_attention[(size_t)i] && D.hq == 0) {
            D.head_dim = f.get(L + "self_attn.q_norm.weight").shape.at(0);
            STRIX_CHECK(D.head_dim > 0 && D.head_dim % 4 == 0, "Qwen35Model: head_dim ", D.head_dim);
            D.hq = f.get(L + "self_attn.q_proj.weight").shape.at(0) / (2 * D.head_dim);
            D.hkv = f.get(L + "self_attn.k_proj.weight").shape.at(0) / D.head_dim;
            D.rot_dim = D.head_dim / 4;  // config.json partial_rotary_factor 0.25
        }
        if (!D.is_attention[(size_t)i] && D.gv == 0) {
            D.gv = f.get(L + "linear_attn.A_log").shape.at(0);
            D.conv_c = f.get(L + "linear_attn.in_proj_qkv.weight").shape.at(0);
            STRIX_CHECK(f.get(L + "linear_attn.norm.weight").numel() == kHD, "Qwen35Model: GDN value head dim ",
                        f.get(L + "linear_attn.norm.weight").numel(), ", the kernels are built for ", kHD);
            STRIX_CHECK((D.conv_c - D.gv * kHD) > 0 && (D.conv_c - D.gv * kHD) % (2 * kHD) == 0,
                        "Qwen35Model: in_proj_qkv rows ", D.conv_c, " don't split into q|k|v of ", kHD, "-wide heads with ",
                        D.gv, " value heads");
            D.gk = (D.conv_c - D.gv * kHD) / (2 * kHD);
        }
    }
    embed_ = DeviceBuffer<uint16_t>::from_host([&] {
        std::vector<uint16_t> b((size_t)emb.numel());
        std::memcpy(b.data(), f.data(emb), emb.byte_length);
        return b;
    }(), "embed_tokens");
    final_norm_ = f32(f, kP + "norm.weight", D.d);
    if (D.hq) inv_freq_ = DeviceBuffer<float>::from_host(kernels::rope_inv_freq(D.rope_theta, D.rot_dim), "inv_freq");

    const int64_t Z = D.gv * kHD;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kP + "layers." + std::to_string(i) + ".";
        Layer l;
        l.in_norm = f32(f, L + "input_layernorm.weight", D.d);
        l.post_norm = f32(f, L + "post_attention_layernorm.weight", D.d);
        l.gate_up = bf16_rows(f, {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}, D.d, {D.inter, D.inter});
        l.down = bf16_rows(f, {L + "mlp.down_proj.weight"}, D.inter, {D.d});
        if (D.is_attention[(size_t)i]) {
            const std::string A = L + "self_attn.";
            l.qkv = bf16_rows(f, {A + "q_proj.weight", A + "k_proj.weight", A + "v_proj.weight"}, D.d,
                              {D.hq * 2 * D.head_dim, D.hkv * D.head_dim, D.hkv * D.head_dim});
            l.o_proj = bf16_rows(f, {A + "o_proj.weight"}, D.hq * D.head_dim, {D.d});
            l.q_norm = f32(f, A + "q_norm.weight", D.head_dim);
            l.k_norm = f32(f, A + "k_norm.weight", D.head_dim);
        } else {
            const std::string G = L + "linear_attn.";
            l.in_proj = bf16_rows(f, {G + "in_proj_qkv.weight", G + "in_proj_z.weight", G + "in_proj_b.weight",
                                      G + "in_proj_a.weight"},
                                  D.d, {D.conv_c, Z, D.gv, D.gv});
            l.out_proj = bf16_rows(f, {G + "out_proj.weight"}, Z, {D.d});
            get(f, G + "conv1d.weight", Dtype::BF16, {D.conv_c, 1, kConvW});
            l.conv_w = f32(f, G + "conv1d.weight", D.conv_c * kConvW);
            l.A_log = f32(f, G + "A_log", D.gv);
            l.dt_bias = f32(f, G + "dt_bias", D.gv);
            l.gdn_norm = f32(f, G + "norm.weight", kHD);
        }
        layers_.push_back(std::move(l));
    }
}

const Qwen35Model::Layer &Qwen35Model::layer(int64_t i) const {
    STRIX_CHECK(i >= 0 && i < (int64_t)layers_.size(), "Qwen35Model::layer: index ", i, " outside [0, ",
                layers_.size(), ")");
    return layers_[(size_t)i];
}

Qwen35Session::Qwen35Session(const Qwen35Model &model, int64_t capacity, int64_t max_tokens)
    : m_(model), capacity_(capacity), max_tokens_(max_tokens) {
    const Qwen35Dims &D = m_.dims();
    STRIX_CHECK(capacity >= 1 && capacity <= (1ll << 24), "Qwen35Session: capacity = ", capacity, ", expected 1..2^24");
    STRIX_CHECK(max_tokens >= 1 && max_tokens <= capacity && max_tokens <= 65535, "Qwen35Session: max_tokens = ",
                max_tokens, ", expected 1..min(capacity = ", capacity, ", 65535)");
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "Qwen35Session: creating its stream");
    const size_t e = es(m_.act());
    const int64_t M = max_tokens, Z = D.gv * kHD;
    const int64_t gstride = D.conv_c + Z + 2 * D.gv, astride = D.hq * 2 * D.head_dim + 2 * D.hkv * D.head_dim;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "layer " + std::to_string(i);
        if (D.is_attention[(size_t)i]) {
            k_cache_.emplace_back((size_t)(capacity * D.hkv * D.head_dim) * e, "K cache " + tag);
            v_cache_.emplace_back((size_t)(capacity * D.hkv * D.head_dim) * e, "V cache " + tag);
            conv_state_.emplace_back();
            rec_state_.emplace_back();
        } else {
            k_cache_.emplace_back();
            v_cache_.emplace_back();
            conv_state_.emplace_back((size_t)((kConvW - 1) * D.conv_c) * e, "conv state " + tag);
            rec_state_.emplace_back((size_t)(D.gv * kHD * kHD), "recurrent state " + tag);
        }
    }
    auto act_buf = [&](int64_t per_token, const char *name) {
        return DeviceBuffer<uint8_t>((size_t)(M * std::max<int64_t>(per_token, 1)) * e, name);
    };
    ids_ = DeviceBuffer<int32_t>((size_t)M, "token ids");
    x_ = act_buf(D.d, "residual");
    n_ = act_buf(D.d, "normed");
    proj_ = act_buf(std::max(D.gv ? gstride : 0, D.hq ? astride : 0), "projection");
    qkv_ = act_buf(D.conv_c, "gdn q|k|v");
    core_ = act_buf(std::max(Z, D.hq * D.head_dim), "mixer core");
    z_ = act_buf(Z, "gdn z");
    gnorm_ = act_buf(Z, "gdn normed");
    mix_ = act_buf(D.d, "mixer/mlp out");
    gu_ = act_buf(2 * D.inter, "gate|up");
    h_ = act_buf(D.inter, "swiglu");
    logits_ = DeviceBuffer<uint8_t>((size_t)(std::min(M, kMaxLogits) * D.vocab) * e, "logits");
    beta_ = DeviceBuffer<float>((size_t)(M * std::max<int64_t>(D.gv, 1)), "gdn beta");
    g_ = DeviceBuffer<float>((size_t)(M * std::max<int64_t>(D.gv, 1)), "gdn g");
    if (D.hq) {
        for (int64_t T = 1; T <= M; ++T)
            attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_workspace_bytes(
                                                          {T, capacity - T, D.hq, D.hkv, D.head_dim, astride, 2 * D.head_dim}));
        attn_ws_ = DeviceBuffer<float>(attn_ws_bytes_ / 4 + 1, "attention workspace");
    }
    emb_err_ = DeviceBuffer<kernels::EmbeddingError>(1, "embedding error");
}

Qwen35Session::~Qwen35Session() {
    if (stream_) (void)hipStreamDestroy(stream_);
}

void Qwen35Session::reset() {
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen35Session::reset");
    pos_ = 0;  // states are reset by the first call (reset_state), the KV cache is overwritten
    broken_ = false;
}

std::vector<float> Qwen35Session::forward(const std::vector<int32_t> &ids, int64_t n_logits, const ForwardProbe &probe) {
    STRIX_CHECK(!broken_, "Qwen35Session::forward: an earlier call failed midway; reset() the session first");
    const Qwen35Dims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t T = (int64_t)ids.size();
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen35Session::forward: ", T, " tokens, expected 1..", max_tokens_);
    STRIX_CHECK(pos_ + T <= capacity_, "Qwen35Session::forward: positions [", pos_, ", ", pos_ + T,
                ") exceed the capacity ", capacity_);
    STRIX_CHECK(n_logits >= 0 && n_logits <= std::min(T, kMaxLogits), "Qwen35Session::forward: n_logits = ", n_logits,
                ", expected 0..", std::min(T, kMaxLogits));
    broken_ = true;  // until this call completes
    const int64_t Z = D.gv * kHD, gstride = D.conv_c + Z + 2 * D.gv;
    const int64_t hd = D.head_dim, astride = D.hq * 2 * hd + 2 * D.hkv * hd;
    const bool fresh = pos_ == 0;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    auto show = [&](const std::string &name, const DeviceBuffer<uint8_t> &b, int64_t cols) {
        if (!probe) return;
        STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "before probe '", name, "'");
        probe(name, b.get(), T, cols);
    };
#define SHOW(name, b, cols)                           \
    do {                                              \
        if (__builtin_expect(probe != nullptr, 0))    \
            show(name, b, cols);                      \
    } while (0)

    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), ids.data(), (size_t)T * 4, hipMemcpyHostToDevice, stream_),
                    "uploading ", T, " token ids");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    kernels::embedding_lookup(ids_.get(), T, m_.embed().get(), D.vocab, D.d, x_.get(), act, emb_err_.get(), stream_);
    SHOW("embed", x_, D.d);

    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen35Model::Layer &l = m_.layer(i);
        const std::string L = __builtin_expect(probe != nullptr, 0) ? ("L" + std::to_string(i) + ".") : "";
        kernels::rmsnorm_zc(x_.get(), l.in_norm.get(), n_.get(), T, D.d, D.eps, act, stream_);
        SHOW(L + "in_norm", n_, D.d);
        if (!D.is_attention[(size_t)i]) {
            kernels::linear_bf16w(n_.get(), l.in_proj.get(), proj_.get(), T, gstride, D.d, act, stream_);
            kernels::gdn_front(proj_.get(), {T, D.gk, kHD, D.gv, kHD, gstride, D.conv_c + Z, D.conv_c + Z + D.gv},
                               l.conv_w.get(), conv_state_[(size_t)i].get(), conv_state_[(size_t)i].get(), fresh,
                               l.A_log.get(), l.dt_bias.get(), qkv_.get(), beta_.get(), g_.get(), act, stream_);
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {T, D.gk, D.gv, D.conv_c},
                                rec_state_[(size_t)i].get(), rec_state_[(size_t)i].get(), fresh, core_.get(), act,
                                stream_);
            // rmsnorm_gated reads z as contiguous rows; z sits inside the in_proj rows.
            STRIX_HIP_CHECK(hipMemcpy2DAsync(z_.get(), (size_t)Z * e, at(proj_, D.conv_c), (size_t)gstride * e,
                                             (size_t)Z * e, (size_t)T, hipMemcpyDeviceToDevice, stream_),
                            "z out of the in_proj rows, layer ", i);
            kernels::rmsnorm_gated(core_.get(), z_.get(), l.gdn_norm.get(), gnorm_.get(), T * D.gv, kHD, D.eps,
                                   kernels::GateAct::SiLU, act,
                                   stream_);
            kernels::linear_bf16w(gnorm_.get(), l.out_proj.get(), mix_.get(), T, D.d, Z, act, stream_);
        } else {
            uint8_t *kc = k_cache_[(size_t)i].get(), *vc = v_cache_[(size_t)i].get();
            kernels::linear_bf16w(n_.get(), l.qkv.get(), proj_.get(), T, astride, D.d, act, stream_);
            kernels::qk_norm_rope(proj_.get(), astride, 2 * hd, proj_.get(), astride, 2 * hd, T, D.hq, hd,
                                  l.q_norm.get(), D.eps, D.rot_dim, pos_, m_.inv_freq().get(), act, stream_);
            kernels::qk_norm_rope(at(proj_, D.hq * 2 * hd), astride, hd, kc + (size_t)(pos_ * D.hkv * hd) * e,
                                  D.hkv * hd, hd, T, D.hkv, hd, l.k_norm.get(), D.eps, D.rot_dim, pos_,
                                  m_.inv_freq().get(), act, stream_);
            STRIX_HIP_CHECK(hipMemcpy2DAsync(vc + (size_t)(pos_ * D.hkv * hd) * e, (size_t)(D.hkv * hd) * e,
                                             at(proj_, D.hq * 2 * hd + D.hkv * hd), (size_t)astride * e,
                                             (size_t)(D.hkv * hd) * e, (size_t)T, hipMemcpyDeviceToDevice, stream_),
                            "v into the cache, layer ", i);
            kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_,
                               {T, pos_, D.hq, D.hkv, hd, astride, 2 * hd}, 1.0f / std::sqrt((float)hd), core_.get(),
                               attn_ws_.get(), attn_ws_bytes_, act, stream_);
            kernels::linear_bf16w(core_.get(), l.o_proj.get(), mix_.get(), T, D.d, D.hq * hd, act, stream_);
        }
        SHOW(L + "mixer", mix_, D.d);
        kernels::residual_add(x_.get(), mix_.get(), T * D.d, act, stream_);
        kernels::rmsnorm_zc(x_.get(), l.post_norm.get(), n_.get(), T, D.d, D.eps, act, stream_);
        SHOW(L + "post_norm", n_, D.d);
        kernels::linear_bf16w(n_.get(), l.gate_up.get(), gu_.get(), T, 2 * D.inter, D.d, act, stream_);
        kernels::swiglu(gu_.get(), h_.get(), T, D.inter, act, stream_);
        kernels::linear_bf16w(h_.get(), l.down.get(), mix_.get(), T, D.d, D.inter, act, stream_);
        SHOW(L + "mlp", mix_, D.d);
        kernels::residual_add(x_.get(), mix_.get(), T * D.d, act, stream_);
        SHOW(L + "out", x_, D.d);
    }
    kernels::rmsnorm_zc(x_.get(), m_.final_norm().get(), n_.get(), T, D.d, D.eps, act, stream_);
    SHOW("final_norm", n_, D.d);
#undef SHOW
    if (n_logits > 0)
        kernels::linear_bf16w(at(n_, (T - n_logits) * D.d), m_.embed().get(), logits_.get(), n_logits, D.vocab, D.d,
                              act, stream_);
    kernels::check_embedding_error(emb_err_.get(), D.vocab, stream_);  // synchronizes the stream

    std::vector<float> logits((size_t)(n_logits * D.vocab));
    if (n_logits > 0) {
        if (act == Act::F32)
            STRIX_HIP_CHECK(hipMemcpy(logits.data(), logits_.get(), logits.size() * 4, hipMemcpyDeviceToHost),
                            "downloading logits");
        else {
            std::vector<uint16_t> bits(logits.size());
            STRIX_HIP_CHECK(hipMemcpy(bits.data(), logits_.get(), bits.size() * 2, hipMemcpyDeviceToHost),
                            "downloading logits");
            for (size_t k = 0; k < bits.size(); ++k) {
                const uint32_t u = (uint32_t)bits[k] << 16;
                std::memcpy(&logits[k], &u, 4);
            }
        }
    }
    pos_ += T;
    broken_ = false;
    return logits;
}

}  // namespace strix
