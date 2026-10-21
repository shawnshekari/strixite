#include "kernels/rope.hpp"

#include "common/hip_check.hpp"
#include "kernels/rope_math.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxPos = 1ll << 24;  // positions are exact in FP32 below this (as in transformers)

using rope_math::bf16_bits_to_f32;
using rope_math::f32_to_bf16_bits;

__device__ inline float load(const float *p, int64_t i) { return p[i]; }
__device__ inline float load(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// One thread per (token, head, pair). BF16 rounds where transformers' BF16 run does.
template <typename T, bool kBF16>
__global__ void __launch_bounds__(kThreads) rope_kernel(T *x, int64_t T_count, int64_t heads, int64_t half,
                                                        int64_t token_stride, int64_t head_stride, int64_t pos0,
                                                        const float *inv_freq, float cos_sin_scale) {
    const int64_t th_total = T_count * heads;
    for (int64_t th = blockIdx.y; th < th_total; th += gridDim.y) {
        const int64_t t = th / heads;
        const int64_t h = th % heads;
        const int64_t pos = pos0 + t;
        const int64_t th_base = t * token_stride + h * head_stride;
        for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < half;
             i += (int64_t)gridDim.x * blockDim.x) {
            float c, s;
            rope_math::cos_sin<kBF16>(pos, inv_freq[i], cos_sin_scale, c, s);
            const int64_t base = th_base + i;
            float y0, y1;
            rope_math::rotate<kBF16>(load(x, base), load(x, base + half), c, s, y0, y1);
            store(x, base, y0);
            store(x, base + half, y1);
        }
    }
}

}  // namespace

std::vector<float> rope_inv_freq(float theta, int64_t rot_dim) {
    STRIX_CHECK(std::isfinite(theta) && theta > 1.0f, "rope_inv_freq: theta = ", theta, ", expected finite and > 1");
    STRIX_CHECK(rot_dim >= 2 && rot_dim % 2 == 0 && rot_dim <= 4096, "rope_inv_freq: rot_dim = ", rot_dim,
                ", expected even, 2..4096");
    std::vector<float> f((size_t)(rot_dim / 2));
    for (int64_t j = 0; j < rot_dim / 2; ++j)
        f[(size_t)j] = 1.0f / std::pow(theta, (float)(2 * j) / (float)rot_dim);  // FP32 pow + divide, as torch
    return f;
}

YarnRope rope_yarn(float theta, int64_t rot_dim, float factor, int64_t original_max_pos) {
    STRIX_CHECK(std::isfinite(factor) && factor > 1.0f && factor <= 64.0f, "rope_yarn: factor = ", factor,
                ", expected finite, > 1 (1 is plain RoPE: use rope_inv_freq) and <= 64");
    STRIX_CHECK(original_max_pos >= 64 && original_max_pos <= (1ll << 24), "rope_yarn: original_max_pos = ",
                original_max_pos, ", expected 64..2^24");
    (void)rope_inv_freq(theta, rot_dim);  // the same theta / rot_dim checks as plain RoPE
    const int64_t pairs = rot_dim / 2;
    // The pair index (fractional) at which a pair turns `turns` times over the original positions, in double as
    // transformers computes it: L * theta^(-2j/rot_dim) / (2 pi) = turns  ->  j = rot_dim ln(L / (2 pi turns)) / (2 ln theta).
    const double kPi = 3.14159265358979323846;
    auto pair_at_turns = [&](double turns) {
        return (double)rot_dim * std::log((double)original_max_pos / (turns * 2.0 * kPi)) / (2.0 * std::log((double)theta));
    };
    const double beta_fast = 32.0, beta_slow = 1.0;  // the paper's (and transformers') defaults
    // Ramp ends, rounded outward to whole pairs and clamped to the table.
    const double low = std::max(std::floor(pair_at_turns(beta_fast)), 0.0);
    double high = std::min(std::ceil(pair_at_turns(beta_slow)), (double)(rot_dim - 1));
    if (low == high) high += 0.001;  // as transformers: no zero-width ramp
    STRIX_CHECK(high > low, "rope_yarn: ramp [", low, ", ", high, "] is empty - original_max_pos ", original_max_pos,
                " is too short for theta ", theta, " (no pair turns between ", beta_slow, " and ", beta_fast, " times)");
    YarnRope r;
    r.inv_freq.resize((size_t)pairs);
    const float ramp_den = (float)(high - low);
    for (int64_t j = 0; j < pairs; ++j) {
        // One FP32 operation per statement (no fused multiply-add), in transformers' order.
        const float power = std::pow(theta, (float)(2 * j) / (float)rot_dim);  // as rope_inv_freq
        const float extrapolated = 1.0f / power;
        const float scaled_power = factor * power;
        const float interpolated = 1.0f / scaled_power;
        const float ramp_num = (float)j - (float)low;
        float ramp = ramp_num / ramp_den;
        ramp = std::min(std::max(ramp, 0.0f), 1.0f);
        const float keep_weight = 1.0f - ramp;           // 1: keep the original frequency
        const float slow_weight = 1.0f - keep_weight;    // 1: divide it by the factor
        const float slow_part = interpolated * slow_weight;
        const float keep_part = extrapolated * keep_weight;
        r.inv_freq[(size_t)j] = slow_part + keep_part;
    }
    r.cos_sin_scale = (float)(0.1 * std::log((double)factor) + 1.0);
    return r;
}

void rope(void *x, int64_t T, int64_t heads, int64_t token_stride, int64_t head_stride, int64_t rot_dim, int64_t pos0,
          const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale) {
    STRIX_CHECK(x != nullptr && inv_freq != nullptr, "rope: null pointer (x=", x, ", inv_freq=", (const void *)inv_freq,
                ")");
    STRIX_CHECK(T >= 1 && heads >= 1, "rope: T = ", T, ", heads = ", heads, ", both must be >= 1");
    STRIX_CHECK(rot_dim >= 2 && rot_dim % 2 == 0, "rope: rot_dim = ", rot_dim, ", expected even and >= 2");
    STRIX_CHECK(heads == 1 || head_stride >= rot_dim, "rope: head_stride = ", head_stride, " < rot_dim = ", rot_dim,
                " - heads would overlap");
    STRIX_CHECK(head_stride >= 0 && token_stride >= (heads - 1) * head_stride + rot_dim, "rope: token_stride = ",
                token_stride, " too small for ", heads, " heads at stride ", head_stride, " + rot_dim ", rot_dim,
                " - tokens would overlap");
    STRIX_CHECK(pos0 >= 0 && pos0 + T <= kMaxPos, "rope: positions [", pos0, ", ", pos0 + T, ") outside [0, ", kMaxPos,
                ")");
    STRIX_CHECK(T <= ((int64_t)1 << 40) / (heads * rot_dim), "rope: T * heads * rot_dim too large (T = ", T, ")");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "rope: unsupported activation dtype ", (int)act);
    STRIX_CHECK(std::isfinite(cos_sin_scale) && cos_sin_scale >= 1.0f && cos_sin_scale <= 2.0f,
                "rope: cos_sin_scale = ", cos_sin_scale, ", expected finite, 1..2 (YaRN's attention factor)");
    const int64_t es = act == Act::F32 ? 4 : 2;
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % es == 0, "rope: x ", x, " not aligned to its ", es, "-byte dtype");
    const int64_t half = rot_dim / 2;
    const int64_t blocks_x = (half + kThreads - 1) / kThreads;
    const int64_t th_total = T * heads;
    dim3 grid((unsigned)blocks_x, (unsigned)(th_total < 65535 ? th_total : 65535)), block(kThreads);
    if (act == Act::F32)
        rope_kernel<float, false><<<grid, block, 0, stream>>>(static_cast<float *>(x), T, heads, half, token_stride,
                                                              head_stride, pos0, inv_freq, cos_sin_scale);
    else
        rope_kernel<uint16_t, true><<<grid, block, 0, stream>>>(static_cast<uint16_t *>(x), T, heads, half,
                                                                token_stride, head_stride, pos0, inv_freq, cos_sin_scale);
    STRIX_HIP_CHECK_LAUNCH("rope ", act_name(act), " T=", T, " heads=", heads, " rot_dim=", rot_dim, " pos0=", pos0);
}

}  // namespace strix::kernels
