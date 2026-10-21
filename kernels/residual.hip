#include "kernels/residual.hpp"

#include "common/hip_check.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxElems = 1ll << 40;  // far above any real activation; catches a size mix-up

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ void __launch_bounds__(kThreads) add_f32(float *x, const float *y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] += y[i];
}

__global__ void __launch_bounds__(kThreads) add_f32_vec4(float *x, const float *y, int64_t n4) {
    float4 *x4 = reinterpret_cast<float4 *>(x);
    const float4 *y4 = reinterpret_cast<const float4 *>(y);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t)gridDim.x * blockDim.x) {
        float4 vx = x4[i];
        float4 vy = y4[i];
        x4[i] = make_float4(vx.x + vy.x, vx.y + vy.y, vx.z + vy.z, vx.w + vy.w);
    }
}

__global__ void __launch_bounds__(kThreads) add_bf16(uint16_t *x, const uint16_t *y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] = f32_to_bf16_bits(bf16_bits_to_f32(x[i]) + bf16_bits_to_f32(y[i]));
}

__global__ void __launch_bounds__(kThreads) add_bf16_vec8(uint16_t *x, const uint16_t *y, int64_t n8) {
    uint4 *x4 = reinterpret_cast<uint4 *>(x);
    const uint4 *y4 = reinterpret_cast<const uint4 *>(y);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (int64_t)gridDim.x * blockDim.x) {
        uint4 vx = x4[i];
        uint4 vy = y4[i];
        uint32_t x_words[4] = {vx.x, vx.y, vx.z, vx.w};
        uint32_t y_words[4] = {vy.x, vy.y, vy.z, vy.w};
        uint32_t res_words[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float x0 = bf16_bits_to_f32((uint16_t)x_words[k]);
            float x1 = bf16_bits_to_f32((uint16_t)(x_words[k] >> 16));
            float y0 = bf16_bits_to_f32((uint16_t)y_words[k]);
            float y1 = bf16_bits_to_f32((uint16_t)(y_words[k] >> 16));
            uint16_t r0 = f32_to_bf16_bits(x0 + y0);
            uint16_t r1 = f32_to_bf16_bits(x1 + y1);
            res_words[k] = (uint32_t)r0 | ((uint32_t)r1 << 16);
        }
        x4[i] = make_uint4(res_words[0], res_words[1], res_words[2], res_words[3]);
    }
}

}  // namespace

void residual_add(void *x, const void *y, int64_t n, Act act, hipStream_t stream) {
    STRIX_CHECK(x != nullptr && y != nullptr, "residual_add: null pointer (x=", x, ", y=", y, ")");
    STRIX_CHECK(n >= 1 && n <= kMaxElems, "residual_add: n = ", n, ", expected 1..", kMaxElems);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "residual_add: unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2;
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % es == 0 && reinterpret_cast<uintptr_t>(y) % es == 0,
                "residual_add: x = ", x, ", y = ", y, " not aligned to ", es, " bytes for ", act_name(act));
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(x == y || xb + (uintptr_t)(n * es) <= yb || yb + (uintptr_t)(n * es) <= xb, "residual_add: y [", y,
                ", +", n * es, " B) partly overlaps x [", x, ", +", n * es, " B)");

    if (act == Act::F32) {
        if ((n % 4 == 0) && (xb % 16 == 0) && (yb % 16 == 0)) {
            const int64_t n4 = n / 4;
            const int64_t blocks = (n4 + kThreads - 1) / kThreads;
            dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
            add_f32_vec4<<<grid, block, 0, stream>>>(static_cast<float *>(x), static_cast<const float *>(y), n4);
        } else {
            const int64_t blocks = (n + kThreads - 1) / kThreads;
            dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
            add_f32<<<grid, block, 0, stream>>>(static_cast<float *>(x), static_cast<const float *>(y), n);
        }
    } else {
        if ((n % 8 == 0) && (xb % 16 == 0) && (yb % 16 == 0)) {
            const int64_t n8 = n / 8;
            const int64_t blocks = (n8 + kThreads - 1) / kThreads;
            dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
            add_bf16_vec8<<<grid, block, 0, stream>>>(static_cast<uint16_t *>(x), static_cast<const uint16_t *>(y), n8);
        } else {
            const int64_t blocks = (n + kThreads - 1) / kThreads;
            dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
            add_bf16<<<grid, block, 0, stream>>>(static_cast<uint16_t *>(x), static_cast<const uint16_t *>(y), n);
        }
    }
    STRIX_HIP_CHECK_LAUNCH("residual_add ", act_name(act), " n=", n);
}

}  // namespace strix::kernels
