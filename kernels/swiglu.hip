#include "kernels/swiglu.hpp"

#include "common/hip_check.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxElems = 1ll << 40;  // far above any real rows * I; catches a rows/I mix-up

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }  // same form as rmsnorm_gated

__global__ void __launch_bounds__(kThreads) swiglu_f32(const float *gu, float *h, int64_t rows, int64_t I) {
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < I;
             j += (int64_t)gridDim.x * blockDim.x) {
            h[r * I + j] = silu(gu[r * 2 * I + j]) * gu[r * 2 * I + I + j];
        }
    }
}

__global__ void __launch_bounds__(kThreads)
    swiglu_f32_vec4(const float *gu, float *h, int64_t rows, int64_t I4) {
    const int64_t I = I4 * 4;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const float4 *g_f4 = reinterpret_cast<const float4 *>(gu + r * 2 * I);
        const float4 *u_f4 = reinterpret_cast<const float4 *>(gu + r * 2 * I + I);
        float4 *h_f4 = reinterpret_cast<float4 *>(h + r * I);
        for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < I4;
             j += (int64_t)gridDim.x * blockDim.x) {
            float4 g4 = g_f4[j];
            float4 u4 = u_f4[j];
            float4 res;
            res.x = silu(g4.x) * u4.x;
            res.y = silu(g4.y) * u4.y;
            res.z = silu(g4.z) * u4.z;
            res.w = silu(g4.w) * u4.w;
            h_f4[j] = res;
        }
    }
}

__global__ void __launch_bounds__(kThreads) swiglu_bf16(const uint16_t *gu, uint16_t *h, int64_t rows, int64_t I) {
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < I;
             j += (int64_t)gridDim.x * blockDim.x) {
            const float s = bf16_bits_to_f32(f32_to_bf16_bits(silu(bf16_bits_to_f32(gu[r * 2 * I + j]))));
            h[r * I + j] = f32_to_bf16_bits(s * bf16_bits_to_f32(gu[r * 2 * I + I + j]));
        }
    }
}

__global__ void __launch_bounds__(kThreads)
    swiglu_bf16_vec8(const uint16_t *gu, uint16_t *h, int64_t rows, int64_t I8) {
    const int64_t I = I8 * 8;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const uint4 *g_u4 = reinterpret_cast<const uint4 *>(gu + r * 2 * I);
        const uint4 *u_u4 = reinterpret_cast<const uint4 *>(gu + r * 2 * I + I);
        uint4 *h_u4 = reinterpret_cast<uint4 *>(h + r * I);
        for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < I8;
             j += (int64_t)gridDim.x * blockDim.x) {
            uint4 g4 = g_u4[j];
            uint4 u4 = u_u4[j];
            uint32_t gw[4] = {g4.x, g4.y, g4.z, g4.w};
            uint32_t uw[4] = {u4.x, u4.y, u4.z, u4.w};
            uint32_t hw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float g0 = bf16_bits_to_f32((uint16_t)gw[k]);
                float g1 = bf16_bits_to_f32((uint16_t)(gw[k] >> 16));
                float u0 = bf16_bits_to_f32((uint16_t)uw[k]);
                float u1 = bf16_bits_to_f32((uint16_t)(uw[k] >> 16));
                float s0 = bf16_bits_to_f32(f32_to_bf16_bits(silu(g0)));
                float s1 = bf16_bits_to_f32(f32_to_bf16_bits(silu(g1)));
                uint16_t h0 = f32_to_bf16_bits(s0 * u0);
                uint16_t h1 = f32_to_bf16_bits(s1 * u1);
                hw[k] = (uint32_t)h0 | ((uint32_t)h1 << 16);
            }
            h_u4[j] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
        }
    }
}

}  // namespace

void swiglu(const void *gu, void *h, int64_t rows, int64_t I, Act act, hipStream_t stream) {
    STRIX_CHECK(gu != nullptr && h != nullptr, "swiglu: null pointer (gu=", gu, ", h=", h, ")");
    STRIX_CHECK(rows >= 1 && I >= 1, "swiglu: rows = ", rows, ", I = ", I, ", both must be >= 1");
    STRIX_CHECK(rows <= kMaxElems / I / 2, "swiglu: rows * 2I = ", rows, " * ", 2 * I, " exceeds ", kMaxElems,
                " elements (rows/I swapped?)");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "swiglu: unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2, n = rows * I;
    auto gb = reinterpret_cast<uintptr_t>(gu), hb = reinterpret_cast<uintptr_t>(h);
    STRIX_CHECK(hb + (uintptr_t)(n * es) <= gb || gb + (uintptr_t)(2 * n * es) <= hb, "swiglu: output h [", h, ", +",
                n * es, " B) overlaps input gu [", gu, ", +", 2 * n * es, " B)");

    unsigned grid_y = (unsigned)(rows < 65535 ? rows : 65535);
    dim3 block(kThreads);

    if (act == Act::F32) {
        if ((I % 4 == 0) && (gb % 16 == 0) && (hb % 16 == 0)) {
            int64_t I4 = I / 4;
            unsigned grid_x = (unsigned)((I4 + kThreads - 1) / kThreads);
            swiglu_f32_vec4<<<dim3(grid_x, grid_y), block, 0, stream>>>(
                static_cast<const float *>(gu), static_cast<float *>(h), rows, I4);
        } else {
            unsigned grid_x = (unsigned)((I + kThreads - 1) / kThreads);
            swiglu_f32<<<dim3(grid_x, grid_y), block, 0, stream>>>(
                static_cast<const float *>(gu), static_cast<float *>(h), rows, I);
        }
    } else {
        if ((I % 8 == 0) && (gb % 16 == 0) && (hb % 16 == 0)) {
            int64_t I8 = I / 8;
            unsigned grid_x = (unsigned)((I8 + kThreads - 1) / kThreads);
            swiglu_bf16_vec8<<<dim3(grid_x, grid_y), block, 0, stream>>>(
                static_cast<const uint16_t *>(gu), static_cast<uint16_t *>(h), rows, I8);
        } else {
            unsigned grid_x = (unsigned)((I + kThreads - 1) / kThreads);
            swiglu_bf16<<<dim3(grid_x, grid_y), block, 0, stream>>>(
                static_cast<const uint16_t *>(gu), static_cast<uint16_t *>(h), rows, I);
        }
    }
    STRIX_HIP_CHECK_LAUNCH("swiglu ", act_name(act), " rows=", rows, " I=", I);
}

}  // namespace strix::kernels
