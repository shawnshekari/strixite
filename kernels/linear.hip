#include "kernels/linear.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = 8;  // x rows sharing one pass over a W row
constexpr int64_t kMaxN = 0x7fffffff;
constexpr int64_t kMaxM = 65535ll * kRowsPerBlock;  // grid.y limit
constexpr int64_t kMaxK = 1 << 24;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline float load(const float *p, int64_t i) { return p[i]; }
__device__ inline float load(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// Block (n, row group): each thread accumulates a K-strided slice of W row n
// against up to kRowsPerBlock x rows; LDS tree reduction per row.
// TX/TY: activation types of x and y (the same, or BF16 -> F32 for outputs kept FP32, e.g. the router logits).
template <typename TX, typename TY>
__global__ void __launch_bounds__(kThreads)
    linear_bf16w_kernel(const TX *x, const uint16_t *w, TY *y, int64_t M, int64_t N, int64_t K) {
    __shared__ float red[kRowsPerBlock][kThreads / 32];
    int64_t n = blockIdx.x;
    int64_t m0 = (int64_t)blockIdx.y * kRowsPerBlock;
    int rows = (int)((M - m0) < kRowsPerBlock ? (M - m0) : kRowsPerBlock);
    const uint16_t *wr = w + n * K;
    float acc[kRowsPerBlock];
#pragma unroll
    for (int r = 0; r < kRowsPerBlock; ++r) acc[r] = 0.f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        float wv = bf16_bits_to_f32(wr[k]);
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r)
            if (r < rows) acc[r] = fmaf(load(x, (m0 + r) * K + k), wv, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1) acc[r] += __shfl_down(acc[r], off, warpSize);
    }
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int num_waves = blockDim.x / warpSize;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r) red[r][wave] = acc[r];
    }
    __syncthreads();
    if (wave == 0) {
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r) {
            float v = (lane < num_waves) ? red[r][lane] : 0.f;
#pragma unroll
            for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off, warpSize);
            if (lane == 0) red[r][0] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int r = 0; r < rows; ++r) store(y, (m0 + r) * N + n, red[r][0]);
}

// Decode-oriented path (K % 8 == 0, x and w 16-byte aligned), the linear_q4 design on BF16 weights: KS waves
// per output row (split-K), each lane loads 8 weights (16 bytes) per step from chunks ks*warp + lane (mod
// KS*warp), wave-shuffle reduction, partial sums of a split row meet in LDS. MR x rows per pass (M == 1 has
// its own instantiation: fewer registers).
constexpr int kW8 = 8;  // weights per 16-byte load

__device__ inline void load_x8(const float *x, int64_t off, float (&v)[kW8]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
    const float4 a = p[0], b = p[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ inline void load_x8(const uint16_t *x, int64_t off, float (&v)[kW8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(x + off);
    const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w4[j] << 16), v[2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
}

template <typename TX, typename TY, int MR>
__global__ void __launch_bounds__(kThreads)
    linear_bf16w_rows_kernel(const TX *x, const uint16_t *w, TY *y, int64_t M, int64_t N, int64_t K, int KS) {
    __shared__ float part[kThreads / 32][MR];  // per wave (wave32 worst case)
    const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / warpSize / KS) + wave / KS;
    const int ks = wave % KS;
    const int64_t m0 = (int64_t)blockIdx.y * MR;
    const int rows = (int)((M - m0) < MR ? (M - m0) : MR);
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) acc[r] = 0.f;
    if (n < N) {  // no early return: every wave reaches the barrier below
        const uint4 *wr = reinterpret_cast<const uint4 *>(w + n * K);
        for (int64_t k = ((int64_t)ks * warpSize + lane) * kW8; k < K; k += (int64_t)KS * warpSize * kW8) {
            const uint4 u = wr[k / kW8];
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
            float wv[kW8];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                wv[2 * j] = __uint_as_float(w4[j] << 16), wv[2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (r < rows) {
                    float xv[kW8];
                    load_x8(x, (m0 + r) * K + k, xv);
#pragma unroll
                    for (int j = 0; j < kW8; ++j) acc[r] = fmaf(xv[j], wv[j], acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r)
        for (int off = warpSize / 2; off > 0; off >>= 1) acc[r] += __shfl_xor(acc[r], off);
    if (KS == 1) {
        if (lane == 0 && n < N)
            for (int r = 0; r < rows; ++r) store(y, (m0 + r) * N + n, acc[r]);
        return;
    }
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < MR; ++r) part[wave][r] = acc[r];
    __syncthreads();
    if (lane == 0 && ks == 0 && n < N)
        for (int r = 0; r < rows; ++r) {
            float v = 0.f;
            for (int i = 0; i < KS; ++i) v += part[wave + i][r];
            store(y, (m0 + r) * N + n, v);
        }
}

int device_warp_size() { return kernels::device_warp_size(kThreads); }

// Waves per output row: split until each lane makes at most ~5 16-byte loads per row (at most 8 waves).
// Swept on gfx1151 at 1/2/4/8: the best factor at every target shape
// left each lane 2.5-5 loads - K = 640 unsplit, 2560 at 2 (router 73 -> 80% of read peak), 6144 at 8
// (out_proj 60 -> 96%).
int split_k(int64_t K) {
    const int64_t per_wave_load = (int64_t)device_warp_size() * kW8;
    int ks = 1;
    while (ks < 8 && K > 5 * per_wave_load * ks) ks *= 2;
    return ks;
}

template <typename TX, typename TY>
void launch_rows(const TX *x, const uint16_t *w, TY *y, int64_t M, int64_t N, int64_t K, hipStream_t stream) {
    const int waves = kThreads / device_warp_size(), KS = split_k(K);
    STRIX_CHECK(KS >= 1 && waves % KS == 0, "linear_bf16w: split-K factor ", KS, " must divide the ", waves,
                " waves of a block");
    const int64_t rows_per_block = waves / KS;
    if (M == 1) {
        dim3 grid((unsigned)((N + rows_per_block - 1) / rows_per_block), 1u);
        linear_bf16w_rows_kernel<TX, TY, 1><<<grid, dim3(kThreads), 0, stream>>>(x, w, y, M, N, K, KS);
    } else {
        dim3 grid((unsigned)((N + rows_per_block - 1) / rows_per_block), (unsigned)((M + kRowsPerBlock - 1) / kRowsPerBlock));
        linear_bf16w_rows_kernel<TX, TY, kRowsPerBlock><<<grid, dim3(kThreads), 0, stream>>>(x, w, y, M, N, K, KS);
    }
}

template <typename TX, typename TY>
void launch_bf16w(const TX *x, const uint16_t *w, TY *y, int64_t M, int64_t N, int64_t K, hipStream_t stream) {
    const bool rows_path = K % kW8 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(w) % 16 == 0;
    if (rows_path) {
        launch_rows(x, w, y, M, N, K, stream);
        return;
    }
    dim3 grid((unsigned)N, (unsigned)((M + kRowsPerBlock - 1) / kRowsPerBlock)), block(kThreads);
    linear_bf16w_kernel<TX, TY><<<grid, block, 0, stream>>>(x, w, y, M, N, K);
}

}  // namespace

void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act,
                  hipStream_t stream) {
    linear_bf16w(x, w, y, M, N, K, act, act, stream);
}

void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act, Act out_act,
                  hipStream_t stream) {
    STRIX_CHECK(x != nullptr && w != nullptr && y != nullptr, "linear_bf16w: null pointer (x=", x, ", w=", w, ", y=", y,
                ")");
    STRIX_CHECK(M >= 1 && M <= kMaxM, "linear_bf16w: M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(N >= 1 && N <= kMaxN, "linear_bf16w: N (rows of W / outputs) = ", N, ", expected 1..", kMaxN);
    STRIX_CHECK(K >= 1 && K <= kMaxK, "linear_bf16w: K (inner dim) = ", K, ", expected 1..", kMaxK);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "linear_bf16w: unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), "linear_bf16w: output dtype ",
                act_name(out_act), " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    uintptr_t xe = xb + (uintptr_t)(M * K) * es, ye = yb + (uintptr_t)(M * N) * oes;
    STRIX_CHECK(ye <= xb || xe <= yb, "linear_bf16w: output y [", y, ", +", M * N * (int64_t)oes,
                " B) overlaps input x [", x, ", +", M * K * (int64_t)es, " B); in-place isn't supported");

    if (act == Act::F32)
        launch_bf16w(static_cast<const float *>(x), w, static_cast<float *>(y), M, N, K, stream);
    else if (out_act == Act::F32)
        launch_bf16w(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, N, K, stream);
    else
        launch_bf16w(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, N, K, stream);
    STRIX_HIP_CHECK_LAUNCH("linear_bf16w ", act_name(act), "->", act_name(out_act), " M=", M, " N=", N, " K=", K);
}

}  // namespace strix::kernels
