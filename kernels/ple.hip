#include "kernels/ple.hpp"

#include "common/hip_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxTokens = 65535;  // grid.y of ple_gate
constexpr int64_t kMaxH = 16, kMaxK = 8, kMaxDilation = 16;
constexpr int64_t kMaxD = 1 << 20;

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

// Block (stream s, token t): the four sums over the stream's d columns, then vn for them.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    ple_gate_kernel(const T *kv, int64_t ld, const T *x, const float *wk, const float *wq, const float *wc,
                    const T *state_in, int64_t d, int64_t H, int64_t S, float eps, T *hist, float *sigma) {
    __shared__ float red[4][kThreads / 32];
    const int64_t s = blockIdx.x, t = blockIdx.y, n4 = H * d;
    if (t == 0)  // this stream's slice of the state rows, ahead of the new ones
        for (int64_t i = threadIdx.x; i < S * d; i += blockDim.x) {
            const int64_t r = i / d, c = s * d + i % d;
            hist[r * n4 + c] = state_in[r * n4 + c];
        }
    const T *key = kv + t * ld + s * d, *value = kv + t * ld + n4, *xs = x + t * n4 + s * d;
    float kk = 0.f, xx = 0.f, kx = 0.f, vv = 0.f;
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        const float k = load(key, j), xv = load(xs, j), v = load(value, j);
        kk = fmaf(k, k, kk);
        xx = fmaf(xv, xv, xx);
        kx = fmaf(k * xv, (1.0f + wk[s * d + j]) * (1.0f + wq[s * d + j]), kx);
        vv = fmaf(v, v, vv);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        kk += __shfl_down(kk, off, 32);
        xx += __shfl_down(xx, off, 32);
        kx += __shfl_down(kx, off, 32);
        vv += __shfl_down(vv, off, 32);
    }
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    if (lane == 0) {
        red[0][wave] = kk;
        red[1][wave] = xx;
        red[2][wave] = kx;
        red[3][wave] = vv;
    }
    __syncthreads();
    if (wave == 0) {
        float rk = lane < (kThreads / 32) ? red[0][lane] : 0.f;
        float rx = lane < (kThreads / 32) ? red[1][lane] : 0.f;
        float rkx = lane < (kThreads / 32) ? red[2][lane] : 0.f;
        float rv = lane < (kThreads / 32) ? red[3][lane] : 0.f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            rk += __shfl_down(rk, off, 32);
            rx += __shfl_down(rx, off, 32);
            rkx += __shfl_down(rkx, off, 32);
            rv += __shfl_down(rv, off, 32);
        }
        if (lane == 0) {
            red[0][0] = rk;
            red[1][0] = rx;
            red[2][0] = rkx;
            red[3][0] = rv;
        }
    }
    __syncthreads();
    const float fd = (float)d;
    const float inv_k = 1.0f / sqrtf(red[0][0] / fd + eps), inv_q = 1.0f / sqrtf(red[1][0] / fd + eps);
    float g = inv_k * inv_q * red[2][0] / sqrtf(fd);
    g = g > 0.f ? sqrtf(fmaxf(g, 1e-6f)) : (g < 0.f ? -sqrtf(fmaxf(-g, 1e-6f)) : 0.f);  // sign(0) = 0
    const float sg = 1.0f / (1.0f + expf(-g));
    const float inv_v = 1.0f / sqrtf(sg * sg * red[3][0] / fd + eps);
    if (threadIdx.x == 0) sigma[t * H + s] = sg;
    T *out = hist + (S + t) * n4 + s * d;
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        const float v = sg * load(value, j);  // the gated value, as the reference stores it
        store(out, j, v * inv_v * (1.0f + wc[s * d + j]));
    }
}

// Element (t, c): the dilated depthwise conv over hist, silu, plus the gated value, into X. The first
// S * n4 extra elements of the index space write the new state.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    ple_conv_kernel(const T *hist, const float *sigma, const T *kv, int64_t ld, const float *conv_w, int64_t T_,
                    int64_t H, int64_t d, int64_t K, int64_t dil, int64_t S, T *x, T *state_out) {
    const int64_t n4 = H * d, n_out = T_ * n4, n_all = n_out + S * n4;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += (int64_t)gridDim.x * blockDim.x) {
        if (i >= n_out) {  // state_out row r = history row T + r
            const int64_t e = i - n_out;
            state_out[e] = hist[T_ * n4 + e];
            continue;
        }
        const int64_t t = i / n4, c = i % n4, s = c / d, j = c % d;
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k) acc = fmaf(conv_w[c * K + k], load(hist, (S + t - (K - 1 - k) * dil) * n4 + c), acc);
        const float cv = acc / (1.0f + expf(-acc));
        const float v = sigma[t * H + s] * load(kv, t * ld + n4 + j);
        store(x, i, load(x, i) + (v + cv));
    }
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + b_bytes && y < x + a_bytes;
}

size_t check_ple(const char *fn, const PleShape &sh, Act act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(sh.T >= 1 && sh.T <= kMaxTokens, fn, ": T = ", sh.T, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(sh.H >= 1 && sh.H <= kMaxH, fn, ": H = ", sh.H, " streams, expected 1..", kMaxH);
    STRIX_CHECK(sh.d >= 1 && sh.d <= kMaxD, fn, ": d = ", sh.d, ", expected 1..", kMaxD);
    STRIX_CHECK(sh.K >= 1 && sh.K <= kMaxK, fn, ": conv taps K = ", sh.K, ", expected 1..", kMaxK);
    STRIX_CHECK(sh.dilation >= 1 && sh.dilation <= kMaxDilation, fn, ": dilation = ", sh.dilation, ", expected 1..",
                kMaxDilation);
    STRIX_CHECK(sh.ld >= sh.H * sh.d + sh.d, fn, ": kv row stride ld = ", sh.ld, " < H*d + d = ", sh.H * sh.d + sh.d,
                " (key then value)");
    return act == Act::F32 ? 4 : 2;
}

}  // namespace

void ple_gate(const void *kv, const void *x, const float *norm_key, const float *norm_query, const float *norm_conv,
              const void *state_in, const PleShape &sh, float eps, void *hist, float *sigma, Act act,
              hipStream_t stream) {
    const char *fn = "ple_gate";
    const size_t es = check_ple(fn, sh, act);
    STRIX_CHECK(kv && x && norm_key && norm_query && norm_conv && hist && sigma && (state_in || sh.state_rows() == 0),
                fn, ": null pointer (kv=", kv, ", x=", x, ", norms=", (const void *)norm_key, "/",
                (const void *)norm_query, "/", (const void *)norm_conv, ", state_in=", state_in, ", hist=", hist,
                ", sigma=", (void *)sigma, ")");
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps must be finite and > 0, got ", eps);
    const int64_t n4 = sh.H * sh.d, S = sh.state_rows();
    const size_t hist_bytes = (size_t)((S + sh.T) * n4) * es;
    STRIX_CHECK(!overlaps(hist, hist_bytes, x, (size_t)(sh.T * n4) * es) &&
                    !overlaps(hist, hist_bytes, kv, (size_t)(sh.T * sh.ld) * es) &&
                    (S == 0 || !overlaps(hist, hist_bytes, state_in, (size_t)(S * n4) * es)),
                fn, ": the history buffer overlaps x, kv or state_in");
    dim3 grid((unsigned)sh.H, (unsigned)sh.T), block(kThreads);
    if (act == Act::F32)
        ple_gate_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float *>(kv), sh.ld, static_cast<const float *>(x), norm_key, norm_query, norm_conv,
            static_cast<const float *>(state_in), sh.d, sh.H, S, eps, static_cast<float *>(hist), sigma);
    else
        ple_gate_kernel<uint16_t><<<grid, block, 0, stream>>>(
            static_cast<const uint16_t *>(kv), sh.ld, static_cast<const uint16_t *>(x), norm_key, norm_query, norm_conv,
            static_cast<const uint16_t *>(state_in), sh.d, sh.H, S, eps, static_cast<uint16_t *>(hist), sigma);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", sh.T, " H=", sh.H, " d=", sh.d);
}

void ple_conv(const void *hist, const float *sigma, const void *kv, const float *conv_w, const PleShape &sh, void *x,
              void *state_out, Act act, hipStream_t stream) {
    const char *fn = "ple_conv";
    const size_t es = check_ple(fn, sh, act);
    const int64_t n4 = sh.H * sh.d, S = sh.state_rows();
    STRIX_CHECK(hist && sigma && kv && conv_w && x && (state_out || S == 0), fn, ": null pointer (hist=", hist,
                ", sigma=", (const void *)sigma, ", kv=", kv, ", conv_w=", (const void *)conv_w, ", x=", x,
                ", state_out=", state_out, ")");
    const size_t hist_bytes = (size_t)((S + sh.T) * n4) * es;
    STRIX_CHECK(!overlaps(x, (size_t)(sh.T * n4) * es, hist, hist_bytes) &&
                    !overlaps(x, (size_t)(sh.T * n4) * es, kv, (size_t)(sh.T * sh.ld) * es) &&
                    (S == 0 || !overlaps(state_out, (size_t)(S * n4) * es, hist, hist_bytes)),
                fn, ": x or state_out overlaps the history or kv");
    const int64_t n = (sh.T + S) * n4;
    const unsigned blocks = (unsigned)std::min<int64_t>((n + kThreads - 1) / kThreads, 8192);
    if (act == Act::F32)
        ple_conv_kernel<float><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float *>(hist), sigma, static_cast<const float *>(kv), sh.ld, conv_w, sh.T, sh.H, sh.d,
            sh.K, sh.dilation, S, static_cast<float *>(x), static_cast<float *>(state_out));
    else
        ple_conv_kernel<uint16_t><<<blocks, kThreads, 0, stream>>>(
            static_cast<const uint16_t *>(hist), sigma, static_cast<const uint16_t *>(kv), sh.ld, conv_w, sh.T, sh.H,
            sh.d, sh.K, sh.dilation, S, static_cast<uint16_t *>(x), static_cast<uint16_t *>(state_out));
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", sh.T, " H=", sh.H, " d=", sh.d);
}

}  // namespace strix::kernels
