#include "kernels/moe_router.hpp"

#include "common/hip_check.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

// One 32-lane group per token (a whole wave on gfx11's wave32; half of one on wave64 hardware - the shuffles
// use width 32, so correctness doesn't depend on the wave size). Decode is M = 1: a single group, latency-bound.
constexpr int kLanes = 32;
constexpr int64_t kMaxExperts = 1024;  // qwen4_exp has 512; up to 32 logits a lane in registers
constexpr int64_t kMaxTopK = kLanes;   // lane r holds the r-th pick
constexpr int64_t kMaxTokens = 0x7fffffff;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

template <typename T> __device__ inline float load(const T *p, int64_t i);
template <> __device__ inline float load<float>(const float *p, int64_t i) { return p[i]; }
template <> __device__ inline float load<uint16_t>(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }

// The logits stay in registers: lane l holds experts l, l + 32, .. (NV of them, -inf past E) and its own best
// (value, index). A round is one 32-lane arg-max over the lanes' bests; only the lane whose best was picked marks it
// -inf and rescans its NV registers. The previous form rescanned all of E from LDS every round (9.6 us traced for
// E = 512, k = 10). Order: larger logit first, then lower index - within a lane the ascending scan with a strict >
// keeps the lower index, across lanes the compare does. Same picks, order and softmax code as before.
template <typename T, int NV>
__global__ void __launch_bounds__(kLanes)
    moe_router_kernel(const T *logits, int64_t ld, int E, int k, const T *shared, int64_t shared_ld, int32_t shared_id,
                      int32_t *ids, float *coef, int A, uint32_t *err) {
    const int64_t m = blockIdx.x;
    const int lane = threadIdx.x;
    const T *row = logits + m * ld;

    // Load; the smallest index of a non-finite logit (E = the shared one), INT_MAX if none.
    float v[NV];
    int bad = INT_MAX;
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int e = lane + j * kLanes;
        const float x = e < E ? load(row, e) : -INFINITY;
        v[j] = x;
        if (e < E && !isfinite(x) && e < bad) bad = e;
    }
    float sl = 0.f;
    if (shared != nullptr) {
        sl = load(shared, m * shared_ld);
        if (!isfinite(sl) && E < bad) bad = E;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) bad = min(bad, __shfl_xor(bad, off, 32));
    int32_t *id_row = ids + m * A;
    float *coef_row = coef + m * A;
    if (bad != INT_MAX) {
        if (lane == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
            err[1] = (uint32_t)m;
            err[2] = (uint32_t)bad;
        }
        for (int a = lane; a < A; a += kLanes) {
            id_row[a] = -1;
            coef_row[a] = __builtin_nanf("");
        }
        return;
    }

    // This lane's best (static register indices only - a dynamic index would put v in scratch).
    auto lane_best = [&](float &bv, int &bi) {
        bv = -INFINITY, bi = INT_MAX;
#pragma unroll
        for (int j = 0; j < NV; ++j)
            if (v[j] > bv) bv = v[j], bi = lane + j * kLanes;
    };
    float lv;
    int li;
    lane_best(lv, li);
    float my_v = -INFINITY, top = 0.f;
    int my_i = -1;
    for (int r = 0; r < k; ++r) {
        float bv = lv;
        int bi = li;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 32);
            const int oi = __shfl_xor(bi, off, 32);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (r == 0) top = bv;
        if (lane == r) my_v = bv, my_i = bi;
        if (bi == li) {  // this lane's best was picked (indices are unique to a lane)
#pragma unroll
            for (int j = 0; j < NV; ++j)
                if (lane + j * kLanes == bi) v[j] = -INFINITY;
            lane_best(lv, li);
        }
    }

    // Softmax over the k picks (= the full softmax renormalized over them).
    const float ex = lane < k ? expf(my_v - top) : 0.f;
    float sum = ex;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    if (lane < k) {
        id_row[lane] = my_i;
        coef_row[lane] = ex / sum;
    }
    if (shared != nullptr && lane == 0) {
        id_row[k] = shared_id;
        coef_row[k] = 1.0f / (1.0f + expf(-sl));
    }
}

constexpr int kAddThreads = 256;

__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
template <typename T> __device__ inline void store(T *p, int64_t i, float v);
template <> __device__ inline void store<float>(float *p, int64_t i, float v) { p[i] = v; }
template <> __device__ inline void store<uint16_t>(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// Grid (column slices, token rows): a block covers kAddThreads columns of one row. One block per row put a decode
// step's whole row on one workgroup (moe_shared_add_inject measured 15.7 us traced vs 3.4 + 1.9 for the two
// kernels it replaced); the first slice's block raises a row's error.
template <typename T>
__global__ void __launch_bounds__(kAddThreads)
    moe_shared_add_kernel(T *y, const T *s, const float *gate_logit, int64_t gate_ld, int64_t N, uint32_t gate_index,
                          uint32_t *err) {
    const int64_t m = blockIdx.y;
    const float gl = gate_logit[m * gate_ld];
    const bool ok = isfinite(gl);
    if (!ok && blockIdx.x == 0 && threadIdx.x == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
        err[1] = (uint32_t)m;
        err[2] = gate_index;
    }
    const float g = ok ? 1.f / (1.f + expf(-gl)) : __builtin_nanf("");
    T *yr = y + m * N;
    const T *sr = s + m * N;
    const int64_t n = (int64_t)blockIdx.x * kAddThreads + threadIdx.x;
    if (n < N) store(yr, n, load(yr, n) + g * load(sr, n));
}

// The stored value of v in T (BF16: rounded once, as store() then load() would give).
template <typename T> __device__ inline float as_stored(float v);
template <> __device__ inline float as_stored<float>(float v) { return v; }
template <> __device__ inline float as_stored<uint16_t>(float v) { return __uint_as_float((uint32_t)f32_to_bf16_bits(v) << 16); }

// moe_shared_add then hc_inject in one pass, bit-identical to the two kernels: y as moe_shared_add stores it, then
// each stream x[m, s*N + n] = fma(w_in[m, s], y, x) rounded once (hc_inject's arithmetic on the stored y).
template <typename T>
__global__ void __launch_bounds__(kAddThreads)
    moe_shared_add_inject_kernel(T *y, const T *s, const float *gate_logit, int64_t gate_ld, int64_t N,
                                 uint32_t gate_index, T *x, const float *w_in, uint32_t *err) {
    constexpr int kStreams = 4;
    const int64_t m = blockIdx.y;
    const float gl = gate_logit[m * gate_ld];
    const bool ok = isfinite(gl);
    if (!ok && blockIdx.x == 0 && threadIdx.x == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
        err[1] = (uint32_t)m;
        err[2] = gate_index;
    }
    const float g = ok ? 1.f / (1.f + expf(-gl)) : __builtin_nanf("");
    float w[kStreams];
#pragma unroll
    for (int st = 0; st < kStreams; ++st) w[st] = w_in[m * kStreams + st];
    T *yr = y + m * N;
    const T *sr = s + m * N;
    T *xr = x + m * kStreams * N;
    const int64_t n = (int64_t)blockIdx.x * kAddThreads + threadIdx.x;
    if (n < N) {
        const float v = as_stored<T>(load(yr, n) + g * load(sr, n));
        store(yr, n, v);
#pragma unroll
        for (int st = 0; st < kStreams; ++st) store(xr, st * N + n, fmaf(w[st], v, load(xr, st * N + n)));
    }
}

}  // namespace

void moe_router(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                hipStream_t stream) {
    STRIX_CHECK(logits != nullptr && ids != nullptr && coef != nullptr && err != nullptr,
                "moe_router: null pointer (logits=", logits, ", ids=", (const void *)ids, ", coef=", (const void *)coef,
                ", err=", (const void *)err, ")");
    STRIX_CHECK(M >= 1 && M <= kMaxTokens, "moe_router: M = ", M, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(E >= 1 && E <= kMaxExperts, "moe_router: E = ", E, " experts, expected 1..", kMaxExperts);
    STRIX_CHECK(k >= 1 && k <= E && k <= kMaxTopK, "moe_router: k = ", k, ", expected 1..min(E = ", E, ", ",
                kMaxTopK, ")");
    STRIX_CHECK(ld >= E, "moe_router: logits row stride ld = ", ld, " < E = ", E);
    if (shared_logit != nullptr) {
        STRIX_CHECK(shared_ld >= 1, "moe_router: shared_ld = ", shared_ld, ", expected >= 1");
        STRIX_CHECK(shared_id >= 0, "moe_router: shared_id = ", shared_id, ", expected >= 0 (its expert row block)");
    }
    STRIX_CHECK((const void *)ids != (const void *)coef, "moe_router: ids and coef are the same buffer");
    const int A = (int)k + (shared_logit != nullptr ? 1 : 0);
    dim3 grid((unsigned)M), block(kLanes);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "moe_router: unsupported activation dtype ", (int)act);
    // Logits per lane: 16 covers qwen4_exp's 512 (4, 8 for smaller E keep the scan short), 32 the 1024 limit.
    const int nv = E <= 128 ? 4 : E <= 256 ? 8 : E <= 512 ? 16 : 32;
    auto go = [&](auto tag, auto nvc) {
        using T = decltype(tag);
        moe_router_kernel<T, decltype(nvc)::value><<<grid, block, 0, stream>>>(
            static_cast<const T *>(logits), ld, (int)E, (int)k, static_cast<const T *>(shared_logit), shared_ld,
            shared_id, ids, coef, A, err);
    };
    auto by_nv = [&](auto tag) {
        switch (nv) {
            case 4: go(tag, std::integral_constant<int, 4>{}); break;
            case 8: go(tag, std::integral_constant<int, 8>{}); break;
            case 16: go(tag, std::integral_constant<int, 16>{}); break;
            default: go(tag, std::integral_constant<int, 32>{}); break;
        }
    };
    if (act == Act::F32) by_nv(float{});
    else by_nv(uint16_t{});
    STRIX_HIP_CHECK_LAUNCH("moe_router ", act_name(act), " M=", M, " E=", E, " k=", k,
                           shared_logit ? " +shared" : "");
}

void moe_shared_add(void *y, const void *s, const float *gate_logit, int64_t gate_ld, int64_t M, int64_t N,
                    uint32_t gate_index, Act act, uint32_t *err, hipStream_t stream) {
    STRIX_CHECK(y != nullptr && s != nullptr && gate_logit != nullptr && err != nullptr, "moe_shared_add: null pointer (y=",
                y, ", s=", s, ", gate_logit=", (const void *)gate_logit, ", err=", (const void *)err, ")");
    STRIX_CHECK(M >= 1 && M <= kMaxTokens, "moe_shared_add: M = ", M, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(N >= 1 && N <= (1ll << 24), "moe_shared_add: N = ", N, ", expected 1..2^24");
    STRIX_CHECK(gate_ld >= 1, "moe_shared_add: gate_ld = ", gate_ld, ", expected >= 1");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "moe_shared_add: unsupported activation dtype ", (int)act);
    const int64_t bytes = M * N * (act == Act::F32 ? 4 : 2);
    const auto yb = reinterpret_cast<uintptr_t>(y), sb = reinterpret_cast<uintptr_t>(s);
    STRIX_CHECK(yb + (uintptr_t)bytes <= sb || sb + (uintptr_t)bytes <= yb, "moe_shared_add: s [", s, ", +", bytes,
                " B) overlaps y [", y, ", +", bytes, " B)");
    STRIX_CHECK(M <= 65535, "moe_shared_add: M = ", M, " tokens, at most 65535 (grid.y)");
    dim3 grid((unsigned)((N + kAddThreads - 1) / kAddThreads), (unsigned)M), block(kAddThreads);
    if (act == Act::F32)
        moe_shared_add_kernel<float><<<grid, block, 0, stream>>>(static_cast<float *>(y), static_cast<const float *>(s),
                                                                 gate_logit, gate_ld, N, gate_index, err);
    else
        moe_shared_add_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<uint16_t *>(y),
                                                                    static_cast<const uint16_t *>(s), gate_logit,
                                                                    gate_ld, N, gate_index, err);
    STRIX_HIP_CHECK_LAUNCH("moe_shared_add ", act_name(act), " M=", M, " N=", N);
}

void moe_shared_add_inject(void *y, const void *s, const float *gate_logit, int64_t gate_ld, int64_t M, int64_t N,
                           uint32_t gate_index, void *x, const float *w_in, int64_t H, Act act, uint32_t *err,
                           hipStream_t stream) {
    const char *fn = "moe_shared_add_inject";
    STRIX_CHECK(y != nullptr && s != nullptr && gate_logit != nullptr && err != nullptr && x != nullptr && w_in != nullptr,
                fn, ": null pointer (y=", y, ", s=", s, ", gate_logit=", (const void *)gate_logit, ", x=", x,
                ", w_in=", (const void *)w_in, ", err=", (const void *)err, ")");
    STRIX_CHECK(M >= 1 && M <= kMaxTokens, fn, ": M = ", M, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(N >= 1 && N <= (1ll << 24), fn, ": N = ", N, ", expected 1..2^24");
    STRIX_CHECK(H == 4, fn, ": H = ", H, " streams, expected 4 (qwen4_exp's hc_count)");
    STRIX_CHECK(gate_ld >= 1, fn, ": gate_ld = ", gate_ld, ", expected >= 1");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2, ybytes = M * N * es, xbytes = M * H * N * es;
    const auto yb = reinterpret_cast<uintptr_t>(y), sb = reinterpret_cast<uintptr_t>(s), xb = reinterpret_cast<uintptr_t>(x);
    auto apart = [](uintptr_t a, int64_t an, uintptr_t b, int64_t bn) { return a + (uintptr_t)an <= b || b + (uintptr_t)bn <= a; };
    STRIX_CHECK(apart(yb, ybytes, sb, ybytes), fn, ": s [", s, ", +", ybytes, " B) overlaps y [", y, ", +", ybytes, " B)");
    STRIX_CHECK(apart(xb, xbytes, yb, ybytes) && apart(xb, xbytes, sb, ybytes), fn, ": x [", x, ", +", xbytes,
                " B) overlaps y [", y, "] or s [", s, "] (", ybytes, " B each)");
    STRIX_CHECK(M <= 65535, fn, ": M = ", M, " tokens, at most 65535 (grid.y)");
    dim3 grid((unsigned)((N + kAddThreads - 1) / kAddThreads), (unsigned)M), block(kAddThreads);
    if (act == Act::F32)
        moe_shared_add_inject_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float *>(y), static_cast<const float *>(s), gate_logit, gate_ld, N, gate_index,
            static_cast<float *>(x), w_in, err);
    else
        moe_shared_add_inject_kernel<uint16_t><<<grid, block, 0, stream>>>(
            static_cast<uint16_t *>(y), static_cast<const uint16_t *>(s), gate_logit, gate_ld, N, gate_index,
            static_cast<uint16_t *>(x), w_in, err);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " M=", M, " N=", N);
}

void check_router_error(const uint32_t *err, const char *what) {
    STRIX_CHECK(err != nullptr, "check_router_error(", what, "): null err");
    uint32_t e[3];
    STRIX_HIP_CHECK(hipMemcpy(e, err, sizeof(e), hipMemcpyDeviceToHost), "reading router error word for ", what);
    report_router_error(e, what);
}

void report_router_error(const uint32_t e[3], const char *what) {
    STRIX_CHECK(e[0] == 0, what, ": non-finite router logit at token ", e[1], ", expert ", e[2],
                " (E = the shared-expert gate); that token's ids are -1 and its coefficients NaN");
}

}  // namespace strix::kernels
