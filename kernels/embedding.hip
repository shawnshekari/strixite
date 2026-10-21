#include "kernels/embedding.hpp"

#include "common/hip_check.hpp"

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr unsigned long long kNoError = ~0ull;
constexpr int64_t kMaxTokens = 0x7fffffff;
constexpr int64_t kMaxRowLen = 1 << 20;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    embedding_kernel(const int32_t *ids, const uint16_t *table, int64_t vocab, int64_t d, T *out, EmbeddingError *err,
                     int64_t copies, int64_t out_ts) {
    int64_t t = blockIdx.x;
    int32_t id = ids[t];
    T *orow = out + t * out_ts;
    if (id < 0 || id >= vocab) {
        // Lowest bad position wins regardless of which block runs first (blocks run in any order).
        if (threadIdx.x == 0) atomicMin(&err->packed, ((unsigned long long)t << 32) | (uint32_t)id);
        for (int64_t j = threadIdx.x; j < d; j += blockDim.x)
            for (int64_t c = 0; c < copies; ++c) orow[c * d + j] = T(0);
        return;
    }
    const uint16_t *row = table + (int64_t)id * d;
    if constexpr (sizeof(T) == 2) {
        if ((d & 7) == 0 && ((uintptr_t)row & 15) == 0 && ((uintptr_t)orow & 15) == 0 && copies == 1) {
            const int64_t d8 = d / 8;
            const uint4 *row4 = reinterpret_cast<const uint4 *>(row);
            uint4 *orow4 = reinterpret_cast<uint4 *>(orow);
            for (int64_t j = threadIdx.x; j < d8; j += blockDim.x) {
                orow4[j] = row4[j];
            }
            return;
        }
    }
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        T v;
        if constexpr (sizeof(T) == 2) v = row[j];  // BF16 -> BF16: copy bits
        else v = bf16_bits_to_f32(row[j]);
        for (int64_t c = 0; c < copies; ++c) orow[c * d + j] = v;
    }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    embedding_q8_kernel(const int32_t *ids, const uint8_t *table_q, const uint16_t *table_scale,
                        const uint16_t *table_minv, int64_t vocab, int64_t d, int64_t G,
                        T *out, EmbeddingError *err, int64_t copies, int64_t out_ts) {
    int64_t t = blockIdx.x;
    int32_t id = ids[t];
    T *orow = out + t * out_ts;
    if (id < 0 || id >= vocab) {
        if (threadIdx.x == 0) atomicMin(&err->packed, ((unsigned long long)t << 32) | (uint32_t)id);
        for (int64_t j = threadIdx.x; j < d; j += blockDim.x)
            for (int64_t c = 0; c < copies; ++c) orow[c * d + j] = T(0);
        return;
    }
    const int64_t gpr = d / G;
    const uint8_t *q_row = table_q + (int64_t)id * d;
    const uint16_t *scale_row = table_scale + (int64_t)id * gpr;
    const uint16_t *min_row = table_minv + (int64_t)id * gpr;

    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        const int64_t g = j / G;
        const float s = bf16_bits_to_f32(scale_row[g]);
        const float m = bf16_bits_to_f32(min_row[g]);
        const float v = (float)q_row[j] * s + m;
        T o;
        if constexpr (sizeof(T) == 2) {
            o = f32_to_bf16_bits(v);
        } else {
            o = v;
        }
        for (int64_t c = 0; c < copies; ++c) orow[c * d + j] = o;
    }
}

}  // namespace

void reset_embedding_error(EmbeddingError *err_dev, hipStream_t stream) {
    STRIX_CHECK(err_dev != nullptr, "embedding error slot is null");
    STRIX_HIP_CHECK(hipMemsetAsync(err_dev, 0xff, sizeof(EmbeddingError), stream), "resetting embedding error slot");
}

// copies / out_ts: checked, and out_ts 0 resolved to copies * d.
static int64_t copy_layout(const char *fn, int64_t d, int64_t copies, int64_t out_ts) {
    STRIX_CHECK(copies >= 1 && copies <= 64, fn, ": copies = ", copies, ", expected 1..64");
    const int64_t ts = out_ts == 0 ? copies * d : out_ts;
    STRIX_CHECK(ts >= copies * d, fn, ": out token stride ", out_ts, " < ", copies, " copies x d = ", copies * d);
    return ts;
}

void embedding_lookup(const int32_t *ids, int64_t n_tokens, const uint16_t *table, int64_t vocab, int64_t d, void *out,
                      Act act, EmbeddingError *err_dev, hipStream_t stream, int64_t copies, int64_t out_ts) {
    STRIX_CHECK(ids != nullptr && table != nullptr && out != nullptr && err_dev != nullptr,
                "embedding_lookup: null pointer (ids=", ids, ", table=", table, ", out=", out, ", err=", err_dev, ")");
    STRIX_CHECK(n_tokens >= 1 && n_tokens <= kMaxTokens, "embedding_lookup: n_tokens = ", n_tokens, ", expected 1..",
                kMaxTokens);
    STRIX_CHECK(vocab >= 1 && vocab <= 0x7fffffff, "embedding_lookup: vocab = ", vocab,
                ", expected 1..2^31-1 (ids are int32)");
    STRIX_CHECK(d >= 1 && d <= kMaxRowLen, "embedding_lookup: d = ", d, ", expected 1..", kMaxRowLen);
    const int64_t ts = copy_layout("embedding_lookup", d, copies, out_ts);
    dim3 grid((unsigned)n_tokens), block(kThreads);
    switch (act) {
        case Act::F32:
            embedding_kernel<float><<<grid, block, 0, stream>>>(ids, table, vocab, d, static_cast<float *>(out), err_dev,
                                                                copies, ts);
            break;
        case Act::BF16:
            embedding_kernel<uint16_t><<<grid, block, 0, stream>>>(ids, table, vocab, d, static_cast<uint16_t *>(out),
                                                                   err_dev, copies, ts);
            break;
        default: STRIX_FAIL("embedding_lookup: unsupported activation dtype ", (int)act);
    }
    STRIX_HIP_CHECK_LAUNCH("embedding_lookup ", act_name(act), " n_tokens=", n_tokens, " vocab=", vocab, " d=", d);
}

void embedding_lookup(const int32_t *ids, int64_t n_tokens, const Q8DeviceView &table, void *out,
                      Act act, EmbeddingError *err_dev, hipStream_t stream, int64_t copies, int64_t out_ts) {
    STRIX_CHECK(ids != nullptr && table.q != nullptr && table.scale != nullptr && table.minv != nullptr &&
                out != nullptr && err_dev != nullptr,
                "embedding_lookup (q8): null pointer");
    STRIX_CHECK(n_tokens >= 1 && n_tokens <= kMaxTokens, "embedding_lookup: n_tokens = ", n_tokens, ", expected 1..",
                kMaxTokens);
    STRIX_CHECK(table.N >= 1 && table.N <= 0x7fffffff, "embedding_lookup: vocab = ", table.N,
                ", expected 1..2^31-1 (ids are int32)");
    STRIX_CHECK(table.K >= 1 && table.K <= kMaxRowLen, "embedding_lookup: d = ", table.K, ", expected 1..", kMaxRowLen);
    STRIX_CHECK(table.G == 32 || table.G == 64 || table.G == 128, "embedding_lookup: G = ", table.G,
                ", expected 32, 64, or 128");
    STRIX_CHECK(table.K % table.G == 0, "embedding_lookup: d (", table.K, ") must be divisible by G (", table.G, ")");
    const int64_t ts = copy_layout("embedding_lookup (q8)", table.K, copies, out_ts);
    dim3 grid((unsigned)n_tokens), block(kThreads);
    switch (act) {
        case Act::F32:
            embedding_q8_kernel<float><<<grid, block, 0, stream>>>(
                ids, table.q, table.scale, table.minv, table.N, table.K, table.G,
                static_cast<float *>(out), err_dev, copies, ts);
            break;
        case Act::BF16:
            embedding_q8_kernel<uint16_t><<<grid, block, 0, stream>>>(
                ids, table.q, table.scale, table.minv, table.N, table.K, table.G,
                static_cast<uint16_t *>(out), err_dev, copies, ts);
            break;
        default: STRIX_FAIL("embedding_lookup: unsupported activation dtype ", (int)act);
    }
    STRIX_HIP_CHECK_LAUNCH("embedding_lookup_q8 ", act_name(act), " n_tokens=", n_tokens, " vocab=", table.N, " d=", table.K);
}

void check_embedding_error(const EmbeddingError *err_dev, int64_t vocab, hipStream_t stream) {
    STRIX_CHECK(err_dev != nullptr, "embedding error slot is null");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "before reading the embedding error slot");
    EmbeddingError e{};
    STRIX_HIP_CHECK(hipMemcpy(&e, err_dev, sizeof(e), hipMemcpyDeviceToHost), "reading the embedding error slot");
    report_embedding_error(e, vocab);
}

void report_embedding_error(const EmbeddingError &e, int64_t vocab) {
    STRIX_CHECK(e.packed == kNoError, "embedding_lookup: token id ", (int32_t)(uint32_t)(e.packed & 0xffffffffull),
                " at position ", e.packed >> 32, " is outside [0, ", vocab,
                ") - that row was zeroed (lowest bad position shown)");
}

}  // namespace strix::kernels
