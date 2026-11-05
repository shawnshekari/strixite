#include "kernels/logits_bias.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"

#include <climits>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;

// One thread per entry, a loop over its rows (a request's rows in one forward: at most the verify's few).
__global__ void __launch_bounds__(kThreads)
    bias_kernel(float *logits, int64_t ld, const LogitBiasEntry *entries, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const LogitBiasEntry e = entries[i];
    for (int32_t r = 0; r < e.rows; ++r) logits[(int64_t)(e.row0 + r) * ld + e.id] += e.v;
}

}  // namespace

void logits_bias(float *logits, int64_t n_rows, int64_t ld, const LogitBiasEntry *entries, int64_t n,
                 hipStream_t stream) {
    const char *fn = "kernels::logits_bias";
    STRIX_CHECK(logits != nullptr && entries != nullptr, fn, ": logits ", (const void *)logits, ", entries ",
                (const void *)entries, ", expected both non-null");
    STRIX_CHECK(n_rows >= 1 && n_rows <= INT32_MAX, fn, ": n_rows = ", n_rows, ", expected 1..", INT32_MAX);
    STRIX_CHECK(ld >= 1 && ld <= INT32_MAX, fn, ": ld = ", ld, ", expected 1..", INT32_MAX);
    STRIX_CHECK(n >= 1 && n <= (int64_t)INT32_MAX * kThreads, fn, ": ", n, " entries, expected >= 1");
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    bias_kernel<<<(unsigned)blocks, kThreads, 0, stream>>>(logits, ld, entries, n);
    STRIX_HIP_CHECK(hipGetLastError(), fn, ": launch of ", blocks, " workgroups for ", n, " entries");
}

}  // namespace strix::kernels
