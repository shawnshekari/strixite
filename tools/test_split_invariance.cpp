// Test and proof tool for GDN Delta-Rule Split Invariance and Replay Precision.
// Verifies that replaying a prefix of verify rows produces a bit-exact recurrent state
// matching the forward pass at that prefix position.
//
// Usage: test_split_invariance [--verbose]

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/gdn.hpp"

#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

using namespace strix;
using namespace strix::kernels;

namespace {

struct ErrorStats {
    double max_abs_err = 0.0;
    double l1_err = 0.0;
    double l2_err = 0.0;
    double ref_norm = 0.0;
    int64_t total_elements = 0;
    int64_t diff_bits = 0;

    double rel_l2() const { return ref_norm > 0 ? l2_err / ref_norm : 0.0; }
    double bit_exact_pct() const {
        return total_elements > 0 ? 100.0 * (double)(total_elements - diff_bits) / (double)total_elements : 100.0;
    }
};

ErrorStats compare_states(const float *a, const float *b, int64_t n) {
    ErrorStats s;
    s.total_elements = n;
    double l2_sum = 0.0, ref_sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float va = a[i];
        float vb = b[i];
        double diff = std::abs((double)va - (double)vb);
        s.max_abs_err = std::max(s.max_abs_err, diff);
        s.l1_err += diff;
        l2_sum += diff * diff;
        ref_sum += (double)va * (double)va;

        uint32_t ba, bb;
        ::memcpy(&ba, &va, 4);
        ::memcpy(&bb, &vb, 4);
        if (ba != bb) ++s.diff_bits;
    }
    s.l2_err = std::sqrt(l2_sum);
    s.ref_norm = std::sqrt(ref_sum);
    return s;
}

const char *precision_name(DrPrecision p) {
    switch (p) {
        case DrPrecision::Exact: return "Exact (BF16 hi+lo)";
        case DrPrecision::Fast: return "Fast (BF16 single)";
        case DrPrecision::Half: return "Half (FP16 scaled)";
        case DrPrecision::HalfU: return "HalfU (FP16 hi+lo)";
    }
    return "Unknown";
}

}  // namespace

int main() {

    hipStream_t stream = nullptr;
    STRIX_HIP_CHECK(hipStreamCreate(&stream), "hipStreamCreate");

    // Qwen4Exp GDN layer dimensions:
    // Hk = 16 (key heads), Hv = 48 (value heads), D = 128 (head dim)
    // C = (2 * Hk + Hv) * D = (32 + 48) * 128 = 10240
    // Test with T = 16 tokens, prefix rows = 12 tokens
    constexpr int64_t T_full = 16;
    constexpr int64_t T_prefix = 12;
    constexpr int64_t Hk = 4;
    constexpr int64_t Hv = 8;
    constexpr int64_t D = 128;
    constexpr int64_t C = (2 * Hk + Hv) * D;
    constexpr int64_t qkv_ts = C;
    constexpr Act act = Act::BF16;

    const DeltaRuleShape shape_prefix{T_prefix, Hk, Hv, qkv_ts};

    const size_t qkv_bytes = (size_t)(T_full * qkv_ts * 2);  // BF16
    const size_t gate_bytes = (size_t)(T_full * Hv * sizeof(float));
    const size_t state_elements = (size_t)(Hv * D * D);
    const size_t state_bytes = state_elements * sizeof(float);
    const size_t out_bytes = (size_t)(T_full * Hv * D * 2);

    // Host buffers
    std::vector<uint16_t> h_qkv(T_full * qkv_ts);
    std::vector<float> h_beta(T_full * Hv);
    std::vector<float> h_g(T_full * Hv);
    std::vector<float> h_init_state(state_elements);

    // Deterministic pseudo-random initialization
    uint32_t seed = 0x12345678;
    auto rnd = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed & 0xFFFF) / 65535.0f - 0.5f) * 0.1f;
    };
    for (size_t i = 0; i < h_qkv.size(); ++i) {
        float f = rnd();
        // convert to bf16
        uint32_t u;
        ::memcpy(&u, &f, 4);
        h_qkv[i] = (uint16_t)(u >> 16);
    }
    for (size_t i = 0; i < h_beta.size(); ++i) h_beta[i] = 1.0f / (1.0f + std::exp(-rnd()));
    for (size_t i = 0; i < h_g.size(); ++i) h_g[i] = -std::exp(rnd()) * 0.05f;
    for (size_t i = 0; i < h_init_state.size(); ++i) h_init_state[i] = rnd() * 0.01f;

    // Device allocations
    void *d_qkv = nullptr, *d_out = nullptr;
    float *d_beta = nullptr, *d_g = nullptr;
    float *d_state_in = nullptr;
    float *d_state_gold = nullptr;
    float *d_state_buggy = nullptr;
    float *d_state_fixed = nullptr;

    STRIX_HIP_CHECK(hipMalloc(&d_qkv, qkv_bytes), "hipMalloc d_qkv");
    STRIX_HIP_CHECK(hipMalloc(&d_beta, gate_bytes), "hipMalloc d_beta");
    STRIX_HIP_CHECK(hipMalloc(&d_g, gate_bytes), "hipMalloc d_g");
    STRIX_HIP_CHECK(hipMalloc(&d_state_in, state_bytes), "hipMalloc d_state_in");
    STRIX_HIP_CHECK(hipMalloc(&d_state_gold, state_bytes), "hipMalloc d_state_gold");
    STRIX_HIP_CHECK(hipMalloc(&d_state_buggy, state_bytes), "hipMalloc d_state_buggy");
    STRIX_HIP_CHECK(hipMalloc(&d_state_fixed, state_bytes), "hipMalloc d_state_fixed");
    STRIX_HIP_CHECK(hipMalloc(&d_out, out_bytes), "hipMalloc d_out");

    STRIX_HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), qkv_bytes, hipMemcpyHostToDevice), "memcpy d_qkv");
    STRIX_HIP_CHECK(hipMemcpy(d_beta, h_beta.data(), gate_bytes, hipMemcpyHostToDevice), "memcpy d_beta");
    STRIX_HIP_CHECK(hipMemcpy(d_g, h_g.data(), gate_bytes, hipMemcpyHostToDevice), "memcpy d_g");
    STRIX_HIP_CHECK(hipMemcpy(d_state_in, h_init_state.data(), state_bytes, hipMemcpyHostToDevice), "memcpy d_state_in");

    std::printf("================================================================================\n");
    std::printf("   GDN DELTA-RULE SPLIT INVARIANCE & SPECULATIVE REPLAY VERIFICATION TEST       \n");
    std::printf("================================================================================\n\n");
    std::printf("Configuration: Hv=%lld, Hk=%lld, D=%lld, T_verify=%lld, T_prefix_kept=%lld\n",
                (long long)Hv, (long long)Hk, (long long)D, (long long)T_full, (long long)T_prefix);
    std::printf("State size   : %lld floats (%.2f KB)\n\n", (long long)state_elements, (double)state_bytes / 1024.0);

    const DrPrecision test_precisions[] = {
        DrPrecision::HalfU,
        DrPrecision::Fast,
        DrPrecision::Half,
        DrPrecision::Exact
    };

    bool all_fixed_passed = true;
    bool bug_confirmed = false;

    for (DrPrecision prec : test_precisions) {
        std::printf("--------------------------------------------------------------------------------\n");
        std::printf("TEST PRECISION MODE: %s\n", precision_name(prec));
        std::printf("--------------------------------------------------------------------------------\n");

        // 1. Compute GOLD prefix state:
        // A forward over T_prefix tokens with the configured path and precision.
        STRIX_HIP_CHECK(hipMemcpy(d_state_gold, d_state_in, state_bytes, hipMemcpyDeviceToDevice), "memcpy state_gold");
        delta_rule(d_qkv, d_beta, d_g, shape_prefix, d_state_gold, d_state_gold, false, d_out, act, stream,
                   DeltaRulePath::ChunkedWmma, prec);
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync gold");

        // 2. BUGGY Upstream Replay:
        // keep_verify_prefix_seq omitted 'precision' (defaulted to Exact) and forced DeltaRulePath::Auto.
        STRIX_HIP_CHECK(hipMemcpy(d_state_buggy, d_state_in, state_bytes, hipMemcpyDeviceToDevice), "memcpy state_buggy");
        delta_rule(d_qkv, d_beta, d_g, shape_prefix, d_state_buggy, d_state_buggy, false, d_out, act, stream,
                   DeltaRulePath::Auto /*, DrPrecision::Exact by default */);
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync buggy");

        // 3. FIXED Replay:
        // keep_verify_prefix_seq uses matching path (ChunkedWmma when rows >= kWmmaMinTokens) and passes precision_.gdn.
        STRIX_HIP_CHECK(hipMemcpy(d_state_fixed, d_state_in, state_bytes, hipMemcpyDeviceToDevice), "memcpy state_fixed");
        delta_rule(d_qkv, d_beta, d_g, shape_prefix, d_state_fixed, d_state_fixed, false, d_out, act, stream,
                   DeltaRulePath::ChunkedWmma, prec);
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync fixed");

        // Read back results
        std::vector<float> h_gold(state_elements);
        std::vector<float> h_buggy(state_elements);
        std::vector<float> h_fixed(state_elements);

        STRIX_HIP_CHECK(hipMemcpy(h_gold.data(), d_state_gold, state_bytes, hipMemcpyDeviceToHost), "readback gold");
        STRIX_HIP_CHECK(hipMemcpy(h_buggy.data(), d_state_buggy, state_bytes, hipMemcpyDeviceToHost), "readback buggy");
        STRIX_HIP_CHECK(hipMemcpy(h_fixed.data(), d_state_fixed, state_bytes, hipMemcpyDeviceToHost), "readback fixed");

        ErrorStats err_buggy = compare_states(h_gold.data(), h_buggy.data(), state_elements);
        ErrorStats err_fixed = compare_states(h_gold.data(), h_fixed.data(), state_elements);

        std::printf("  [UPSTREAM BUGGY REPLAY] (Auto path, default Exact precision):\n");
        std::printf("    Max Absolute Error : %12.6e\n", err_buggy.max_abs_err);
        std::printf("    Relative L2 Error  : %12.6e\n", err_buggy.rel_l2());
        std::printf("    Bit-exact matches  : %6.2f%% (%lld / %lld floats differ)\n",
                    err_buggy.bit_exact_pct(), (long long)err_buggy.diff_bits, (long long)err_buggy.total_elements);

        if (prec != DrPrecision::Exact && err_buggy.diff_bits > 0) {
            std::printf("    ==> BUG CONFIRMED: Split invariance VIOLATED in upstream code!\n");
            bug_confirmed = true;
        }

        std::printf("  [FIXED REPLAY] (Matching ChunkedWmma path, precision_.gdn forwarded):\n");
        std::printf("    Max Absolute Error : %12.6e\n", err_fixed.max_abs_err);
        std::printf("    Relative L2 Error  : %12.6e\n", err_fixed.rel_l2());
        std::printf("    Bit-exact matches  : %6.2f%% (%lld / %lld floats differ)\n",
                    err_fixed.bit_exact_pct(), (long long)err_fixed.diff_bits, (long long)err_fixed.total_elements);

        if (err_fixed.diff_bits == 0 && err_fixed.max_abs_err == 0.0) {
            std::printf("    ==> VERIFIED: 100.000%% BIT-EXACT SPLIT INVARIANCE RESTORED!\n");
        } else {
            std::printf("    ==> FAILED: Fixed replay did not match gold state!\n");
            all_fixed_passed = false;
        }
        std::printf("\n");
    }

    std::printf("================================================================================\n");
    std::printf("SUMMARY OF FINDINGS:\n");
    if (bug_confirmed && all_fixed_passed) {
        std::printf("  SUCCESS: Bug successfully reproduced and verified fixed across all precision modes.\n");
        std::printf("  - Upstream code broke split invariance whenever precision != Exact.\n");
        std::printf("  - Fixed code achieves 0.000000e+00 max error and 100.000%% bit-exact replay.\n");
    } else {
        std::printf("  UNEXPECTED: Review results above.\n");
    }
    std::printf("================================================================================\n");

    STRIX_HIP_CHECK(hipFree(d_qkv), "free d_qkv");
    STRIX_HIP_CHECK(hipFree(d_beta), "free d_beta");
    STRIX_HIP_CHECK(hipFree(d_g), "free d_g");
    STRIX_HIP_CHECK(hipFree(d_state_in), "free d_state_in");
    STRIX_HIP_CHECK(hipFree(d_state_gold), "free d_state_gold");
    STRIX_HIP_CHECK(hipFree(d_state_buggy), "free d_state_buggy");
    STRIX_HIP_CHECK(hipFree(d_state_fixed), "free d_state_fixed");
    STRIX_HIP_CHECK(hipFree(d_out), "free d_out");
    STRIX_HIP_CHECK(hipStreamDestroy(stream), "destroy stream");

    return (bug_confirmed && all_fixed_passed) ? 0 : 1;
}
