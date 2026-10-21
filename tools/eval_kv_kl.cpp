// tools/eval_kv_kl.cpp: Measures exact next-token KL divergence and logit error
// between unquantized BF16 KV cache and quantized KV caches (Q8, Q8H) at various
// context positions (from 1k up to 65k+ tokens).
//
// Calculates:
//   D_KL(P_bf16 || P_q8)   vs   D_KL(P_bf16 || P_q8h)
// where P = softmax(logits / T).
//
// Also measures max logit difference, mean absolute error (MAE), cosine similarity,
// and Top-1 token agreement.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/tokenizer.hpp"
#include "serve/log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

struct LogitStats {
    double kl_div = 0.0;
    double max_diff = 0.0;
    double mean_diff = 0.0;
    double cosine_sim = 0.0;
    bool top1_match = false;
    int32_t top1_ref = 0;
    int32_t top1_test = 0;
};

LogitStats compare_logits(const float *ref, const float *test, int64_t n) {
    float max_ref = -std::numeric_limits<float>::infinity();
    float max_test = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        if (ref[i] > max_ref) max_ref = ref[i];
        if (test[i] > max_test) max_test = test[i];
    }

    double sum_exp_ref = 0.0;
    double sum_exp_test = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum_exp_ref += std::exp((double)ref[i] - (double)max_ref);
        sum_exp_test += std::exp((double)test[i] - (double)max_test);
    }
    double lse_ref = (double)max_ref + std::log(sum_exp_ref);
    double lse_test = (double)max_test + std::log(sum_exp_test);

    double kl = 0.0;
    double max_d = 0.0;
    double sum_d = 0.0;
    double dot = 0.0, norm_ref = 0.0, norm_test = 0.0;

    int32_t top1_r = 0, top1_t = 0;
    float best_r = -std::numeric_limits<float>::infinity();
    float best_t = -std::numeric_limits<float>::infinity();

    for (int64_t i = 0; i < n; ++i) {
        double log_p = (double)ref[i] - lse_ref;
        double log_q = (double)test[i] - lse_test;
        double p = std::exp(log_p);

        if (p > 1e-12) {
            kl += p * (log_p - log_q);
        }

        double diff = std::abs((double)ref[i] - (double)test[i]);
        if (diff > max_d) max_d = diff;
        sum_d += diff;

        dot += (double)ref[i] * (double)test[i];
        norm_ref += (double)ref[i] * (double)ref[i];
        norm_test += (double)test[i] * (double)test[i];

        if (ref[i] > best_r) { best_r = ref[i]; top1_r = (int32_t)i; }
        if (test[i] > best_t) { best_t = test[i]; top1_t = (int32_t)i; }
    }

    double cos_sim = (norm_ref > 0 && norm_test > 0) ? dot / (std::sqrt(norm_ref) * std::sqrt(norm_test)) : 0.0;

    LogitStats res;
    res.kl_div = std::max(0.0, kl);
    res.max_diff = max_d;
    res.mean_diff = sum_d / (double)n;
    res.cosine_sim = cos_sim;
    res.top1_match = (top1_r == top1_t);
    res.top1_ref = top1_r;
    res.top1_test = top1_t;
    return res;
}

// Generates high-entropy technical paragraphs for arbitrary sequence length
std::string make_technical_text(size_t approx_words) {
    const std::vector<std::string> snippets = {
        "The memory management unit translates virtual addresses through hierarchical page table walks. ",
        "Walsh-Hadamard orthonormal transforms diffuse high-variance channel activations across orthogonal dimensions. ",
        "Speculative multi-token drafting validates candidate sequences using tree-structured verification steps. ",
        "Recurrent gated linear attention maintains constant-memory hidden states across arbitrarily deep sequences. ",
        "Dual-subchannel LPDDR5X buses achieve high aggregate bandwidth through interleaved memory controllers. ",
        "Dead code elimination operates over Static Single Assignment graphs with post-dominator frontier pruning. ",
        "Lock-free single-producer single-consumer ring buffers coordinate producer and consumer threads using acquire-release fences. ",
        "Zero-knowledge arguments of knowledge transform Rank-1 Constraint Systems into Quadratic Arithmetic Programs. ",
        "Cache coherency controllers broadcast invalidation messages across shared interconnects under the MESI protocol. ",
        "FlashAttention decomposes query-key projections into SRAM-sized tiles with online numerical softmax stabilization. "
    };
    std::string text;
    size_t count = 0;
    size_t idx = 0;
    while (count < approx_words) {
        text += "[Section " + std::to_string(idx) + "]: " + snippets[idx % snippets.size()];
        count += 20;
        ++idx;
    }
    return text;
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights_path = "/models/strixite-swift15/weights.strixw";
    std::string ngram_path = "/models/strixite-swift15/ngram.table";
    std::string tokenizer_path = "/models/strixite-swift15/tokenizer.json";
    std::string output_json = "eval_kv_kl_results.json";
    int64_t capacity = 65536;
    int64_t chunk = 16384;
    std::vector<int64_t> eval_positions = {1024, 2048, 4096, 8192, 16384, 32768, 65536};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--ngram" && i + 1 < argc) ngram_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--capacity" && i + 1 < argc) capacity = std::atoll(argv[++i]);
        else if (arg == "--chunk" && i + 1 < argc) chunk = std::atoll(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_json = argv[++i];
        else if (arg == "--positions" && i + 1 < argc) {
            std::string pos_str = argv[++i];
            eval_positions.clear();
            std::stringstream ss(pos_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) eval_positions.push_back(std::atoll(item.c_str()));
            }
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "       STRIXITE KV CACHE MATHEMATICAL KL DIVERGENCE EVALUATOR\n";
    std::cout << "========================================================================\n";
    std::cout << "Weights:   " << weights_path << "\n";
    std::cout << "Capacity:  " << capacity << " tokens\n";
    std::cout << "Tokenizer: " << tokenizer_path << "\n\n";

    // 1. Load tokenizer and encode prompt
    Tokenizer tok(tokenizer_path);
    std::cout << "[*] Encoding prompt text...\n";
    std::string prompt_text = make_technical_text(capacity * 2);
    std::vector<int32_t> all_tokens = tok.encode(prompt_text);
    if ((int64_t)all_tokens.size() > capacity) {
        all_tokens.resize((size_t)capacity);
    }
    std::cout << "[+] Encoded " << all_tokens.size() << " tokens.\n";

    // Evaluation evaluation checkpoints
    std::vector<int64_t> valid_positions;
    for (int64_t p : eval_positions) {
        if (p <= (int64_t)all_tokens.size()) valid_positions.push_back(p);
    }

    // 2. Load model weights (immutable, shared)
    std::cout << "[*] Loading model weights into device memory...\n";
    Qwen4ExpModel model(weights_path, ngram_path, kernels::Act::BF16, /*allow_truncated=*/false,
                        /*ngram_cache_rows=*/8388608, /*yarn_factor=*/1.0f, WeightPlacement::Device);
    std::cout << "[+] Model loaded successfully.\n";

    PrecisionOptions precision;

    const std::vector<KvFormat> formats = {KvFormat::Act, KvFormat::Q8, KvFormat::Q8H};
    std::map<KvFormat, std::vector<std::vector<float>>> recorded_logits;

    for (KvFormat fmt : formats) {
        std::string fmt_name = kv_format_name(fmt);
        std::cout << "\n[*] Running forward passes for format: " << fmt_name << "...\n";

        // Create fresh backend for this format
        auto backend = std::make_unique<Qwen4ExpBackend>(model, capacity, chunk, /*use_mtp=*/false, 0,
                                                         precision, fmt);

        int64_t curr_pos = 0;
        for (int64_t p : valid_positions) {
            std::cout << "    Advancing to position " << p << "..." << std::flush;
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> logits;

            while (curr_pos < p) {
                int64_t step = std::min(p - curr_pos, chunk);
                bool is_last = (curr_pos + step == p);
                std::vector<int32_t> slice(all_tokens.begin() + curr_pos, all_tokens.begin() + curr_pos + step);
                auto res = backend->forward(slice, is_last ? int64_t{1} : int64_t{0});
                if (is_last) {
                    logits = std::move(res);
                }
                curr_pos += step;
            }

            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::cout << " done in " << std::fixed << std::setprecision(2) << s << "s.\n";

            recorded_logits[fmt].push_back(std::move(logits));
        }

        // Backend destruction frees device buffers for the next format
        backend.reset();
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "backend teardown");
    }

    // 3. Mathematical Comparison & Report
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "           EXACT KL DIVERGENCE & LOGIT ERROR REPORT (VS BF16 BASELINE)\n";
    std::cout << std::string(90, '=') << "\n";

    std::printf("%-10s | %-16s | %-16s | %-12s | %-12s | %-8s\n",
                "Position", "KL Div (Q8)", "KL Div (Q8H)", "MAE (Q8)", "MAE (Q8H)", "Winner");
    std::cout << std::string(90, '-') << "\n";

    std::vector<std::string> json_entries;
    int q8h_wins = 0, q8_wins = 0;

    for (size_t i = 0; i < valid_positions.size(); ++i) {
        int64_t pos = valid_positions[i];
        const auto &ref = recorded_logits[KvFormat::Act][i];
        const auto &q8 = recorded_logits[KvFormat::Q8][i];
        const auto &q8h = recorded_logits[KvFormat::Q8H][i];

        int64_t vocab = (int64_t)ref.size();
        STRIX_CHECK(vocab > 0 && ref.size() == q8.size() && ref.size() == q8h.size(),
                    "eval_kv_kl: logits size mismatch (ref=", ref.size(), ", q8=", q8.size(), ", q8h=", q8h.size(), ")");
        LogitStats s_q8 = compare_logits(ref.data(), q8.data(), vocab);
        LogitStats s_q8h = compare_logits(ref.data(), q8h.data(), vocab);

        std::string winner = (s_q8h.kl_div < s_q8.kl_div) ? "Q8H" : (s_q8.kl_div < s_q8h.kl_div ? "Q8" : "TIE");
        if (winner == "Q8H") ++q8h_wins;
        else if (winner == "Q8") ++q8_wins;

        std::printf("%-10lld | %-16.6e | %-16.6e | %-12.4f | %-12.4f | %-8s\n",
                    (long long)pos, s_q8.kl_div, s_q8h.kl_div, s_q8.mean_diff, s_q8h.mean_diff, winner.c_str());

        std::ostringstream ss;
        ss << "    {\n"
           << "      \"position\": " << pos << ",\n"
           << "      \"kl_divergence_q8\": " << s_q8.kl_div << ",\n"
           << "      \"kl_divergence_q8h\": " << s_q8h.kl_div << ",\n"
           << "      \"mae_q8\": " << s_q8.mean_diff << ",\n"
           << "      \"mae_q8h\": " << s_q8h.mean_diff << ",\n"
           << "      \"max_diff_q8\": " << s_q8.max_diff << ",\n"
           << "      \"max_diff_q8h\": " << s_q8h.max_diff << ",\n"
           << "      \"cosine_sim_q8\": " << s_q8.cosine_sim << ",\n"
           << "      \"cosine_sim_q8h\": " << s_q8h.cosine_sim << ",\n"
           << "      \"top1_match_q8\": " << (s_q8.top1_match ? "true" : "false") << ",\n"
           << "      \"top1_match_q8h\": " << (s_q8h.top1_match ? "true" : "false") << ",\n"
           << "      \"winner\": \"" << winner << "\"\n"
           << "    }";
        json_entries.push_back(ss.str());
    }

    std::cout << std::string(90, '-') << "\n";
    std::cout << "Summary: Q8H won " << q8h_wins << " positions, Q8 won " << q8_wins << " positions.\n";

    // Write JSON output
    std::ofstream out(output_json);
    if (out.good()) {
        out << "{\n  \"timestamp\": " << std::time(nullptr) << ",\n  \"results\": [\n";
        for (size_t i = 0; i < json_entries.size(); ++i) {
            out << json_entries[i] << (i + 1 < json_entries.size() ? ",\n" : "\n");
        }
        out << "  ]\n}\n";
        std::cout << "[+] Saved detailed metrics to " << output_json << "\n";
    }

    return 0;
}
