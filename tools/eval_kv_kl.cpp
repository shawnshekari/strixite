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

// Generates high-entropy technical paragraphs for arbitrary sequence length and seed
std::string make_technical_text(size_t approx_words, int seed = 0) {
    const std::vector<std::vector<std::string>> topic_corpora = {
        // Topic 0: Memory hierarchy & MMU
        {
            "The memory management unit translates virtual addresses through hierarchical page table walks. ",
            "Translation lookaside buffers cache linear mappings to minimize page miss latency across execution units. ",
            "Dual-subchannel LPDDR5X buses achieve high aggregate bandwidth through interleaved memory controllers. ",
            "Unified memory architectures arbitrate contention between latency-critical scalar cores and high-throughput vector compute. ",
            "Page table walk caches accelerate multi-level translation by bypassing redundant physical address queries. ",
            "Cache allocation technologies partition shared system-level cache capacity across isolated security domains. ",
            "Memory encryption engines apply hardware AES-XTS ciphers to protect unified physical frames from physical bus snooping. ",
            "Asynchronous writeback queues coalesce partial dirty cacheline transfers into full burst DRAM cycles. "
        },
        // Topic 1: Orthonormal Transforms & Linear Algebra
        {
            "Walsh-Hadamard orthonormal transforms diffuse high-variance channel activations across orthogonal dimensions. ",
            "Fast Walsh-Hadamard butterfly networks perform recursive Kronecker factorizations in logarithmic time depth. ",
            "Orthogonal state rotations preserve vector dot products and attention energy geometries without norm dilation. ",
            "Givens rotations eliminate subdiagonal matrix elements while maintaining numerical stability in least-squares systems. ",
            "Singular value decomposition isolates dominant eigenspaces under low-rank tensor approximation constraints. ",
            "Unitary matrix transformations preserve Hermitian inner products across complex-valued latent feature representations. ",
            "Spectral decomposition isolates principal variance components across multi-channel activation distributions. ",
            "Householder reflections map arbitrary coordinate vectors onto canonical basis axes with minimal rounding drift. "
        },
        // Topic 2: Speculative Decoding & Verification
        {
            "Speculative multi-token drafting validates candidate sequences using tree-structured verification steps. ",
            "Target model verification matches candidate draft tokens against top-1 argmax logits using greedy prefix bounds. ",
            "Asynchronous stream rollbacks prune uncommitted speculative branches without stalling hardware decode execution. ",
            "Draft acceptance ratios scale with contextual predictability and draft vocabulary alignment precision. ",
            "Residual speculative drafting heads predict sequential token offsets from final hidden layer activation states. ",
            "Confidence-gated drafting terminates speculative proposals when entropy exceeds verification acceptance thresholds. ",
            "Dynamic speculation depth tunes target token horizon according to instantaneous memory bandwidth headroom. ",
            "Prefix-sharing draft verification executes speculative candidate subtrees through unified batched forward kernels. "
        },
        // Topic 3: Recurrent Attention & DeltaNet
        {
            "Recurrent gated linear attention maintains constant-memory hidden states across arbitrarily deep sequences. ",
            "Delta rule recurrent updates subtract outer-product projections to eliminate redundant associative memory entries. ",
            "Chunked DeltaNet scan partitions sequence tokens into SRAM-sized blocks for parallel matrix multiplication. ",
            "Hardware matrix accelerators evaluate unit triangular state inversions using specialized wave-level accumulators. ",
            "Decay factors modulate temporal retention across recurrent hidden state transitions in linear attention layers. ",
            "Fused scan kernels integrate elementwise activation gates directly with inter-chunk state propagation buffers. ",
            "Norm folding operations fuse downstream layer normalization scales into recurrent projection weight matrices. ",
            "Linear state mixers update key-value associative memories through rank-one exterior differential modifications. "
        },
        // Topic 4: Compilers & SSA Graph Optimizations
        {
            "Dead code elimination operates over Static Single Assignment graphs with post-dominator frontier pruning. ",
            "Control dependence graphs identify conditional branch instructions governing execution of downstream basic blocks. ",
            "Sparse conditional constant propagation combines lattice value propagation with executable edge discovery. ",
            "Global value numbering collapses congruent expression subtrees into canonical equivalence class leaders. ",
            "Loop invariant code motion hoists computation outside loop preheaders when memory dependencies allow. ",
            "Register allocation resolves interference graphs using iterated register coalescing and priority-driven spilling. ",
            "Induction variable substitution replaces secondary loop counter updates with primary affine expressions. ",
            "Instruction scheduling reorders machine operations to hide pipeline execution latencies and bypass stalls. "
        },
        // Topic 5: Concurrency & Lock-Free Data Structures
        {
            "Lock-free single-producer single-consumer ring buffers coordinate producer and consumer threads using acquire-release fences. ",
            "Hardware destructive interference size alignment prevents cacheline bouncing across independent core private L1s. ",
            "Atomic compare-and-swap primitives guarantee forward progress across lockless multi-threaded queue operations. ",
            "Memory models prescribe partial visibility orderings between concurrent store releases and subsequent loads. ",
            "Sequential consistency imposes global total order across all observable memory operations across processor cores. ",
            "Epoch-based memory reclamation defers object destruction until all active reader threads retire from critical sections. ",
            "Hazard pointers protect concurrent shared pointer dereferences by publishing reader references in thread-local records. ",
            "Wait-free consensus protocols guarantee deterministic step bounds for concurrent operations regardless of thread contention. "
        },
        // Topic 6: Cryptography & Zero-Knowledge Proofs
        {
            "Zero-knowledge arguments of knowledge transform Rank-1 Constraint Systems into Quadratic Arithmetic Programs. ",
            "Pairing-friendly elliptic curves enable succinct bilinear group verifications with constant-time pairing operations. ",
            "Polynomial commitment schemes prove evaluation correctness over secret vectors without opening individual coefficients. ",
            "Fiat-Shamir heuristics transform interactive public-coin protocols into non-interactive cryptographic proofs. ",
            "Lattice-based cryptography constructs post-quantum secure signatures over hard Learning With Errors problems. ",
            "Arithmetic circuits decompose non-linear algebraic assertions into fan-in two addition and multiplication gates. ",
            "Lagrange interpolation determines minimal degree polynomials vanishing over structured multiplicative evaluation domains. ",
            "Number theoretic transforms perform high-degree polynomial multiplications in quasi-linear logarithmic time. "
        },
        // Topic 7: Distributed Systems & Consensus
        {
            "Distributed consensus protocols establish uniform replicated state machine progression despite network partitions. ",
            "Raft consensus elects term-scoped leaders through majority quorum voting and monotonically advancing term indices. ",
            "Vector clocks establish causal happens-before relations across asynchronous message-passing distributed nodes. ",
            "Byzantine fault tolerance maintains state agreement in networks containing arbitrary adversarial participants. ",
            "Two-phase commit coordinates atomic multi-shard transactions across heterogeneous database partition nodes. ",
            "Chubby lock services leverage Paxos consensus to coordinate coarse-grained distributed cluster leases. ",
            "Quorum intersection guarantees that overlap between write quorums and read quorums exposes latest state commits. ",
            "Gossip protocols disseminate cluster membership and node failure state updates via randomized epidemic broadcasts. "
        },
        // Topic 8: Database Engines & Storage
        {
            "B-link tree indexes allow concurrent node traversal without locking internal search path ancestors during split operations. ",
            "Write-ahead logging guarantees atomicity and durability by forcing log records to persistent storage prior to page modification. ",
            "Multi-version concurrency control isolates transaction read snapshots from concurrent uncommitted update transactions. ",
            "LSM tree compaction merges sorted run files to reclaim obsolete tombstones and maintain bounded lookup amplification. ",
            "Bloom filter bit vectors eliminate unnecessary disk page reads for absent keys during point query lookups. ",
            "Group commit protocols amortize rotational and NVMe sync latencies by batching synchronous log flushes. ",
            "Buffer pool management implements two-queue eviction policies to prevent sequential scans from polluting hot caches. ",
            "Adaptive radix trees optimize memory footprint and cache locality for variable-length lexicographical key sets. "
        }
    };

    const auto &snippets = topic_corpora[(size_t)(std::abs(seed)) % topic_corpora.size()];
    std::string text;
    size_t count = 0;
    size_t idx = 0;
    while (count < approx_words) {
        text += "[Seed " + std::to_string(seed) + ", Sec " + std::to_string(idx) + "]: " + snippets[idx % snippets.size()];
        count += 20;
        ++idx;
    }
    return text;
}

double calc_mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

double calc_stddev(const std::vector<double> &v) {
    if (v.size() <= 1) return 0.0;
    double m = calc_mean(v);
    double s2 = 0;
    for (double x : v) s2 += (x - m) * (x - m);
    return std::sqrt(s2 / (double)(v.size() - 1));
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights_path = "/models/strixite-swift15/weights.strixw";
    std::string ngram_path = "/models/strixite-swift15/ngram.table";
    std::string tokenizer_path = "/models/strixite-swift15/tokenizer.json";
    std::string output_json = "eval_kv_kl_results.json";
    int64_t capacity = 262144;
    int64_t chunk = 16384;
    int num_samples = 9;
    std::vector<int64_t> eval_positions = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 199608, 262144};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--ngram" && i + 1 < argc) ngram_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--capacity" && i + 1 < argc) capacity = std::atoll(argv[++i]);
        else if (arg == "--chunk" && i + 1 < argc) chunk = std::atoll(argv[++i]);
        else if (arg == "--samples" && i + 1 < argc) num_samples = std::atoi(argv[++i]);
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
    std::cout << "Tokenizer: " << tokenizer_path << "\n";
    std::cout << "Samples:   " << num_samples << " (independent technical corpora)\n\n";

    // 1. Load tokenizer
    Tokenizer tok(tokenizer_path);

    // 2. Load model weights (immutable, shared across all samples)
    std::cout << "[*] Loading model weights into device memory...\n";
    Qwen4ExpModel model(weights_path, ngram_path, kernels::Act::BF16, /*allow_truncated=*/false,
                        /*ngram_cache_rows=*/8388608, /*yarn_factor=*/1.0f, WeightPlacement::Device);
    std::cout << "[+] Model loaded successfully.\n";

    PrecisionOptions precision;

    const std::vector<KvFormat> formats = {KvFormat::Act, KvFormat::Q8, KvFormat::Q8H};

    struct PosStatsAccum {
        std::vector<double> kl_q8;
        std::vector<double> kl_q8h;
        std::vector<double> mae_q8;
        std::vector<double> mae_q8h;
        std::vector<double> cos_q8;
        std::vector<double> cos_q8h;
        std::vector<int> top1_q8;
        std::vector<int> top1_q8h;
        int top1_match_q8 = 0;
        int top1_match_q8h = 0;
        int q8h_wins = 0;
        int q8_wins = 0;
        int ties = 0;
    };
    std::map<int64_t, PosStatsAccum> accum_by_pos;

    // Run over all samples
    for (int s_idx = 0; s_idx < num_samples; ++s_idx) {
        std::cout << "\n>>> [Sample " << (s_idx + 1) << " / " << num_samples << "] Encoding technical corpus seed=" << s_idx << "...\n";
        std::string prompt_text = make_technical_text(capacity * 2, s_idx);
        std::vector<int32_t> all_tokens = tok.encode(prompt_text);
        if ((int64_t)all_tokens.size() > capacity) {
            all_tokens.resize((size_t)capacity);
        }
        std::cout << "[+] Encoded " << all_tokens.size() << " tokens.\n";

        std::vector<int64_t> valid_positions;
        for (int64_t p : eval_positions) {
            if (p <= (int64_t)all_tokens.size()) valid_positions.push_back(p);
        }

        std::map<KvFormat, std::vector<std::vector<float>>> recorded_logits;

        for (KvFormat fmt : formats) {
            std::string fmt_name = kv_format_name(fmt);
            std::cout << "  [*] Format: " << fmt_name << "... " << std::flush;
            auto t_fmt0 = std::chrono::steady_clock::now();

            int64_t actual_chunk = std::min(chunk, capacity);
            auto backend = std::make_unique<Qwen4ExpBackend>(model, capacity, actual_chunk, /*use_mtp=*/false, 0,
                                                             precision, fmt);

            int64_t curr_pos = 0;
            for (int64_t p : valid_positions) {
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
                recorded_logits[fmt].push_back(std::move(logits));
            }

            backend.reset();
            STRIX_HIP_CHECK(hipDeviceSynchronize(), "backend teardown");
            auto t_fmt1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t_fmt1 - t_fmt0).count();
            std::cout << "done in " << std::fixed << std::setprecision(1) << s << "s\n";
        }

        // Compute sample metrics
        std::cout << "\n    --- Sample " << (s_idx + 1) << " Individual Checkpoint Metrics ---\n";
        std::printf("    %-10s | %-14s | %-14s | %-10s | %-10s | %-6s\n", "Position", "KL (Q8)", "KL (Q8H)", "MAE (Q8)", "MAE (Q8H)", "Winner");
        std::cout << "    " << std::string(70, '-') << "\n";

        for (size_t i = 0; i < valid_positions.size(); ++i) {
            int64_t pos = valid_positions[i];
            const auto &ref = recorded_logits[KvFormat::Act][i];
            const auto &q8 = recorded_logits[KvFormat::Q8][i];
            const auto &q8h = recorded_logits[KvFormat::Q8H][i];

            int64_t vocab = (int64_t)ref.size();
            STRIX_CHECK(vocab > 0 && ref.size() == q8.size() && ref.size() == q8h.size(),
                        "eval_kv_kl: logits size mismatch");

            LogitStats s_q8 = compare_logits(ref.data(), q8.data(), vocab);
            LogitStats s_q8h = compare_logits(ref.data(), q8h.data(), vocab);

            auto &acc = accum_by_pos[pos];
            acc.kl_q8.push_back(s_q8.kl_div);
            acc.kl_q8h.push_back(s_q8h.kl_div);
            acc.mae_q8.push_back(s_q8.mean_diff);
            acc.mae_q8h.push_back(s_q8h.mean_diff);
            acc.cos_q8.push_back(s_q8.cosine_sim);
            acc.cos_q8h.push_back(s_q8h.cosine_sim);
            acc.top1_q8.push_back(s_q8.top1_match ? 1 : 0);
            acc.top1_q8h.push_back(s_q8h.top1_match ? 1 : 0);
            if (s_q8.top1_match) acc.top1_match_q8++;
            if (s_q8h.top1_match) acc.top1_match_q8h++;

            std::string win = (s_q8h.kl_div < s_q8.kl_div) ? "Q8H" : (s_q8.kl_div < s_q8h.kl_div ? "Q8" : "TIE");
            if (win == "Q8H") acc.q8h_wins++;
            else if (win == "Q8") acc.q8_wins++;
            else acc.ties++;

            std::printf("    %-10lld | %-14.6e | %-14.6e | %-10.4f | %-10.4f | %-6s\n",
                        (long long)pos, s_q8.kl_div, s_q8h.kl_div, s_q8.mean_diff, s_q8h.mean_diff, win.c_str());
        }
        std::cout << "    " << std::string(70, '-') << "\n";
    }

    // 3. Mathematical Comparison & Report across all samples
    std::cout << "\n" << std::string(105, '=') << "\n";
    std::cout << "           EXACT KL DIVERGENCE & LOGIT ERROR REPORT (N = " << num_samples << " SAMPLES VS BF16 BASELINE)\n";
    std::cout << std::string(105, '=') << "\n";

    std::printf("%-10s | %-24s | %-24s | %-12s | %-12s | %-10s\n",
                "Position", "D_KL (Q8) Mean ± Std", "D_KL (Q8H) Mean ± Std", "MAE (Q8)", "MAE (Q8H)", "Q8H Wins");
    std::cout << std::string(105, '-') << "\n";

    std::vector<std::string> json_entries;

    auto format_vector_json = [](const std::vector<double> &vec) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            oss << vec[i] << (i + 1 < vec.size() ? ", " : "");
        }
        oss << "]";
        return oss.str();
    };

    auto format_int_vector_json = [](const std::vector<int> &vec) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            oss << vec[i] << (i + 1 < vec.size() ? ", " : "");
        }
        oss << "]";
        return oss.str();
    };

    for (int64_t pos : eval_positions) {
        if (accum_by_pos.find(pos) == accum_by_pos.end()) continue;
        const auto &acc = accum_by_pos[pos];

        double kl_q8_m = calc_mean(acc.kl_q8), kl_q8_s = calc_stddev(acc.kl_q8);
        double kl_q8h_m = calc_mean(acc.kl_q8h), kl_q8h_s = calc_stddev(acc.kl_q8h);
        double mae_q8_m = calc_mean(acc.mae_q8);
        double mae_q8h_m = calc_mean(acc.mae_q8h);
        double cos_q8_m = calc_mean(acc.cos_q8);
        double cos_q8h_m = calc_mean(acc.cos_q8h);

        char buf_q8[64], buf_q8h[64];
        std::snprintf(buf_q8, sizeof(buf_q8), "%.3e ± %.2e", kl_q8_m, kl_q8_s);
        std::snprintf(buf_q8h, sizeof(buf_q8h), "%.3e ± %.2e", kl_q8h_m, kl_q8h_s);

        char win_buf[32];
        std::snprintf(win_buf, sizeof(win_buf), "%d / %d", acc.q8h_wins, num_samples);

        std::printf("%-10lld | %-24s | %-24s | %-12.4f | %-12.4f | %-10s\n",
                    (long long)pos, buf_q8, buf_q8h, mae_q8_m, mae_q8h_m, win_buf);

        std::ostringstream ss;
        ss << "    {\n"
           << "      \"position\": " << pos << ",\n"
           << "      \"kl_divergence_q8_mean\": " << kl_q8_m << ",\n"
           << "      \"kl_divergence_q8_std\": " << kl_q8_s << ",\n"
           << "      \"kl_divergence_q8h_mean\": " << kl_q8h_m << ",\n"
           << "      \"kl_divergence_q8h_std\": " << kl_q8h_s << ",\n"
           << "      \"individual_kl_q8\": " << format_vector_json(acc.kl_q8) << ",\n"
           << "      \"individual_kl_q8h\": " << format_vector_json(acc.kl_q8h) << ",\n"
           << "      \"mae_q8_mean\": " << mae_q8_m << ",\n"
           << "      \"mae_q8h_mean\": " << mae_q8h_m << ",\n"
           << "      \"individual_mae_q8\": " << format_vector_json(acc.mae_q8) << ",\n"
           << "      \"individual_mae_q8h\": " << format_vector_json(acc.mae_q8h) << ",\n"
           << "      \"cosine_sim_q8_mean\": " << cos_q8_m << ",\n"
           << "      \"cosine_sim_q8h_mean\": " << cos_q8h_m << ",\n"
           << "      \"individual_cos_q8\": " << format_vector_json(acc.cos_q8) << ",\n"
           << "      \"individual_cos_q8h\": " << format_vector_json(acc.cos_q8h) << ",\n"
           << "      \"top1_match_q8_rate\": " << (double)acc.top1_match_q8 / num_samples << ",\n"
           << "      \"top1_match_q8h_rate\": " << (double)acc.top1_match_q8h / num_samples << ",\n"
           << "      \"individual_top1_q8\": " << format_int_vector_json(acc.top1_q8) << ",\n"
           << "      \"individual_top1_q8h\": " << format_int_vector_json(acc.top1_q8h) << ",\n"
           << "      \"q8h_wins\": " << acc.q8h_wins << ",\n"
           << "      \"q8_wins\": " << acc.q8_wins << "\n"
           << "    }";
        json_entries.push_back(ss.str());
    }

    std::cout << std::string(105, '-') << "\n";

    // Write JSON output
    std::ofstream out(output_json);
    if (out.good()) {
        out << "{\n  \"timestamp\": " << std::time(nullptr) << ",\n  \"samples\": " << num_samples << ",\n  \"results\": [\n";
        for (size_t i = 0; i < json_entries.size(); ++i) {
            out << json_entries[i] << (i + 1 < json_entries.size() ? ",\n" : "\n");
        }
        out << "  ]\n}\n";
        std::cout << "[+] Saved detailed metrics to " << output_json << "\n";
    }

    return 0;
}
