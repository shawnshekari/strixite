#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/server_config.hpp"
#include "serve/log.hpp"
#include "serve/tokenizer.hpp"
#include "serve/qwen4exp_backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include <numeric>
#include <string>
#include <iomanip>

using namespace strix;

struct StepData {
    int round = 0;
    std::vector<int32_t> drafts;
    int64_t accepted_count = 0;
    int32_t emitted_token = 0;
    std::vector<float> row0_logits; // Row 0 of base verify (full vocab)
    int32_t top_token = 0;
    float top_logit = 0.0f;
};

struct TaskData {
    int task_id = 0;
    std::string prompt_text;
    std::vector<int32_t> prompt_tokens;
    std::vector<float> prefill_logits;
    int32_t prefill_top_token = 0;
    float prefill_top_logit = 0.0f;
    std::vector<StepData> steps;
    std::vector<int32_t> all_generated_tokens;
};

struct SlotResult {
    int slot_id = 0;
    std::vector<TaskData> tasks;
};

struct ExperimentResult {
    int num_slots = 0;
    double total_ms = 0;
    int64_t total_tokens = 0;
    std::vector<SlotResult> slots;
};

static int32_t get_argmax(const float *logits, int64_t vocab, float *out_val = nullptr) {
    int32_t b = 0;
    float bv = logits[0];
    for (int32_t i = 1; i < vocab; ++i) {
        if (logits[i] > bv) {
            bv = logits[i];
            b = i;
        }
    }
    if (out_val) *out_val = bv;
    return b;
}

static ExperimentResult run_experiment(Qwen4ExpBackend &backend,
                                      int64_t vocab,
                                      const Tokenizer &tokenizer,
                                      const std::vector<std::string> &prompt_texts,
                                      int num_slots,
                                      int kRoundsPerTask,
                                      int64_t kMtpDraft,
                                      float kMtpMargin) {
    printf("\n======================================================================\n");
    printf(">>> RUNNING EXPERIMENT: %d SLOT(S) (%d PROMPTS, %d ROUNDS/TASK) <<<\n",
           num_slots, (int)prompt_texts.size(), kRoundsPerTask);
    printf("======================================================================\n");
    fflush(stdout);

    ExperimentResult exp;
    exp.num_slots = num_slots;
    exp.slots.resize(num_slots);
    for (int s = 0; s < num_slots; ++s) {
        exp.slots[s].slot_id = s;
        exp.slots[s].tasks.resize(prompt_texts.size());
    }

    // Determine max draft steps for this slot count to obey kMaxLogits = 16
    // draft_max <= max_verify_rows / num_slots - 1
    const int64_t draft_max = std::min(kMtpDraft, std::max<int64_t>(1, backend.max_verify_rows() / num_slots - 1));
    printf("Engine draft budget for %d slot(s): max_draft = %ld (user requested %ld, margin = %.2f)\n",
           num_slots, (long)draft_max, (long)kMtpDraft, kMtpMargin);
    fflush(stdout);

    auto t_start = std::chrono::steady_clock::now();

    for (size_t task_idx = 0; task_idx < prompt_texts.size(); ++task_idx) {
        const std::string &p_text = prompt_texts[task_idx];
        std::vector<int32_t> p_tokens = tokenizer.encode(p_text);
        printf("\n--- Task %zu / %zu (Prompt: \"%s...\", %zu tokens) ---\n",
               task_idx, prompt_texts.size(),
               p_text.substr(0, std::min<size_t>(p_text.size(), 40)).c_str(),
               p_tokens.size());
        fflush(stdout);

        for (int s = 0; s < num_slots; ++s) {
            exp.slots[s].tasks[task_idx].task_id = (int)task_idx;
            exp.slots[s].tasks[task_idx].prompt_text = p_text;
            exp.slots[s].tasks[task_idx].prompt_tokens = p_tokens;
            backend.reset(s);
        }

        // 1. Prefill
        std::vector<int32_t> cur_tokens(num_slots);
        for (int s = 0; s < num_slots; ++s) {
            LogitRows pr = backend.forward_rows(s, p_tokens, 1, false, 0);
            float top_v = 0;
            int32_t top_tok = get_argmax(pr.full.data(), vocab, &top_v);
            exp.slots[s].tasks[task_idx].prefill_logits = pr.full;
            exp.slots[s].tasks[task_idx].prefill_top_token = top_tok;
            exp.slots[s].tasks[task_idx].prefill_top_logit = top_v;
            cur_tokens[s] = top_tok;
        }

        // 2. Generation Loop (kRoundsPerTask speculative rounds)
        for (int r = 0; r < kRoundsPerTask; ++r) {
            StepData step;
            step.round = r;

            // (a) Parallel MTP Draft across active slots
            std::vector<std::vector<int32_t>> drafts(num_slots);
            for (int s = 0; s < num_slots; ++s) drafts[s] = {cur_tokens[s]};

            std::vector<bool> drafting_active(num_slots, true);
            for (int dstep = 0; dstep < draft_max; ++dstep) {
                std::vector<LmBackend::MtpDraftRun> runs;
                for (int s = 0; s < num_slots; ++s) {
                    if (drafting_active[s]) runs.push_back({s, drafts[s].back(), dstep});
                }
                if (runs.empty()) break;

                std::vector<Top2> picks = backend.forward_mtp_top2_batch(runs);
                for (size_t i = 0; i < runs.size(); ++i) {
                    int s = runs[i].seq;
                    Top2 p = picks[i];
                    if (p.nan || p.best_v - p.second_v < kMtpMargin) {
                        drafting_active[s] = false;
                    } else {
                        drafts[s].push_back(p.best);
                    }
                }
            }

            // (b) Batched Verify
            std::vector<LmBackend::VerifyRun> vruns(num_slots);
            for (int s = 0; s < num_slots; ++s) {
                vruns[s].seq = s;
                vruns[s].ids = drafts[s];
            }
            std::vector<LogitRows> vouts = backend.verify_batch(vruns, false, 0);

            // (c) Acceptance & State update
            for (int s = 0; s < num_slots; ++s) {
                StepData sd;
                sd.round = r;
                for (size_t di = 1; di < drafts[s].size(); ++di) {
                    sd.drafts.push_back(drafts[s][di]);
                }

                // Row 0 of verify corresponds to cur_tokens[s]
                const float *row0_ptr = vouts[s].full.data();
                sd.row0_logits.assign(row0_ptr, row0_ptr + vocab);
                float top_v = 0;
                sd.top_token = get_argmax(row0_ptr, vocab, &top_v);
                sd.top_logit = top_v;

                // Greedy speculative verification
                const int64_t n_drafts = drafts[s].size() - 1;
                int64_t accepted = 0;
                int32_t next_id = -1;

                for (int64_t j = 0; j < n_drafts; ++j) {
                    const float *row_j = vouts[s].full.data() + j * vocab;
                    int32_t pred = get_argmax(row_j, vocab);
                    if (pred == drafts[s][j + 1]) {
                        accepted++;
                        exp.slots[s].tasks[task_idx].all_generated_tokens.push_back(pred);
                    } else {
                        next_id = pred;
                        break;
                    }
                }

                if (accepted == n_drafts) {
                    // All accepted: next token is sampled from the last row
                    const float *last_row = vouts[s].full.data() + n_drafts * vocab;
                    next_id = get_argmax(last_row, vocab);
                    backend.keep_verify(s, n_drafts + 1, 0);
                } else {
                    backend.keep_verify(s, accepted + 1, 0);
                }

                sd.accepted_count = accepted;
                sd.emitted_token = next_id;
                exp.slots[s].tasks[task_idx].all_generated_tokens.push_back(next_id);
                exp.slots[s].tasks[task_idx].steps.push_back(sd);

                cur_tokens[s] = next_id;
                exp.total_tokens += (accepted + 1);
            }
        }
        printf("Completed %d rounds for Task %zu across %d slot(s). Total generated: %zu tokens.\n",
               kRoundsPerTask, task_idx, num_slots,
               exp.slots[0].tasks[task_idx].all_generated_tokens.size());
        fflush(stdout);
    }

    auto t_end = std::chrono::steady_clock::now();
    exp.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("\nExperiment (%d slots): %ld tokens in %.2f ms -> %.2f aggregate tok/s\n",
           num_slots, (long)exp.total_tokens, exp.total_ms,
           (double)exp.total_tokens / (exp.total_ms / 1000.0));
    fflush(stdout);

    return exp;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const std::string conf_path = argc > 1 ? argv[1] : "/home/cheblan/strix-bench.conf";
    printf("======================================================================\n");
    printf(">>> STRIXITE CONCURRENCY & LOGIT IDENTITY BENCHMARK <<<\n");
    printf("======================================================================\n");
    printf("Config: %s\n", conf_path.c_str());
    fflush(stdout);

    ServerSettings settings(strix_server_options("/models"));
    settings.load({"--config", conf_path});

    const std::string weights = settings.text("weights");
    const std::string ngram = settings.text("ngram");
    const std::string tok_path = settings.text("tokenizer");
    const int64_t capacity = 65536;
    const int64_t chunk = 16384;
    const int64_t total_slots = 4;
    const int64_t kv_generations = 128;

    printf("Loading Tokenizer: %s\n", tok_path.c_str());
    Tokenizer tokenizer(tok_path);

    printf("Loading Qwen4ExpModel: weights=%s\n", weights.c_str());
    PrecisionOptions prec;
    prec.gdn = parse_gdn_precision(settings.text("gdn-precision"));
    prec.attention = parse_attention_precision(settings.text("attention-precision"));
    Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, 8388608, 1.0f);

    printf("Initializing Qwen4ExpBackend with %ld slots at capacity %ld...\n",
           (long)total_slots, (long)capacity);
    Qwen4ExpBackend backend(model, capacity, chunk, true, 0, prec, KvFormat::Q8, total_slots, kv_generations);

    const int64_t vocab = model.dims().vocab;

    // 4 standard prompts
    std::vector<std::string> prompt_texts = {
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nWrite a short haiku about code.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nExplain bubble sort in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nWhat is 17 times 19?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    };

    const int kRoundsPerTask = 10;
    const int64_t kMtpDraft = 5;
    const float kMtpMargin = 1.5f;

    // RUN 1: 1 slot (4 prompts sequentially)
    ExperimentResult res1 = run_experiment(backend, vocab, tokenizer, prompt_texts, 1, kRoundsPerTask, kMtpDraft, kMtpMargin);

    // RUN 2: 2 slots (each slot executes the 4 prompts sequentially, 8 tasks total)
    ExperimentResult res2 = run_experiment(backend, vocab, tokenizer, prompt_texts, 2, kRoundsPerTask, kMtpDraft, kMtpMargin);

    // RUN 3: 4 slots (each slot executes the 4 prompts, 16 tasks total)
    ExperimentResult res4 = run_experiment(backend, vocab, tokenizer, prompt_texts, 4, kRoundsPerTask, kMtpDraft, kMtpMargin);

    // ======================================================================
    // VERIFICATION & ANALYSIS REPORT
    // ======================================================================
    printf("\n\n");
    printf("======================================================================\n");
    printf("==================== DETAILED VERIFICATION REPORT ====================\n");
    printf("======================================================================\n");

    auto compare_logits = [](const std::vector<float> &a, const std::vector<float> &b) {
        float max_d = 0.0f;
        double sum_d = 0.0;
        size_t mismatches = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            float d = std::fabs(a[i] - b[i]);
            if (d > max_d) max_d = d;
            sum_d += d;
            if (a[i] != b[i]) mismatches++;
        }
        return std::make_tuple(max_d, (float)(sum_d / a.size()), mismatches);
    };

    // 1. INTRA-RUN PARALLEL CONCURRENCY (Slot 0 vs Slot 1 in 2-slot; Slot 0 vs 1,2,3 in 4-slot)
    printf("\n############################################################\n");
    printf("### PART 1: INTRA-RUN BIT IDENTITY (CROSS-SLOT DETERMINISM) ###\n");
    printf("############################################################\n");

    printf("\n--- Run 2 (2 concurrent slots): Slot 0 vs Slot 1 ---\n");
    for (size_t t = 0; t < prompt_texts.size(); ++t) {
        auto [p_max, p_mean, p_mism] = compare_logits(res2.slots[0].tasks[t].prefill_logits,
                                                      res2.slots[1].tasks[t].prefill_logits);
        printf("Task %zu: Prefill logits max_diff = %f (mismatches: %zu / %ld)\n",
               t, p_max, p_mism, (long)vocab);

        float max_step_d = 0.0f;
        size_t total_step_mism = 0;
        for (int r = 0; r < kRoundsPerTask; ++r) {
            auto [s_max, s_mean, s_mism] = compare_logits(res2.slots[0].tasks[t].steps[r].row0_logits,
                                                          res2.slots[1].tasks[t].steps[r].row0_logits);
            if (s_max > max_step_d) max_step_d = s_max;
            total_step_mism += s_mism;
        }
        printf("Task %zu: Verify row0 logits max_diff across %d rounds = %f (mismatches: %zu)\n",
               t, kRoundsPerTask, max_step_d, total_step_mism);

        bool tok_match = (res2.slots[0].tasks[t].all_generated_tokens == res2.slots[1].tasks[t].all_generated_tokens);
        printf("Task %zu: Generated token sequence match: %s (%zu tokens)\n",
               t, tok_match ? "100% BIT-EXACT MATCH" : "MISMATCH",
               res2.slots[0].tasks[t].all_generated_tokens.size());
    }

    printf("\n--- Run 3 (4 concurrent slots): Slot 0 vs Slots 1, 2, 3 ---\n");
    for (size_t t = 0; t < prompt_texts.size(); ++t) {
        printf("Task %zu:\n", t);
        for (int s = 1; s < 4; ++s) {
            auto [p_max, p_mean, p_mism] = compare_logits(res4.slots[0].tasks[t].prefill_logits,
                                                          res4.slots[s].tasks[t].prefill_logits);
            float max_step_d = 0.0f;
            size_t total_step_mism = 0;
            for (int r = 0; r < kRoundsPerTask; ++r) {
                auto [s_max, s_mean, s_mism] = compare_logits(res4.slots[0].tasks[t].steps[r].row0_logits,
                                                              res4.slots[s].tasks[t].steps[r].row0_logits);
                if (s_max > max_step_d) max_step_d = s_max;
                total_step_mism += s_mism;
            }
            bool tok_match = (res4.slots[0].tasks[t].all_generated_tokens == res4.slots[s].tasks[t].all_generated_tokens);
            printf("  Slot 0 vs Slot %d: Prefill max_diff = %f, Verify max_diff = %f, Tokens: %s\n",
                   s, p_max, max_step_d, tok_match ? "EXACT MATCH" : "MISMATCH");
        }
    }

    // 2. INTER-RUN LOGIT COMPARISON (1 Slot vs 2 Slots vs 4 Slots)
    printf("\n############################################################\n");
    printf("### PART 2: INTER-RUN COMPARISON (1-SLOT vs 2-SLOTS vs 4-SLOTS) ###\n");
    printf("############################################################\n");

    for (size_t t = 0; t < prompt_texts.size(); ++t) {
        printf("\n============================================================\n");
        printf(">>> TASK %zu LOGIT ANALYSIS <<<\n", t);
        printf("============================================================\n");

        // Prefill comparison
        auto [pf_1v2_max, pf_1v2_mean, pf_1v2_mism] = compare_logits(res1.slots[0].tasks[t].prefill_logits,
                                                                      res2.slots[0].tasks[t].prefill_logits);
        auto [pf_1v4_max, pf_1v4_mean, pf_1v4_mism] = compare_logits(res1.slots[0].tasks[t].prefill_logits,
                                                                      res4.slots[0].tasks[t].prefill_logits);
        printf("Prefill Logits Comparison:\n");
        printf("  1-Slot vs 2-Slots: max_diff = %f, mismatches = %zu\n", pf_1v2_max, pf_1v2_mism);
        printf("  1-Slot vs 4-Slots: max_diff = %f, mismatches = %zu\n", pf_1v4_max, pf_1v4_mism);

        printf("\nRaw Prefill Logits for Task %zu (first 5 vocabulary IDs):\n", t);
        printf("  ID     | 1-Slot Logit    | 2-Slots Logit   | 4-Slots Logit   | Delta (1 vs 4)\n");
        printf("  -------|-----------------|-----------------|-----------------|---------------\n");
        for (int i = 0; i < 5; ++i) {
            float l1 = res1.slots[0].tasks[t].prefill_logits[i];
            float l2 = res2.slots[0].tasks[t].prefill_logits[i];
            float l4 = res4.slots[0].tasks[t].prefill_logits[i];
            printf("  %-6d | %15.6f | %15.6f | %15.6f | %13.6e\n",
                   i, l1, l2, l4, std::fabs(l1 - l4));
        }
        printf("  Top Token: 1-Slot = %d (%.4f), 2-Slots = %d (%.4f), 4-Slots = %d (%.4f)\n",
               res1.slots[0].tasks[t].prefill_top_token, res1.slots[0].tasks[t].prefill_top_logit,
               res2.slots[0].tasks[t].prefill_top_token, res2.slots[0].tasks[t].prefill_top_logit,
               res4.slots[0].tasks[t].prefill_top_token, res4.slots[0].tasks[t].prefill_top_logit);

        // Verify Step 0 row 0 comparison
        auto [v0_1v2_max, v0_1v2_mean, v0_1v2_mism] = compare_logits(res1.slots[0].tasks[t].steps[0].row0_logits,
                                                                      res2.slots[0].tasks[t].steps[0].row0_logits);
        auto [v0_1v4_max, v0_1v4_mean, v0_1v4_mism] = compare_logits(res1.slots[0].tasks[t].steps[0].row0_logits,
                                                                      res4.slots[0].tasks[t].steps[0].row0_logits);
        auto [v0_2v4_max, v0_2v4_mean, v0_2v4_mism] = compare_logits(res2.slots[0].tasks[t].steps[0].row0_logits,
                                                                      res4.slots[0].tasks[t].steps[0].row0_logits);
        printf("\nVerify Step 0 Row 0 Logits Comparison:\n");
        printf("  1-Slot vs 2-Slots: max_diff = %f, mean_diff = %e (mismatches: %zu)\n",
               v0_1v2_max, v0_1v2_mean, v0_1v2_mism);
        printf("  1-Slot vs 4-Slots: max_diff = %f, mean_diff = %e (mismatches: %zu)\n",
               v0_1v4_max, v0_1v4_mean, v0_1v4_mism);
        printf("  2-Slots vs 4-Slots: max_diff = %f, mean_diff = %e (mismatches: %zu)\n",
               v0_2v4_max, v0_2v4_mean, v0_2v4_mism);

        printf("\nRaw Verify Step 0 Logits for Task %zu (first 5 vocabulary IDs):\n", t);
        printf("  ID     | 1-Slot Logit    | 2-Slots Logit   | 4-Slots Logit   | Delta (1 vs 4)\n");
        printf("  -------|-----------------|-----------------|-----------------|---------------\n");
        for (int i = 0; i < 5; ++i) {
            float l1 = res1.slots[0].tasks[t].steps[0].row0_logits[i];
            float l2 = res2.slots[0].tasks[t].steps[0].row0_logits[i];
            float l4 = res4.slots[0].tasks[t].steps[0].row0_logits[i];
            printf("  %-6d | %15.6f | %15.6f | %15.6f | %13.6e\n",
                   i, l1, l2, l4, std::fabs(l1 - l4));
        }

        // Generated text tokens
        const auto &toks1 = res1.slots[0].tasks[t].all_generated_tokens;
        const auto &toks2 = res2.slots[0].tasks[t].all_generated_tokens;
        const auto &toks4 = res4.slots[0].tasks[t].all_generated_tokens;

        printf("\nGenerated Tokens Sequence (first 10 tokens):\n");
        printf("  1-Slot : ");
        for (size_t i = 0; i < std::min<size_t>(10, toks1.size()); ++i) printf("%d ", toks1[i]);
        printf("\n  2-Slots: ");
        for (size_t i = 0; i < std::min<size_t>(10, toks2.size()); ++i) printf("%d ", toks2[i]);
        printf("\n  4-Slots: ");
        for (size_t i = 0; i < std::min<size_t>(10, toks4.size()); ++i) printf("%d ", toks4[i]);
        printf("\n");

        // Divergence point
        size_t div_1v2 = toks1.size();
        for (size_t i = 0; i < std::min(toks1.size(), toks2.size()); ++i) {
            if (toks1[i] != toks2[i]) { div_1v2 = i; break; }
        }
        size_t div_1v4 = toks1.size();
        for (size_t i = 0; i < std::min(toks1.size(), toks4.size()); ++i) {
            if (toks1[i] != toks4[i]) { div_1v4 = i; break; }
        }
        printf("  First token mismatch 1-Slot vs 2-Slots: index %zu\n", div_1v2);
        printf("  First token mismatch 1-Slot vs 4-Slots: index %zu\n", div_1v4);
    }

    printf("\n======================================================================\n");
    printf(">>> OVERALL THROUGHPUT SUMMARY <<<\n");
    printf("======================================================================\n");
    printf("1 Slot  (4 tasks):  %4ld tokens in %8.2f ms -> %6.2f tok/s\n",
           (long)res1.total_tokens, res1.total_ms,
           (double)res1.total_tokens / (res1.total_ms / 1000.0));
    printf("2 Slots (8 tasks):  %4ld tokens in %8.2f ms -> %6.2f tok/s (%.2fx vs 1 slot)\n",
           (long)res2.total_tokens, res2.total_ms,
           (double)res2.total_tokens / (res2.total_ms / 1000.0),
           ((double)res2.total_tokens / (res2.total_ms / 1000.0)) / ((double)res1.total_tokens / (res1.total_ms / 1000.0)));
    printf("4 Slots (16 tasks): %4ld tokens in %8.2f ms -> %6.2f tok/s (%.2fx vs 1 slot)\n",
           (long)res4.total_tokens, res4.total_ms,
           (double)res4.total_tokens / (res4.total_ms / 1000.0),
           ((double)res4.total_tokens / (res4.total_ms / 1000.0)) / ((double)res1.total_tokens / (res1.total_ms / 1000.0)));
    printf("======================================================================\n");

    return 0;
}
