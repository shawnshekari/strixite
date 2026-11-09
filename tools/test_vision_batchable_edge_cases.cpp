#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/server_config.hpp"
#include "serve/log.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vision_request.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/seq_backend.hpp"

#include <stb_image_write.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>

using namespace strix;

// Helper to create a valid PNG in memory (256x256 RGB solid colour)
static std::vector<uint8_t> make_test_png(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    std::vector<uint8_t> rgb((size_t)(w * h * 3));
    for (size_t i = 0; i < (size_t)(w * h); ++i) {
        rgb[3 * i + 0] = r;
        rgb[3 * i + 1] = g;
        rgb[3 * i + 2] = b;
    }
    std::vector<uint8_t> out;
    auto write_cb = [](void *context, void *data, int size) {
        auto *vec = reinterpret_cast<std::vector<uint8_t> *>(context);
        const auto *bytes = reinterpret_cast<const uint8_t *>(data);
        vec->insert(vec->end(), bytes, bytes + size);
    };
    int ok = stbi_write_png_to_func(write_cb, &out, w, h, 3, rgb.data(), w * 3);
    STRIX_CHECK(ok && !out.empty(), "stbi_write_png_to_func failed");
    return out;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const std::string conf_path = argc > 1 ? argv[1] : "/home/cheblan/strix-bench.conf";
    printf("======================================================================\n");
    printf(">>> TEST: VISION BATCHABLE & ROPE_DELTA EDGE CASES <<<\n");
    printf("======================================================================\n");
    printf("Config: %s\n", conf_path.c_str());
    fflush(stdout);

    ServerSettings settings(strix_server_options("/models"));
    settings.load({"--config", conf_path});

    const std::string weights = settings.text("weights");
    const std::string ngram = settings.text("ngram");
    const std::string tok_path = settings.text("tokenizer");
    const std::string vision_path = "/models/Thinking-Cap-Qwen-3.8-FN-strixw/vision.strixw";
    const std::string preproc_cfg = "/models/Thinking-Cap-Qwen-3.8-FN-strixw/preprocessor_config.json";
    const std::string model_cfg = "/models/Thinking-Cap-Qwen-3.8-FN-strixw/config.json";

    const int64_t capacity = 65536;
    const int64_t chunk = 16384;
    const int64_t total_slots = 4;
    const int64_t kv_generations = 128;

    printf("Loading Tokenizer: %s\n", tok_path.c_str());
    Tokenizer tokenizer(tok_path);

    printf("Loading Qwen4ExpModel with vision tower: %s\n", vision_path.c_str());
    PrecisionOptions prec;
    prec.gdn = parse_gdn_precision(settings.text("gdn-precision"));
    prec.attention = parse_attention_precision(settings.text("attention-precision"));
    Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, 8388608, 1.0f, WeightPlacement::Device, vision_path);
    STRIX_CHECK(model.has_vision(), "Model must have vision tower loaded");

    printf("Initializing Qwen4ExpBackend with %ld slots at capacity %ld...\n",
           (long)total_slots, (long)capacity);
    Qwen4ExpBackend backend(model, capacity, chunk, true, 0, prec, KvFormat::Q8, total_slots, kv_generations);

    printf("Loading Vision Preprocessing Config: %s\n", preproc_cfg.c_str());
    vision::ServedVision sv = vision::load_vision_config(preproc_cfg, model_cfg, 8192);

    int passed = 0;
    int failed = 0;

    auto test_assert = [&](bool cond, const char *name) {
        if (cond) {
            printf("[PASS] %s\n", name);
            passed++;
        } else {
            printf("[FAIL] %s\n", name);
            failed++;
        }
        fflush(stdout);
    };

    // ======================================================================
    // EDGE CASE 1: Pure text slot before any vision
    // ======================================================================
    printf("\n--- Edge Case 1: Pure text slot state before vision ---\n");
    backend.reset(0);
    test_assert(backend.rope_delta(0) == 0, "Slot 0 initial rope_delta == 0");
    test_assert(backend.batchable(0) == true, "Slot 0 initial batchable() == true");

    std::vector<int32_t> text_prompt = {151644, 872, 198, 2610, 525, 264, 10925, 1782, 151645, 198, 151646, 198};
    backend.forward_rows(0, text_prompt, 1, false, 0);
    test_assert(backend.rope_delta(0) == 0, "Slot 0 post-text-prefill rope_delta == 0");
    test_assert(backend.batchable(0) == true, "Slot 0 post-text-prefill batchable() == true");

    // ======================================================================
    // EDGE CASE 2: In-flight 3D positions (Vision prefill setup)
    // ======================================================================
    printf("\n--- Edge Case 2: In-flight 3D positions (pending3 non-empty) ---\n");
    backend.reset(0);
    // Setting 1 position triple {0, 0, 0} matching uniform start
    backend.session().set_positions({0, 0, 0});
    test_assert(backend.batchable(0) == false, "Slot 0 with pending3 positions batchable() == false");
    // Clear pending positions
    backend.session().set_positions({});
    test_assert(backend.batchable(0) == true, "Slot 0 after clearing pending3 batchable() == true");

    // ======================================================================
    // EDGE CASE 3: Completed vision forward sets rope_delta > 0, pending3 empty
    // ======================================================================
    printf("\n--- Edge Case 3: Completed vision forward sets rope_delta > 0 ---\n");
    backend.reset(0);
    // Create 256x256 test image
    std::vector<uint8_t> png_bytes = make_test_png(256, 256, 220, 50, 50);
    std::vector<vision::ParsedImage> parsed_images = {{png_bytes, "image/png"}};
    std::vector<vision::ServedImage> served_images = vision::prepare_images(std::move(parsed_images), sv);

    std::vector<int64_t> counts;
    for (const auto &im : served_images) counts.push_back(im.merged_tokens);
    std::vector<int32_t> prompt_with_image = {151644, 872, 198, sv.image_token_id, 151645, 198, 151646, 198};
    prompt_with_image = vision::expand_image_pads(prompt_with_image, sv.image_token_id, counts);

    vision::VisionPromptPlan plan = vision::plan_vision_prompt(prompt_with_image, served_images, sv.cfg,
                                                             sv.image_token_id, backend.max_positions());

    PrefillImage prefill_im{std::move(served_images[0].prefill), served_images[0].row_start, served_images[0].rows};

    printf("Executing forward_image_prefill for Slot 0 (%zu tokens, %zu pos3 values)...\n",
           prompt_with_image.size(), plan.pos3.size());
    LogitRows im_logits = backend.forward_image_prefill(prompt_with_image, prefill_im, plan.pos3, 1, false, 0);
    test_assert(im_logits.rows == 1, "Image forward returns 1 logits row");
    test_assert(!im_logits.full.empty(), "Image forward logits are non-empty");

    const int64_t post_vis_delta = backend.rope_delta(0);
    printf("Slot 0 post-vision rope_delta: %ld\n", (long)post_vis_delta);
    test_assert(post_vis_delta != 0, "Slot 0 post-vision rope_delta != 0 (negative due to 2D image packing)");
    test_assert(backend.batchable(0) == true, "CRITICAL BUGFIX: Slot 0 post-vision batchable() == true");

    // ======================================================================
    // EDGE CASE 4: Mixed Batched Verify (Vision Slot 0 + Text Slot 1)
    // ======================================================================
    printf("\n--- Edge Case 4: Mixed Batched Verify (Vision Slot 0 + Text Slot 1) ---\n");
    // Slot 1: pure text prompt
    backend.reset(1);
    backend.forward_rows(1, text_prompt, 1, false, 0);
    test_assert(backend.rope_delta(1) == 0, "Slot 1 pure text rope_delta == 0");
    test_assert(backend.batchable(1) == true, "Slot 1 pure text batchable() == true");

    // Both slots verify together: Slot 0 has rope_delta > 0, Slot 1 has rope_delta == 0
    std::vector<LmBackend::VerifyRun> mixed_runs = {
        {0, {1234}, nullptr},
        {1, {5678}, nullptr}
    };
    std::vector<LogitRows> mixed_out;
    bool verify_threw = false;
    try {
        mixed_out = backend.verify_batch(mixed_runs, false, 0);
    } catch (const std::exception &e) {
        printf("verify_batch threw exception: %s\n", e.what());
        verify_threw = true;
    }
    test_assert(!verify_threw, "Mixed verify_batch (Vision + Text) executed without throwing");
    test_assert(mixed_out.size() == 2, "Mixed verify_batch returned 2 results");
    backend.keep_verify(0, 1, 0);
    backend.keep_verify(1, 1, 0);

    // Solo verify for comparison to check for cross-talk
    std::vector<LogitRows> solo0_out = backend.verify_batch({{0, {1234}}}, false, 0);
    backend.keep_verify(0, 1, 0);

    std::vector<LogitRows> solo1_out = backend.verify_batch({{1, {5678}}}, false, 0);
    backend.keep_verify(1, 1, 0);

    test_assert(!solo0_out[0].full.empty() && !solo1_out[0].full.empty(),
                "Solo verify for both slots produced non-empty logits");

    // Check that logits on Slot 1 are finite and sane
    bool s1_finite = true;
    for (float x : mixed_out[1].full) if (!std::isfinite(x)) { s1_finite = false; break; }
    test_assert(s1_finite, "Slot 1 logits in mixed batch are all finite");

    bool s0_finite = true;
    for (float x : mixed_out[0].full) if (!std::isfinite(x)) { s0_finite = false; break; }
    test_assert(s0_finite, "Slot 0 logits in mixed batch are all finite");

    // ======================================================================
    // EDGE CASE 5: Multi-step decode/verify loop under mixed concurrency
    // ======================================================================
    printf("\n--- Edge Case 5: 10 consecutive decode/verify steps under mixed concurrency ---\n");
    int32_t cur_tok0 = 1234;
    int32_t cur_tok1 = 5678;
    bool steps_ok = true;
    for (int step = 0; step < 10; ++step) {
        std::vector<LmBackend::VerifyRun> loop_runs = {
            {0, {cur_tok0}, nullptr},
            {1, {cur_tok1}, nullptr}
        };
        try {
            auto out = backend.verify_batch(loop_runs, false, 0);
            backend.keep_verify(0, 1, 0);
            backend.keep_verify(1, 1, 0);
            cur_tok0 = (cur_tok0 + 17) % 50000;
            cur_tok1 = (cur_tok1 + 23) % 50000;
        } catch (const std::exception &e) {
            printf("Step %d failed with exception: %s\n", step, e.what());
            steps_ok = false;
            break;
        }
    }
    test_assert(steps_ok, "10 consecutive mixed-verify steps completed cleanly");

    // ======================================================================
    // EDGE CASE 6: Slot Reset after Vision
    // ======================================================================
    printf("\n--- Edge Case 6: Slot Reset restores clean text state ---\n");
    backend.reset(0);
    test_assert(backend.rope_delta(0) == 0, "Slot 0 reset restores rope_delta == 0");
    test_assert(backend.batchable(0) == true, "Slot 0 reset restores batchable() == true");
    backend.forward_rows(0, text_prompt, 1, false, 0);
    test_assert(backend.rope_delta(0) == 0, "Slot 0 post-reset text forward keeps rope_delta == 0");

    // ======================================================================
    // EDGE CASE 7: GpuArbiter / SeqBackend Integration (verify_alone metric)
    // ======================================================================
    printf("\n--- Edge Case 7: GpuArbiter batching behavior with post-vision slot ---\n");
    // Setup vision again on slot 0
    backend.reset(0);
    std::vector<vision::ServedImage> s_images2 = vision::prepare_images(
        {vision::ParsedImage{make_test_png(256, 256, 100, 200, 100), "image/png"}}, sv);
    vision::VisionPromptPlan plan2 = vision::plan_vision_prompt(prompt_with_image, s_images2, sv.cfg,
                                                               sv.image_token_id, backend.max_positions());
    PrefillImage prefill_im2{std::move(s_images2[0].prefill), s_images2[0].row_start, s_images2[0].rows};
    backend.forward_image_prefill(prompt_with_image, prefill_im2, plan2.pos3, 1, false, 0);

    // Setup text on slot 1
    backend.reset(1);
    backend.forward_rows(1, text_prompt, 1, false, 0);

    GpuArbiter arbiter(backend);
    SeqBackend seq0(arbiter, 0);
    SeqBackend seq1(arbiter, 1);

    // Both slots submit verify via SeqBackend simultaneously on separate threads
    std::thread t0([&] {
        seq0.forward_verify_rows({2001}, 1, false, 0);
        seq0.keep_verify();
    });
    std::thread t1([&] {
        seq1.forward_verify_rows({3001}, 1, false, 0);
        seq1.keep_verify();
    });
    t0.join();
    t1.join();

    GpuArbiter::Stats arb_stats = arbiter.stats();
    printf("Arbiter Stats: verify_batches = %ld, verify_runs = %ld, verify_alone = %ld\n",
           (long)arb_stats.verify_batches, (long)arb_stats.verify_runs, (long)arb_stats.verify_alone);

    test_assert(arb_stats.verify_alone == 0,
                "GpuArbiter verify_alone == 0 (post-vision slot was NOT forced to run alone!)");
    test_assert(arb_stats.verify_batches >= 1, "GpuArbiter verify_batches >= 1");

    // ======================================================================
    // FINAL SUMMARY
    // ======================================================================
    printf("\n======================================================================\n");
    printf(">>> TEST SUMMARY: %d PASSED, %d FAILED <<<\n", passed, failed);
    printf("======================================================================\n");

    return failed == 0 ? 0 : 1;
}
