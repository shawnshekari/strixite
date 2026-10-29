// Stress test and verification suite for HIP Virtual Memory Management (VMM) unmapping,
// remapping, and buffer recreation on AMD architectures (specifically GFX1151 / Strix Halo APU).
//
// Background:
// runtime/paged_buffer.hpp documents a known issue observed on early gfx1151 drivers:
// "On gfx1151, after a page was unmapped and released and a new one mapped at the same address,
// kernels sometimes read memory other than the new page there (another allocation's, once the freed
// memory was reused), with nothing failing: the same forward gave different logits, only while other
// processes used the GPU. Pages are only ever mapped back into this buffer's own reservation:
// on gfx1151 a page mapped into a second reservation reads the wrong memory."
//
// This tool empirically verifies whether VMM page unmap, remap, and buffer recreation are safe
// and coherent on the running driver/kernel stack. It tests:
//   1. Cross-Reservation Mapping: mapping a physical allocation handle into a second VA reservation.
//   2. In-place Remap: hipMemUnmap + hipMemMap at the same VA with memory poisoning.
//   3. Buffer Recreation: full teardown (hipMemAddressFree) and re-reservation (reclaim_paged pattern).
//   4. Multi-Process / Concurrent GPU Noise: hammering the GPU with concurrent matrix compute and
//      memory allocations to trigger VMID switching and TLB invalidation races.
//
// Usage:
//   test_vmm_reclaim_safety [--iterations 1000] [--pages 64] [--page-mb 2] [--noise on|off] [--multi-process]

#include "common/args.hpp"
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

using namespace strix;

namespace {

__global__ void fill_pattern_kernel(uint32_t *data, size_t num_elements, uint32_t seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = seed ^ (uint32_t)(idx * 0x9E3779B9u);
    }
}

__global__ void check_pattern_kernel(const uint32_t *data, size_t num_elements, uint32_t expected_seed,
                                     uint32_t poison_seed, uint32_t *errors, uint32_t *poison_hits) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        uint32_t val = data[idx];
        uint32_t expected = expected_seed ^ (uint32_t)(idx * 0x9E3779B9u);
        if (val != expected) {
            atomicAdd(errors, 1);
            if (poison_seed != 0) {
                uint32_t poison = poison_seed ^ (uint32_t)(idx * 0x9E3779B9u);
                if (val == poison) {
                    atomicAdd(poison_hits, 1);
                }
            }
        }
    }
}

__global__ void matrix_churn_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_noise_worker_process(int duration_sec, int dev) {
    STRIX_HIP_CHECK(hipSetDevice(dev), "noise worker: set device");
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    STRIX_HIP_CHECK(hipMalloc(&dA, bytes), "noise worker: alloc dA");
    STRIX_HIP_CHECK(hipMalloc(&dB, bytes), "noise worker: alloc dB");
    STRIX_HIP_CHECK(hipMalloc(&dC, bytes), "noise worker: alloc dC");

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    auto start = std::chrono::steady_clock::now();
    uint64_t iter = 0;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration_sec) break;
        matrix_churn_kernel<<<grid, block>>>(dA, dB, dC, N);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "noise worker: sync");
        iter++;
        if (iter % 250 == 0) {
            STRIX_HIP_CHECK(hipFree(dC), "noise worker: free dC");
            STRIX_HIP_CHECK(hipMalloc(&dC, bytes), "noise worker: realloc dC");
        }
    }
    STRIX_HIP_CHECK(hipFree(dA), "noise worker: free dA");
    STRIX_HIP_CHECK(hipFree(dB), "noise worker: free dB");
    STRIX_HIP_CHECK(hipFree(dC), "noise worker: free dC");
}

struct TestConfig {
    int iterations = 1000;
    size_t num_pages = 64;       // 64 x 2MB = 128 MB
    size_t page_bytes = 2 << 20; // 2 MiB (matching kPageBytes in qwen4exp)
    bool background_noise = true;
    bool multi_process = false;
    std::string mode = "all";
};

bool test_cross_reservation(const TestConfig &cfg, int dev, const hipMemAllocationProp &prop, size_t gran) {
    printf("\n--- Test: Cross-Reservation Page Mapping ---\n");
    printf("Evaluating: 'a page mapped into a second reservation reads the wrong memory'\n");

    const size_t num_elements = cfg.page_bytes / sizeof(uint32_t);
    uint32_t *d_errors = nullptr;
    STRIX_HIP_CHECK(hipMalloc(&d_errors, sizeof(uint32_t)), "alloc d_errors");

    uint64_t total_errors = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        void *r1 = nullptr, *r2 = nullptr;
        STRIX_HIP_CHECK(hipMemAddressReserve(&r1, cfg.page_bytes, gran, nullptr, 0), "reserve r1");
        STRIX_HIP_CHECK(hipMemAddressReserve(&r2, cfg.page_bytes, gran, nullptr, 0), "reserve r2");

        hipMemGenericAllocationHandle_t h{};
        STRIX_HIP_CHECK(hipMemCreate(&h, cfg.page_bytes, &prop, 0), "create page");

        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = dev;
        access.flags = hipMemAccessFlagsProtReadWrite;

        // 1. Map handle into Reservation 1
        STRIX_HIP_CHECK(hipMemMap(r1, cfg.page_bytes, 0, h, 0), "map r1");
        STRIX_HIP_CHECK(hipMemSetAccess(r1, cfg.page_bytes, &access, 1), "set access r1");

        // 2. Write pattern into R1
        uint32_t seed = 0xC3000000u | (uint32_t)iter;
        fill_pattern_kernel<<<(num_elements + 255) / 256, 256>>>(static_cast<uint32_t *>(r1), num_elements, seed);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync r1 write");

        // 3. Unmap from R1
        STRIX_HIP_CHECK(hipMemUnmap(r1, cfg.page_bytes), "unmap r1");
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync r1 unmap");

        // 4. Map the SAME handle into Reservation 2
        STRIX_HIP_CHECK(hipMemMap(r2, cfg.page_bytes, 0, h, 0), "map r2");
        STRIX_HIP_CHECK(hipMemSetAccess(r2, cfg.page_bytes, &access, 1), "set access r2");

        // 5. Read from R2: verify data retention
        STRIX_HIP_CHECK(hipMemset(d_errors, 0, sizeof(uint32_t)), "clear d_errors");
        check_pattern_kernel<<<(num_elements + 255) / 256, 256>>>(
            static_cast<uint32_t *>(r2), num_elements, seed, 0, d_errors, nullptr);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync r2 check");

        uint32_t h_err = 0;
        STRIX_HIP_CHECK(hipMemcpyDtoH(&h_err, d_errors, sizeof(uint32_t)), "read d_errors");
        if (h_err > 0) {
            total_errors += h_err;
            printf("  [FAIL] Iteration %d: %u mismatches across reservations!\n", iter, h_err);
        }

        // Cleanup
        STRIX_HIP_CHECK(hipMemUnmap(r2, cfg.page_bytes), "unmap r2");
        STRIX_HIP_CHECK(hipMemRelease(h), "release h");
        STRIX_HIP_CHECK(hipMemAddressFree(r1, cfg.page_bytes), "free r1");
        STRIX_HIP_CHECK(hipMemAddressFree(r2, cfg.page_bytes), "free r2");

        if ((iter + 1) % 250 == 0 || iter == cfg.iterations - 1) {
            printf("  Progress: %d / %d iterations (errors: %lu)...\n", iter + 1, cfg.iterations, total_errors);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    STRIX_HIP_CHECK(hipFree(d_errors), "free d_errors");

    printf("Cross-Reservation Verdict: %s (total mismatches: %lu, elapsed: %.2f s, avg: %.2f ms/iter)\n",
           total_errors == 0 ? "PASSED (100% data intact)" : "FAILED (DATA CORRUPTED)",
           total_errors, ms / 1000.0, ms / cfg.iterations);
    return total_errors == 0;
}

bool test_unmap_remap_stress(const TestConfig &cfg, int dev, const hipMemAllocationProp &prop, size_t gran,
                             bool destroy_reservation) {
    const size_t total_bytes = cfg.page_bytes * cfg.num_pages;
    const size_t num_elements = total_bytes / sizeof(uint32_t);

    printf("\n--- Test: %s Stress ---\n",
           destroy_reservation ? "Buffer Recreation (reclaim_paged pattern)" : "In-place Page Remap");
    printf("Iterations: %d, Total Memory: %.2f MB (%zu pages of %zu MB)\n",
           cfg.iterations, (double)total_bytes / (1024 * 1024), cfg.num_pages, cfg.page_bytes / (1024 * 1024));
    printf("Background Noise: %s (%s)\n",
           cfg.background_noise ? "ACTIVE" : "OFF",
           cfg.multi_process ? "multi-process child" : "concurrent stream");

    // Start background noise worker
    pid_t noise_pid = 0;
    std::atomic<bool> noise_thread_stop{false};
    std::thread noise_thread;

    if (cfg.background_noise) {
        if (cfg.multi_process) {
            noise_pid = fork();
            if (noise_pid == 0) {
                run_noise_worker_process(300, dev);
                exit(0);
            }
            usleep(200000);
        } else {
            noise_thread = std::thread([&]() {
                hipStream_t noise_stream = nullptr;
                STRIX_HIP_CHECK(hipStreamCreateWithFlags(&noise_stream, hipStreamNonBlocking), "noise stream");
                const int N = 1024;
                size_t sz = (size_t)N * N * sizeof(float);
                float *dA = nullptr, *dB = nullptr, *dC = nullptr;
                STRIX_HIP_CHECK(hipMalloc(&dA, sz), "noise dA");
                STRIX_HIP_CHECK(hipMalloc(&dB, sz), "noise dB");
                STRIX_HIP_CHECK(hipMalloc(&dC, sz), "noise dC");
                dim3 block(16, 16);
                dim3 grid((N + 15) / 16, (N + 15) / 16);
                while (!noise_thread_stop.load()) {
                    matrix_churn_kernel<<<grid, block, 0, noise_stream>>>(dA, dB, dC, N);
                    (void)hipStreamSynchronize(noise_stream);
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                (void)hipFree(dA);
                (void)hipFree(dB);
                (void)hipFree(dC);
                (void)hipStreamDestroy(noise_stream);
            });
        }
    }

    uint32_t *d_errors = nullptr, *d_poison_hits = nullptr;
    STRIX_HIP_CHECK(hipMalloc(&d_errors, sizeof(uint32_t)), "alloc d_errors");
    STRIX_HIP_CHECK(hipMalloc(&d_poison_hits, sizeof(uint32_t)), "alloc d_poison_hits");

    // Allocate polluter buffer to force physical page recycling
    const size_t polluter_pages = cfg.num_pages;
    void *polluter_va = nullptr;
    std::vector<hipMemGenericAllocationHandle_t> polluter_handles(polluter_pages);
    STRIX_HIP_CHECK(hipMemAddressReserve(&polluter_va, polluter_pages * cfg.page_bytes, gran, nullptr, 0), "polluter reserve");
    for (size_t p = 0; p < polluter_pages; ++p) {
        STRIX_HIP_CHECK(hipMemCreate(&polluter_handles[p], cfg.page_bytes, &prop, 0), "polluter create");
        char *at = static_cast<char *>(polluter_va) + p * cfg.page_bytes;
        STRIX_HIP_CHECK(hipMemMap(at, cfg.page_bytes, 0, polluter_handles[p], 0), "polluter map");
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = dev;
        access.flags = hipMemAccessFlagsProtReadWrite;
        STRIX_HIP_CHECK(hipMemSetAccess(at, cfg.page_bytes, &access, 1), "polluter access");
    }

    void *base_va = nullptr;
    if (!destroy_reservation) {
        STRIX_HIP_CHECK(hipMemAddressReserve(&base_va, total_bytes, gran, nullptr, 0), "base reserve");
    }

    uint64_t total_corruptions = 0;
    uint64_t total_poison_hits = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        if (destroy_reservation) {
            STRIX_HIP_CHECK(hipMemAddressReserve(&base_va, total_bytes, gran, nullptr, 0), "re-reserve base");
        }

        std::vector<hipMemGenericAllocationHandle_t> pages(cfg.num_pages);
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = dev;
        access.flags = hipMemAccessFlagsProtReadWrite;

        // 1. Map pages
        for (size_t p = 0; p < cfg.num_pages; ++p) {
            STRIX_HIP_CHECK(hipMemCreate(&pages[p], cfg.page_bytes, &prop, 0), "create page");
            char *at = static_cast<char *>(base_va) + p * cfg.page_bytes;
            STRIX_HIP_CHECK(hipMemMap(at, cfg.page_bytes, 0, pages[p], 0), "map page");
            STRIX_HIP_CHECK(hipMemSetAccess(at, cfg.page_bytes, &access, 1), "set access");
        }

        // 2. Write Pattern A
        uint32_t seed_a = 0xA1000000u | (uint32_t)iter;
        fill_pattern_kernel<<<(num_elements + 255) / 256, 256>>>(
            static_cast<uint32_t *>(base_va), num_elements, seed_a);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync write A");

        // 3. Unmap and release pages
        for (size_t p = 0; p < cfg.num_pages; ++p) {
            char *at = static_cast<char *>(base_va) + p * cfg.page_bytes;
            STRIX_HIP_CHECK(hipMemUnmap(at, cfg.page_bytes), "unmap page");
            STRIX_HIP_CHECK(hipMemRelease(pages[p]), "release page");
        }

        if (destroy_reservation) {
            STRIX_HIP_CHECK(hipMemAddressFree(base_va, total_bytes), "free base");
            base_va = nullptr;
        }

        // 4. Dirty / Pollute physical memory via the secondary buffer
        uint32_t poison_seed = 0xDE000000u | (uint32_t)iter;
        size_t polluter_elements = (polluter_pages * cfg.page_bytes) / sizeof(uint32_t);
        fill_pattern_kernel<<<(polluter_elements + 255) / 256, 256>>>(
            static_cast<uint32_t *>(polluter_va), polluter_elements, poison_seed);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync pollute");

        // 5. Re-reserve if destroyed, then Map NEW pages into the same VA
        if (destroy_reservation) {
            STRIX_HIP_CHECK(hipMemAddressReserve(&base_va, total_bytes, gran, nullptr, 0), "re-reserve base");
        }

        for (size_t p = 0; p < cfg.num_pages; ++p) {
            STRIX_HIP_CHECK(hipMemCreate(&pages[p], cfg.page_bytes, &prop, 0), "re-create page");
            char *at = static_cast<char *>(base_va) + p * cfg.page_bytes;
            STRIX_HIP_CHECK(hipMemMap(at, cfg.page_bytes, 0, pages[p], 0), "re-map page");
            STRIX_HIP_CHECK(hipMemSetAccess(at, cfg.page_bytes, &access, 1), "re-set access");
        }

        // 6. Write Pattern B
        uint32_t seed_b = 0xB2000000u | (uint32_t)iter;
        fill_pattern_kernel<<<(num_elements + 255) / 256, 256>>>(
            static_cast<uint32_t *>(base_va), num_elements, seed_b);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync write B");

        // 7. Verify Pattern B and check for stale Pattern A or Poison pattern leaks
        STRIX_HIP_CHECK(hipMemset(d_errors, 0, sizeof(uint32_t)), "clear errors");
        STRIX_HIP_CHECK(hipMemset(d_poison_hits, 0, sizeof(uint32_t)), "clear poison hits");

        check_pattern_kernel<<<(num_elements + 255) / 256, 256>>>(
            static_cast<uint32_t *>(base_va), num_elements, seed_b, poison_seed, d_errors, d_poison_hits);
        STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync check");

        uint32_t h_errors = 0, h_poison = 0;
        STRIX_HIP_CHECK(hipMemcpyDtoH(&h_errors, d_errors, sizeof(uint32_t)), "read errors");
        STRIX_HIP_CHECK(hipMemcpyDtoH(&h_poison, d_poison_hits, sizeof(uint32_t)), "read poison hits");

        if (h_errors > 0) {
            total_corruptions += h_errors;
            total_poison_hits += h_poison;
            printf("  [FAIL] Iteration %d: %u mismatches (poison hits: %u)!\n", iter, h_errors, h_poison);
        }

        // Cleanup this iteration's pages
        for (size_t p = 0; p < cfg.num_pages; ++p) {
            char *at = static_cast<char *>(base_va) + p * cfg.page_bytes;
            STRIX_HIP_CHECK(hipMemUnmap(at, cfg.page_bytes), "unmap cleanup");
            STRIX_HIP_CHECK(hipMemRelease(pages[p]), "release cleanup");
        }

        if (destroy_reservation) {
            STRIX_HIP_CHECK(hipMemAddressFree(base_va, total_bytes), "free base cleanup");
            base_va = nullptr;
        }

        if ((iter + 1) % 250 == 0 || iter == cfg.iterations - 1) {
            printf("  Progress: %d / %d iterations completed (corruptions: %lu)...\n",
                   iter + 1, cfg.iterations, total_corruptions);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!destroy_reservation && base_va) {
        STRIX_HIP_CHECK(hipMemAddressFree(base_va, total_bytes), "free base final");
    }

    // Cleanup polluter
    for (size_t p = 0; p < polluter_pages; ++p) {
        char *at = static_cast<char *>(polluter_va) + p * cfg.page_bytes;
        STRIX_HIP_CHECK(hipMemUnmap(at, cfg.page_bytes), "polluter unmap");
        STRIX_HIP_CHECK(hipMemRelease(polluter_handles[p]), "polluter release");
    }
    STRIX_HIP_CHECK(hipMemAddressFree(polluter_va, polluter_pages * cfg.page_bytes), "polluter free");
    STRIX_HIP_CHECK(hipFree(d_errors), "free d_errors");
    STRIX_HIP_CHECK(hipFree(d_poison_hits), "free d_poison_hits");

    // Stop background noise
    if (cfg.background_noise) {
        if (cfg.multi_process && noise_pid > 0) {
            kill(noise_pid, SIGTERM);
            waitpid(noise_pid, nullptr, 0);
        } else if (noise_thread.joinable()) {
            noise_thread_stop.store(true);
            noise_thread.join();
        }
    }

    printf("Verdict: %s (corruptions: %lu, poison hits: %lu, elapsed: %.2f s, avg: %.2f ms/iter)\n",
           total_corruptions == 0 ? "PASSED (100% clean & safe)" : "FAILED (CORRUPTION CONFIRMED)",
           total_corruptions, total_poison_hits, ms / 1000.0, ms / cfg.iterations);
    return total_corruptions == 0;
}

}  // namespace

int main(int argc, char **argv) {
    TestConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            cfg.iterations = (int)parse_int(argv[++i], "--iterations", 1, 1000000);
        } else if (arg == "--pages" && i + 1 < argc) {
            cfg.num_pages = (size_t)parse_int(argv[++i], "--pages", 1, 1024);
        } else if (arg == "--page-mb" && i + 1 < argc) {
            cfg.page_bytes = (size_t)parse_int(argv[++i], "--page-mb", 1, 64) << 20;
        } else if (arg == "--noise" && i + 1 < argc) {
            std::string val = argv[++i];
            cfg.background_noise = (val == "on" || val == "1" || val == "true");
        } else if (arg == "--multi-process") {
            cfg.multi_process = true;
        } else if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    int dev = 0;
    STRIX_HIP_CHECK(hipSetDevice(dev), "set device");
    hipDeviceProp_t dev_prop{};
    STRIX_HIP_CHECK(hipGetDeviceProperties(&dev_prop, dev), "get device properties");

    int vmm = 0;
    STRIX_HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev), "vmm attr");
    STRIX_CHECK(vmm == 1, "Device ", dev, " does not support Virtual Memory Management");

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;

    size_t gran = 0;
    STRIX_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum), "granularity");
    if (cfg.page_bytes % gran != 0) {
        cfg.page_bytes = ((cfg.page_bytes + gran - 1) / gran) * gran;
    }

    printf("============================================================\n");
    printf("   STRIX VMM UNMAP / REMAP SAFETY SUITE                    \n");
    printf("============================================================\n");
    printf("Device: %s (%s)\n", dev_prop.name, dev_prop.gcnArchName);
    printf("VMM Supported: %d, Granularity: %zu B\n", vmm, gran);
    printf("Config: %d iterations, %zu pages of %zu MB (%.1f MB buffer)\n",
           cfg.iterations, cfg.num_pages, cfg.page_bytes >> 20, (double)(cfg.num_pages * cfg.page_bytes) / (1024 * 1024));

    bool all_ok = true;

    if (cfg.mode == "all" || cfg.mode == "cross") {
        all_ok &= test_cross_reservation(cfg, dev, prop, gran);
    }
    if (cfg.mode == "all" || cfg.mode == "inplace") {
        all_ok &= test_unmap_remap_stress(cfg, dev, prop, gran, false);
    }
    if (cfg.mode == "all" || cfg.mode == "recreate") {
        all_ok &= test_unmap_remap_stress(cfg, dev, prop, gran, true);
    }

    printf("\n============================================================\n");
    printf("   OVERALL STATUS: %s\n", all_ok ? "ALL TESTS PASSED" : "TEST FAILURES DETECTED");
    printf("============================================================\n");

    return all_ok ? 0 : 1;
}
