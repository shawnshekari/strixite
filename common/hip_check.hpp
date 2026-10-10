#pragma once

// HIP return-code check in the common/check.hpp format:
//   STRIX_HIP_CHECK(hipMalloc(&p, bytes), "device buffer for '", name, "', ", bytes, " bytes");
//   -> "... check `hipMalloc(&p, bytes)` failed: hipErrorOutOfMemory (out of memory): device buffer ..."
// A sticky error (hip_error_is_sticky) throws GpuFatalError instead of Error.
// Separate from check.hpp so host-only code doesn't need the HIP headers.

#include "common/check.hpp"

#include "common/hip_runtime.hpp"

namespace strix {

// The HIP errors that leave the device context unusable (sticky: every later call returns an error, as after an
// illegal address or a memory aperture violation - strixite issue #6): a kernel fault, an assert, a launch that
// timed out, or a context already destroyed. Everything else (out of memory, a bad argument) leaves the GPU usable.
inline bool hip_error_is_sticky(hipError_t e) {
    switch (e) {
    case hipErrorIllegalAddress:
    case hipErrorLaunchFailure:
    case hipErrorLaunchTimeOut:
    case hipErrorAssert:
    case hipErrorContextIsDestroyed:
        return true;
    default:
        return false;
    }
}

[[noreturn]] inline void hip_check_failed(const char *file, int line, const char *func, const char *cond, hipError_t e,
                                          const std::string &msg) {
    const std::string what = cat(func, " (", file, ":", line, "): check `", cond, "` failed: ", hipGetErrorName(e), " (",
                                 hipGetErrorString(e), "): ", msg);
    if (hip_error_is_sticky(e)) throw GpuFatalError(what + " - the GPU context is lost; only a restart recovers");
    throw Error(what);
}

}  // namespace strix

#define STRIX_HIP_CHECK(expr, ...)                                                               \
    do {                                                                                         \
        hipError_t strix_hip_err_ = (expr);                                                      \
        if (strix_hip_err_ != hipSuccess)                                                        \
            ::strix::hip_check_failed(__FILE__, __LINE__, __func__, #expr, strix_hip_err_,       \
                                      ::strix::cat(__VA_ARGS__));                                \
    } while (0)

// After a kernel launch: catches bad launch configs immediately rather than at the next sync.
#define STRIX_HIP_CHECK_LAUNCH(...) STRIX_HIP_CHECK(hipGetLastError(), "kernel launch: ", __VA_ARGS__)
