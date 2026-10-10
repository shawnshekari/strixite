#pragma once

// Precondition / error checks. Every function
// validates its parameters with these; only hot loops are exempt. A failure
// throws strix::Error whose message carries file:line, the function, the
// failed condition, and a caller-supplied description of expected vs found:
//
//   STRIX_CHECK(n % 4 == 0, "row length ", n, " must be a multiple of 4 for tensor '", name, "'");
//   -> "rmsnorm (kernels/rmsnorm.hip:42): check `n % 4 == 0` failed: row length 6 must be ..."
//
// Header-only and dependency-free so host code, tools and tests can all use it.

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A GPU error after which the device context is gone (an illegal address, a kernel fault): every later call fails
// too, so the request fails and the process exits for its supervisor to restart (common/hip_check.hpp classifies).
class GpuFatalError : public Error {
public:
    using Error::Error;
};

template <typename... Args>
std::string cat(Args &&...args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return os.str();
}

// "[a,b,c]" for shapes/index lists in messages.
template <typename T>
std::string list_str(const std::vector<T> &v) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << ']';
    return os.str();
}

[[noreturn]] inline void check_failed(const char *file, int line, const char *func, const char *cond,
                                      const std::string &msg) {
    throw Error(cat(func, " (", file, ":", line, "): check `", cond, "` failed: ", msg));
}

[[noreturn]] inline void fail_at(const char *file, int line, const char *func, const std::string &msg) {
    throw Error(cat(func, " (", file, ":", line, "): ", msg));
}

}  // namespace strix

#define STRIX_CHECK(cond, ...)                                                                   \
    do {                                                                                         \
        if (!(cond)) ::strix::check_failed(__FILE__, __LINE__, __func__, #cond,                  \
                                           ::strix::cat(__VA_ARGS__));                           \
    } while (0)

// Unconditional failure: "func (file:line): message".
#define STRIX_FAIL(...) ::strix::fail_at(__FILE__, __LINE__, __func__, ::strix::cat(__VA_ARGS__))
