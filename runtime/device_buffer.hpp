#pragma once

// Owning, move-only device allocation with checked upload/download. Every
// buffer carries a name so allocation and copy errors say which buffer failed.

#include "common/hip_check.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strix {

// Live GPU-visible allocations (DeviceBuffer and PinnedHostBuffer; on gfx1151 both come out of GTT, i.e. system
// RAM), summed per name category: the name up to its first digit or quote ("K cache 12" -> "K cache"), so a
// startup line can say where the memory went. Allocations outside these two classes are not counted.
class GpuMemoryTally {
public:
    static void add(const std::string &name, int64_t bytes) {
        std::lock_guard<std::mutex> lock(mu());  // (called from destructors too: no throwing here)
        map()[category(name)] += bytes;
    }
    // (category, bytes) with bytes > 0, largest first.
    static std::vector<std::pair<std::string, int64_t>> snapshot() {
        std::vector<std::pair<std::string, int64_t>> out;
        {
            std::lock_guard<std::mutex> lock(mu());
            for (const auto &kv : map()) {
                STRIX_CHECK(kv.second >= 0, "GpuMemoryTally: category '", kv.first, "' holds ", kv.second,
                            " bytes, expected >= 0 (a free counted twice?)");
                if (kv.second > 0) out.emplace_back(kv.first, kv.second);
            }
        }
        std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
        return out;
    }
    static std::string category(const std::string &name) {
        size_t end = 0;
        while (end < name.size() && !std::isdigit((unsigned char)name[end]) && name[end] != '\'') ++end;
        while (end > 0 && (name[end - 1] == ' ' || name[end - 1] == '.' || name[end - 1] == ',')) --end;
        return end == 0 ? std::string("(unnamed)") : name.substr(0, end);
    }

private:
    // Never destroyed: a static DeviceBuffer elsewhere may be freed after a function-local static would be (test_mtp
    // aborted at exit that way).
    static std::mutex &mu() {
        static std::mutex *m = new std::mutex;
        return *m;
    }
    static std::map<std::string, int64_t> &map() {
        static auto *m = new std::map<std::string, int64_t>;
        return *m;
    }
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(size_t n, std::string name) : n_(n), name_(std::move(name)) {
        STRIX_CHECK(n > 0, "device buffer '", name_, "' requested with 0 elements");
        STRIX_CHECK(n <= SIZE_MAX / sizeof(T), "device buffer '", name_, "': ", n, " elements overflow size_t");
        STRIX_HIP_CHECK(hipMalloc(&ptr_, n * sizeof(T)), "allocating device buffer '", name_, "', ", n, " x ",
                        sizeof(T), " B = ", n * sizeof(T), " bytes");
        GpuMemoryTally::add(name_, (int64_t)(n * sizeof(T)));
    }
    static DeviceBuffer from_host(const std::vector<T> &host, std::string name) {
        DeviceBuffer b(host.size(), std::move(name));
        b.upload(host);
        return b;
    }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_), name_(std::move(o.name_)) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            n_ = o.n_;
            name_ = std::move(o.name_);
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const {
        STRIX_CHECK(ptr_ != nullptr, "device buffer '", name_, "' is empty (moved-from or default-constructed)");
        return ptr_;
    }
    size_t size() const { return n_; }
    const std::string &name() const { return name_; }

    void upload(const std::vector<T> &host) {
        STRIX_CHECK(host.size() == n_, "upload to '", name_, "': host has ", host.size(), " elements, buffer has ", n_);
        STRIX_HIP_CHECK(hipMemcpy(get(), host.data(), n_ * sizeof(T), hipMemcpyHostToDevice), "upload to '", name_,
                        "', ", n_ * sizeof(T), " bytes");
    }
    std::vector<T> to_host() const {
        std::vector<T> host(n_);
        STRIX_HIP_CHECK(hipMemcpy(host.data(), get(), n_ * sizeof(T), hipMemcpyDeviceToHost), "download of '", name_,
                        "', ", n_ * sizeof(T), " bytes");
        return host;
    }

private:
    void release() {
        if (!ptr_) return;
        (void)hipFree(ptr_);
        GpuMemoryTally::add(name_, -(int64_t)(n_ * sizeof(T)));
        ptr_ = nullptr;
    }
    T *ptr_ = nullptr;
    size_t n_ = 0;
    std::string name_;
};

// Owning, move-only pinned (page-locked) host allocation, for async host -> device copies. gfx1151 hazard:
// never write it again for the next transfer until the previous copy from it has completed (an event or
// stream sync) - ungated reuse corrupts data.
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    PinnedHostBuffer(size_t bytes, std::string name) : bytes_(bytes), name_(std::move(name)) {
        STRIX_CHECK(bytes > 0, "pinned host buffer '", name_, "' requested with 0 bytes");
        STRIX_HIP_CHECK(hipHostMalloc(&ptr_, bytes, hipHostMallocDefault), "allocating pinned host buffer '", name_,
                        "', ", bytes, " bytes");
        GpuMemoryTally::add(name_ + " (pinned host)", (int64_t)bytes);
    }
    ~PinnedHostBuffer() { release(); }
    PinnedHostBuffer(PinnedHostBuffer &&o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_), name_(std::move(o.name_)) {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    PinnedHostBuffer &operator=(PinnedHostBuffer &&o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_, bytes_ = o.bytes_, name_ = std::move(o.name_);
            o.ptr_ = nullptr, o.bytes_ = 0;
        }
        return *this;
    }
    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;
    void *get() const { return ptr_; }
    size_t size() const { return bytes_; }  // bytes
    const std::string &name() const { return name_; }

private:
    void release() {
        if (!ptr_) return;
        (void)hipHostFree(ptr_);
        GpuMemoryTally::add(name_ + " (pinned host)", -(int64_t)bytes_);
        ptr_ = nullptr;
    }
    void *ptr_ = nullptr;
    size_t bytes_ = 0;
    std::string name_;
};

}  // namespace strix
