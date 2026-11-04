#pragma once

// A device array that takes memory as it is used: the address range for the
// full capacity is reserved once (hipMemAddressReserve), and physical pages (hipMemCreate) are mapped into it where
// it is used - kernels see one array at a fixed pointer with its layout unchanged, as with DeviceBuffer. An array of
// several capacity-sized regions (the 8-bit K/V: codes and params of K and V) maps each region's used prefix: ranges
// of bytes, any page they touch. On Strix Halo the GPU's pages come from system RAM, so a slot's K/V costs the
// deepest point its conversations have reached, not its full capacity.
//
// A mapped page stays mapped until the buffer is destroyed - there is no runtime release. On gfx1151, after a page
// was unmapped and released and a new one mapped at the same address, kernels sometimes read memory other than the
// new page there (another allocation's, once the freed memory was reused), with nothing failing: the same forward
// gave different logits, only while other processes used the GPU.
//
// Pages are only ever mapped back into the reservation they were carved from: on gfx1151 a page mapped into a
// second reservation reads the wrong memory.
// That is what VmmReservation (below) is for: arrays that must hand pages to one another (the slots' K/V, the
// parked-page region) share one reservation instead of each reserving its own.
//
// Not thread-safe (as DeviceBuffer). Growing while a kernel reads the range is safe: ensure() maps pages that weren't
// mapped, so kernels reading mapped bytes are unaffected.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "runtime/device_buffer.hpp"

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strix {

// One HIP virtual address reservation (hipMemAddressReserve) shared by several paged arrays: they are carved out of
// it at page-aligned offsets and their physical pages can move between them without a copy - on gfx1151 a page
// mapped into a second reservation reads the wrong memory, so everything that must hand pages to something else
// (all serving slots' K/V, the parked-page region) lives in one of these. The reservation is
// address space only: it costs no RAM; the pages its buffers map do, and the pages outlive their mapping
// (hipMemRelease is the free, and nothing does that at runtime - see the hazard note above).
//
// Not thread-safe. Outlives the buffers carved from it (their pages unmap before the range is freed).
class VmmReservation {
public:
    VmmReservation() = default;
    // bytes: the total range (slot regions + parked region); page_bytes: the carving / mapping unit shared by every
    // buffer of the reservation (the caller's layout rounds each carve up to it). Checked like PagedDeviceBuffer's.
    VmmReservation(size_t bytes, size_t page_bytes, std::string name) : name_(std::move(name)) {
        STRIX_CHECK(bytes > 0, "reservation '", name_, "': size of 0 bytes");
        STRIX_CHECK(page_bytes > 0, "reservation '", name_, "': page of 0 bytes");
        STRIX_HIP_CHECK(hipGetDevice(&dev_), "reservation '", name_, "': current device");
        int vmm = 0;
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev_),
                        "reservation '", name_, "': VMM support attribute");
        STRIX_CHECK(vmm == 1, "reservation '", name_, "': device ", dev_, " reports no virtual memory management ",
                    "(attribute ", vmm, ")");
        const hipMemAllocationProp prop = page_prop();
        size_t gran = 0;
        STRIX_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum),
                        "reservation '", name_, "': granularity");
        STRIX_CHECK(gran > 0 && page_bytes % gran == 0, "reservation '", name_, "': page of ", page_bytes,
                    " B is not a multiple of the device's VMM granularity ", gran, " B");
        STRIX_HIP_CHECK(hipMemAddressReserve(&base_, bytes, gran, nullptr, 0), "reservation '", name_,
                        "': reserving ", bytes, " B of address space");
        size_ = bytes;
        page_ = page_bytes;
    }
    ~VmmReservation() {
        if (base_ != nullptr) (void)hipMemAddressFree(base_, size_);
    }
    VmmReservation(VmmReservation &&o) noexcept { swap(o); }
    // Not noexcept: assigning onto a live range is an error, and it must say so (the destructor would leak the old
    // range's address space, which the no-throw move-assign path could not report).
    VmmReservation &operator=(VmmReservation &&o) {
        if (this != &o) {
            STRIX_CHECK(base_ == nullptr, "VmmReservation: move-assign onto a live reservation (", name_, ", ",
                        size_, " B) - release it first");
            swap(o);
        }
        return *this;
    }
    VmmReservation(const VmmReservation &) = delete;
    VmmReservation &operator=(const VmmReservation &) = delete;

    void *base() const { return base_; }
    size_t size() const { return size_; }
    size_t page_bytes() const { return page_; }
    const std::string &name() const { return name_; }
    int device() const { return dev_; }
    hipMemAllocationProp page_prop() const {
        hipMemAllocationProp p = {};
        p.type = hipMemAllocationTypePinned;
        p.location.type = hipMemLocationTypeDevice;
        p.location.id = dev_;
        return p;
    }

private:
    void swap(VmmReservation &o) noexcept {
        std::swap(base_, o.base_);
        std::swap(size_, o.size_);
        std::swap(page_, o.page_);
        std::swap(dev_, o.dev_);
        std::swap(name_, o.name_);
    }
    void *base_ = nullptr;
    size_t size_ = 0, page_ = 0;
    int dev_ = 0;
    std::string name_;
};

// One physical page kept across a hand-off: its index within the paged array, and its HIP handle. The units
// of parking: park takes them out of a slot's generation, the
// parked side maps them into the parked region (with host access - the disk writer reads the parked pages over
// the host bus, probe check 2 measured CPU-read-needs-host), resume maps them back into a slot's next
// generation, eviction releases them (RAM back, the parked generation's VA spent).
struct PageHandle {
    size_t page = 0;
    hipMemGenericAllocationHandle_t handle = nullptr;
};

// ParkedKv (the record of what one park kept) is in runtime/qwen4exp.hpp: it is the runtime's business, and
// the host-only view of a parked conversation is runtime/paged_state.hpp's PagedState.

// Maps kept handles into a range of one reservation, base + page * page_bytes for each, with device read-write
// access - and host read-write too when host_access. All-or-nothing: the first map or set-access that fails
// unmaps the ones already made (the handles stay kept - the caller owns them either way) and throws. The parked
// side of parking maps whole arrays' handle lists with it; PagedDeviceBuffer::adopt_pages wraps it for a slot.
inline void map_pages(void *base, size_t page_bytes, const std::vector<PageHandle> &pages, bool host_access,
                      const std::string &what) {
    STRIX_CHECK(base != nullptr, "'", what, "': map_pages onto a null base");
    STRIX_CHECK(page_bytes > 0, "'", what, "': map_pages with a page of 0 bytes");
    hipMemAccessDesc acc[2] = {};
    acc[0].location.type = hipMemLocationTypeDevice;
    acc[0].flags = hipMemAccessFlagsProtReadWrite;
    acc[1].location.type = hipMemLocationTypeHost;
    acc[1].flags = hipMemAccessFlagsProtReadWrite;
    const size_t n_desc = host_access ? 2 : 1;
    for (size_t k = 0; k < pages.size(); ++k) {
        STRIX_CHECK(pages[k].handle != nullptr, "'", what, "': null handle for page ", pages[k].page);
        char *at = static_cast<char *>(base) + pages[k].page * page_bytes;
        const hipError_t m = hipMemMap(at, page_bytes, 0, pages[k].handle, 0);
        if (m != hipSuccess) {
            (void)hipGetLastError();  // the caller asked whether the handle is alive; a dead one is an answer,
                                      // not a crash - clear the flag before the unwind and the throw
            for (size_t q = 0; q < k; ++q)  // unwind: the caller keeps the handles, only the mappings go
                (void)hipMemUnmap(static_cast<char *>(base) + pages[q].page * page_bytes, page_bytes);
            STRIX_HIP_CHECK(m, "'", what, "': mapping kept page ", pages[k].page, " at ", (const void *)at);
        }
        const hipError_t a = hipMemSetAccess(at, page_bytes, acc, (unsigned)n_desc);
        if (a != hipSuccess) {
            (void)hipGetLastError();  // as above
            (void)hipMemUnmap(at, page_bytes);
            for (size_t q = 0; q < k; ++q)
                (void)hipMemUnmap(static_cast<char *>(base) + pages[q].page * page_bytes, page_bytes);
            STRIX_HIP_CHECK(a, "'", what, "': ", host_access ? "host + device" : "device", " access for kept page ",
                            pages[k].page, " at ", (const void *)at);
        }
    }
}

// Tears down a map_pages mapping at base: unmap every page, and release the handles when release (eviction -
// RAM back; the addresses stay spent like every other generation's). The handles otherwise stay kept (resume:
// they are mapped into a slot right after). Every page gets its attempt; if any step failed, throws AFTER
// (half a parked region torn down is still a torn-down record; bytes_done reports the bytes whose RAM came
// back - counted out of `pages`, the caller's tally term for them).
inline void unmap_pages(void *base, size_t page_bytes, const std::vector<PageHandle> &pages, bool release,
                         const std::string &what, int64_t *bytes_done = nullptr) {
    STRIX_CHECK(base != nullptr, "'", what, "': unmap_pages from a null base");
    STRIX_CHECK(page_bytes > 0, "'", what, "': unmap_pages with a page of 0 bytes");
    int64_t done = 0;
    const char *first_err = nullptr;
    size_t first_page = 0;
    for (const PageHandle &ph : pages) {
        STRIX_CHECK(ph.handle != nullptr || !release, "'", what, "': null handle for page ", ph.page);
        char *at = static_cast<char *>(base) + ph.page * page_bytes;
        if (hipMemUnmap(at, page_bytes) != hipSuccess) {
            (void)hipGetLastError();  // a caller probing a torn-down range wants the named error, not a flag
            if (first_err == nullptr) first_err = "hipMemUnmap", first_page = ph.page;
            continue;
        }
        if (!release) {
            done += (int64_t)page_bytes;
            continue;
        }
        if (hipMemRelease(ph.handle) == hipSuccess)
            done += (int64_t)page_bytes;
        else if (first_err == nullptr) {
            (void)hipGetLastError();
            first_err = "hipMemRelease", first_page = ph.page;
        }
    }
    if (bytes_done != nullptr) *bytes_done = done;
    STRIX_CHECK(first_err == nullptr, "'", what, "': ", first_err, " failed for page ", first_page, " (",
                done / (int64_t)page_bytes, " of ", pages.size(), " pages completed - their RAM is back, the rest "
                "stays held; the generation's addresses are spent either way)");
}

class PagedDeviceBuffer {
public:
    PagedDeviceBuffer() = default;
    // capacity_bytes: the most this array will ever hold (rounded up to whole pages); page_bytes: the mapping unit, a
    // multiple of the device's minimum VMM granularity. No page is mapped yet: get() is valid, size() is 0.
    PagedDeviceBuffer(size_t capacity_bytes, size_t page_bytes, std::string name) : name_(std::move(name)) {
        STRIX_CHECK(capacity_bytes > 0, "paged buffer '", name_, "': capacity of 0 bytes");
        STRIX_CHECK(page_bytes > 0, "paged buffer '", name_, "': page of 0 bytes");
        STRIX_HIP_CHECK(hipGetDevice(&dev_), "paged buffer '", name_, "': current device");
        int vmm = 0;
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev_),
                        "paged buffer '", name_, "': VMM support attribute");
        STRIX_CHECK(vmm == 1, "paged buffer '", name_, "': device ", dev_, " reports no virtual memory management ",
                    "(attribute ", vmm, ")");
        const hipMemAllocationProp prop = page_prop();
        size_t gran = 0;
        STRIX_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum),
                        "paged buffer '", name_, "': granularity");
        STRIX_CHECK(gran > 0 && page_bytes % gran == 0, "paged buffer '", name_, "': page of ", page_bytes,
                    " B is not a multiple of the device's VMM granularity ", gran, " B");
        STRIX_CHECK(capacity_bytes <= SIZE_MAX - page_bytes, "paged buffer '", name_, "': capacity ", capacity_bytes,
                    " B overflows when rounded to ", page_bytes, " B pages");
        page_ = page_bytes;
        reserved_ = (capacity_bytes + page_bytes - 1) / page_bytes * page_bytes;
        STRIX_HIP_CHECK(hipMemAddressReserve(&base_, reserved_, gran, nullptr, 0), "paged buffer '", name_,
                        "': reserving ", reserved_, " B of address space");
        pages_.assign(reserved_ / page_, nullptr);
    }
    // The same array carved out of a shared reservation (VmmReservation): its pages live at
    // reservation base + offset, which the caller took from the layout's carve (page-aligned, the array's pages never
    // overlap another buffer's). get() is base() + offset; the reservation must outlive the buffer.
    PagedDeviceBuffer(VmmReservation &res, size_t offset, size_t capacity_bytes, std::string name)
        : name_(std::move(name)), shared_(true) {
        STRIX_CHECK(capacity_bytes > 0, "paged buffer '", name_, "': capacity of 0 bytes");
        STRIX_CHECK(offset % res.page_bytes() == 0, "paged buffer '", name_, "': carve offset ", offset,
                    " is not a multiple of the reservation's page ", res.page_bytes(), " B");
        const size_t page_bytes = res.page_bytes();
        STRIX_CHECK(capacity_bytes <= SIZE_MAX - page_bytes, "paged buffer '", name_, "': capacity ", capacity_bytes,
                    " B overflows when rounded to ", page_bytes, " B pages");
        page_ = page_bytes;
        reserved_ = (capacity_bytes + page_bytes - 1) / page_bytes * page_bytes;
        STRIX_CHECK(offset <= res.size() && reserved_ <= res.size() - offset, "paged buffer '", name_,
                    "': carve of ", reserved_, " B at offset ", offset, " is past the reservation '", res.name(),
                    "' of ", res.size(), " B");
        dev_ = res.device();
        base_ = static_cast<char *>(res.base()) + offset;
        offset_ = offset;
        res_bytes_ = res.size();
        pages_.assign(reserved_ / page_, nullptr);
    }

    // The ONE-USE-ADDRESS generations: enable_generations carves the buffer's range into
    // `cap` generations of `stride` bytes within the shared reservation - the caller laid them out, generation g's
    // array starts at the carve offset + g * stride; generation() starts at 0. While a generation is current,
    // pages_ / ensure() / get() address it. release_and_next_generation() unmaps + releases the current
    // generation's pages and advances the cursor, so the next conversation maps NEW handles at addresses this
    // process has never mapped: on gfx1151, unmapping + releasing a page and mapping a new one at the same
    // address while the process lives reads wrong memory silently - so here, no address is ever used twice.
    // Running out of generations fails cleanly; the kv-generations budget sizes the cap and the reservation's
    // virtual size (address space only - no RAM until pages map).
    void enable_generations(size_t stride_bytes, uint32_t cap) {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': enable_generations on an empty (moved-from or "
                    "default) buffer");
        STRIX_CHECK(shared_, "paged buffer '", name_, "': enable_generations on a non-shared buffer");
        STRIX_CHECK(gen_cap_ == 1, "paged buffer '", name_, "': generations already enabled (cap ", gen_cap_, ")");
        STRIX_CHECK(cap >= 2, "paged buffer '", name_, "': enable_generations cap ", cap, ", expected >= 2");
        STRIX_CHECK(mapped_ == 0, "paged buffer '", name_, "': enable_generations after ", mapped_,
                    " pages were mapped - the generations must be set before the array maps anything");
        STRIX_CHECK(stride_bytes >= reserved_ && stride_bytes % page_ == 0, "paged buffer '", name_,
                    "': generation stride ", stride_bytes, " B must be page-aligned and >= the array's ", reserved_,
                    " B");
        const size_t room = offset_ <= res_bytes_ ? res_bytes_ - offset_ : 0;
        STRIX_CHECK(room / stride_bytes >= cap, "paged buffer '", name_, "': ", cap, " generations of ", stride_bytes,
                    " B do not fit the reservation's ", room, " B past carve offset ", offset_);
        gen_stride_ = stride_bytes;
        gen_cap_ = cap;
    }
    bool generations_enabled() const { return gen_cap_ > 1; }
    uint32_t generation() const { return gen_; }
    // True while nothing was ever mapped (or attempted) in the current generation: its addresses are still
    // unused, so a reset has nothing to release and a resume may map kept pages straight into it - neither
    // needs to spend the next generation. A park or release advances into a fresh one; any map makes it used.
    bool generation_fresh() const { return !gen_used_; }
    uint32_t generation_capacity() const { return gen_cap_; }
    // Unmap + release every mapped page of the current generation (the conversation's K/V is done - the only
    // runtime release a generation-enabled buffer performs), then advance to the next, never-before-mapped
    // generation. Checked + throwing (unlike the destructor's release path): a failed step must stop the request,
    // not half-change the cursor.
    void release_and_next_generation() {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': release_and_next_generation on an empty buffer");
        STRIX_CHECK(gen_cap_ > 1, "paged buffer '", name_, "': generations not enabled");
        // Refuse BEFORE touching anything: past the last generation the conversation is over, but half-advancing
        // (pages gone, cursor kept) would leave the slot reading a released generation - the failure is clean.
        STRIX_CHECK(gen_ + 1 < gen_cap_, "paged buffer '", name_, "': K/V generations exhausted - generation ",
                    gen_ + 1, " of ", gen_cap_, ", the reservation holds ", gen_cap_, " x ", gen_stride_,
                    " B at offset ", offset_, " (one-use addresses: a released generation's addresses are never "
                    "re-mapped while the process lives; this is the kv-generations exhaustion refusal, and only a "
                    "restart reclaims)");
        char *at = static_cast<char *>(base_) + (size_t)gen_ * gen_stride_;
        for (size_t p = 0; p < pages_.size(); ++p) {
            if (pages_[p] == nullptr) continue;
            STRIX_HIP_CHECK(hipMemUnmap(at + p * page_, page_), "paged buffer '", name_, "': unmap generation ",
                             gen_, " page ", p, " (", mapped_, " of ", pages_.size(), " mapped)");
            STRIX_HIP_CHECK(hipMemRelease(pages_[p]), "paged buffer '", name_, "': release generation ", gen_,
                             " page ", p);
            pages_[p] = nullptr;
            GpuMemoryTally::add(name_, -(int64_t)page_);
            --mapped_;
        }
        STRIX_CHECK(mapped_ == 0, "paged buffer '", name_, "': ", mapped_, " pages counted after unmapping all");
        ++gen_;
        gen_used_ = false;
    }

    // The park/resume primitives: release_and_next_generation's handle-KEEPING siblings. park unmaps the current generation's pages and RETURNS
    // their handles instead of releasing them, then advances the cursor: the conversation's pages go on living
    // in the caller's hands (mapped into the parked region by map_pages), and this slot never maps at these
    // addresses again. adopt is the inverse: it maps kept handles into the current generation, which must be
    // one nothing was ever mapped at (the resume maps the parked pages into the slot's next generation). A
    // resume that fails part-way hands pages back with discard_current_pages - unmapping without touching the
    // handles, which the record (and the parked-side mapping) still names.
    std::vector<PageHandle> park_and_next_generation() {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': park_and_next_generation on an empty (moved-from or default) buffer");
        STRIX_CHECK(gen_cap_ > 1, "paged buffer '", name_, "': generations not enabled");
        // Refuse BEFORE touching anything, as release does: the park wants the slot's next generation too.
        STRIX_CHECK(gen_ + 1 < gen_cap_, "paged buffer '", name_, "': parking K/V generations exhausted - the park "
                    "wants generation ", gen_ + 1, " of ", gen_cap_, " (one-use addresses: the parked pages leave "
                    "this slot's generation, and the next conversation's pages need an address never used while "
                    "the process lives; this is the kv-generations exhaustion refusal, and only a restart "
                    "reclaims)");
        char *at = static_cast<char *>(base_) + (size_t)gen_ * gen_stride_;
        std::vector<PageHandle> out;
        out.reserve(mapped_);
        for (size_t p = 0; p < pages_.size(); ++p) {
            if (pages_[p] == nullptr) continue;
            const hipError_t u = hipMemUnmap(at + p * page_, page_);
            if (u != hipSuccess) {
                for (const PageHandle &ph : out)  // the caller never received these - nothing may leak
                    (void)hipMemRelease(ph.handle);
                STRIX_HIP_CHECK(u, "paged buffer '", name_, "': parking unmap generation ", gen_, " page ", p,
                                " (", mapped_, " of ", pages_.size(), " mapped)");
            }
            out.push_back({p, pages_[p]});
            pages_[p] = nullptr;
            GpuMemoryTally::add(name_, -(int64_t)page_);  // the pages are no longer this buffer's; the parker
            --mapped_;                                    // charges them to its parked-entry category
        }
        STRIX_CHECK(mapped_ == 0, "paged buffer '", name_, "': ", mapped_, " pages counted after parking all");
        ++gen_;
        gen_used_ = false;
        return out;
    }
    // Maps kept handles into the CURRENT generation - which must have nothing mapped yet (the slot advanced
    // here for exactly this). Takes ownership on success (the tally follows, like map_page); on a failed map
    // nothing is taken: the generation stays empty and the caller still owns every handle.
    void adopt_pages(const std::vector<PageHandle> &pages) {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': adopt_pages on an empty (moved-from or default) buffer");
        STRIX_CHECK(gen_cap_ > 1, "paged buffer '", name_, "': generations not enabled");
        STRIX_CHECK(mapped_ == 0, "paged buffer '", name_, "': adopt_pages into generation ", gen_, ", which has ",
                    mapped_, " pages mapped - a resume maps into the slot's NEXT generation (advance first)");
        std::vector<char> seen(pages_.size(), 0);
        for (const PageHandle &ph : pages) {
            STRIX_CHECK(ph.page < pages_.size(), "paged buffer '", name_, "': adopted page ", ph.page, " is past the "
                        "array's ", pages_.size(), " pages");
            STRIX_CHECK(ph.handle != nullptr, "paged buffer '", name_, "': null handle for page ", ph.page);
            STRIX_CHECK(!seen[ph.page], "paged buffer '", name_, "': two handles for page ", ph.page);
            seen[ph.page] = 1;
        }
        gen_used_ = true;  // as in map_page: an attempt counts, a failed map_pages unwinds but used the addresses
        map_pages(get(), page_, pages, /*host_access=*/false, name_);  // device access, as a normal page; throws
                                                        // with nothing taken on failure (map_pages unwinds)
        for (const PageHandle &ph : pages) {
            pages_[ph.page] = ph.handle;
            ++mapped_;
            GpuMemoryTally::add(name_, (int64_t)page_);
        }
    }
    // Unmaps the current generation's pages, keeping every handle (the caller's record names them, and the
    // parked-side mapping still has them too - the unwind of a failed resume). Checked + throwing like the
    // other cursor steps: a failed unmap here leaves a double mapping, which is the caller's problem to fail clean.
    void discard_current_pages() {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': discard_current_pages on an empty (moved-from or default) buffer");
        char *at = static_cast<char *>(base_) + (size_t)gen_ * gen_stride_;
        int64_t bytes_done = 0;
        std::vector<PageHandle> mapped;
        for (size_t p = 0; p < pages_.size(); ++p)
            if (pages_[p] != nullptr) mapped.push_back({p, pages_[p]});
        unmap_pages(at, page_, mapped, /*release=*/false, name_, &bytes_done);
        for (PageHandle &ph : mapped) {
            pages_[ph.page] = nullptr;
            --mapped_;
        }
        GpuMemoryTally::add(name_, -bytes_done);
        STRIX_CHECK(mapped_ == 0, "paged buffer '", name_, "': ", mapped_, " pages counted after discarding all");
    }

    ~PagedDeviceBuffer() { release(); }
    PagedDeviceBuffer(PagedDeviceBuffer &&o) noexcept { swap(o); }
    PagedDeviceBuffer &operator=(PagedDeviceBuffer &&o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }
    PagedDeviceBuffer(const PagedDeviceBuffer &) = delete;
    PagedDeviceBuffer &operator=(const PagedDeviceBuffer &) = delete;

    // Bytes [off, off + bytes) of the array (bytes 0: none).
    struct Range {
        size_t off = 0, bytes = 0;
    };

    uint8_t *get() const {  // the current generation's array (gen_ = 0 without generations)
        return static_cast<uint8_t *>(base_) + (size_t)gen_ * gen_stride_;
    }
    size_t capacity() const { return reserved_; }  // bytes the range can ever map
    size_t size() const { return mapped_ * page_; }  // bytes mapped now (whole pages)
    size_t page_bytes() const { return page_; }
    const std::string &name() const { return name_; }
    // The bytes ensure(ranges) would map (pages they touch that aren't mapped): what the caller asks the memory budget
    // for before growing.
    size_t missing_bytes(const std::vector<Range> &ranges) const {
        std::vector<char> seen(pages_.size(), 0);  // ranges may share a page: count it once
        size_t n = 0;
        for_pages(ranges, "missing_bytes", [&](size_t p) {
            if (!seen[p] && pages_[p] == nullptr) seen[p] = 1, n += page_;
        });
        return n;
    }
    size_t grow_bytes(size_t bytes) const { return missing_bytes({{0, bytes}}); }
    // The bytes of every page the ranges touch, mapped or not (what they cost when fully mapped).
    size_t touched_bytes(const std::vector<Range> &ranges) const {
        std::vector<char> seen(pages_.size(), 0);
        size_t n = 0;
        for_pages(ranges, "touched_bytes", [&](size_t p) {
            if (!seen[p]) seen[p] = 1, n += page_;
        });
        return n;
    }
    // Maps every page the ranges touch. Mapped pages keep their contents; new pages' contents are undefined (the
    // caller writes before it reads, as with a fresh DeviceBuffer).
    void ensure(const std::vector<Range> &ranges) {
        for_pages(ranges, "ensure", [&](size_t p) {
            if (pages_[p] == nullptr) map_page(p);
        });
    }
    void ensure(size_t bytes) { ensure({{0, bytes}}); }

private:
    hipMemAllocationProp page_prop() const {
        hipMemAllocationProp p = {};
        p.type = hipMemAllocationTypePinned;
        p.location.type = hipMemLocationTypeDevice;
        p.location.id = dev_;
        return p;
    }
    template <typename F>
    void for_pages(const std::vector<Range> &ranges, const char *what, F &&f) const {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': ", what, " on an empty (moved-from or default) buffer");
        for (const Range &r : ranges) {
            STRIX_CHECK(r.off <= reserved_ && r.bytes <= reserved_ - r.off, "paged buffer '", name_, "': ", what, " of [",
                        r.off, ", +", r.bytes, " B) past its capacity of ", reserved_, " B");
            if (r.bytes == 0) continue;
            for (size_t p = r.off / page_; p <= (r.off + r.bytes - 1) / page_; ++p) f(p);
        }
    }
    void map_page(size_t p) {
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = dev_;
        access.flags = hipMemAccessFlagsProtReadWrite;
        const hipMemAllocationProp prop = page_prop();
        char *at = static_cast<char *>(base_) + (size_t)gen_ * gen_stride_ + p * page_;
        hipMemGenericAllocationHandle_t h{};
        STRIX_HIP_CHECK(hipMemCreate(&h, page_, &prop, 0), "paged buffer '", name_, "': creating page ", p, " (", page_,
                        " B; ", size(), " of ", reserved_, " B mapped)");
        gen_used_ = true;  // from the first map attempt: a failed map may still have touched the address
        const hipError_t m = hipMemMap(at, page_, 0, h, 0);
        if (m != hipSuccess) {
            (void)hipMemRelease(h);
            STRIX_HIP_CHECK(m, "paged buffer '", name_, "': mapping page ", p);
        }
        const hipError_t a = hipMemSetAccess(at, page_, &access, 1);
        if (a != hipSuccess) {
            (void)hipMemUnmap(at, page_);
            (void)hipMemRelease(h);
            STRIX_HIP_CHECK(a, "paged buffer '", name_, "': device access for page ", p);
        }
        pages_[p] = h;
        ++mapped_;
        GpuMemoryTally::add(name_, (int64_t)page_);
    }
    void release() noexcept {  // destructor path: no throwing; a failed unmap leaks rather than aborts
        if (base_ == nullptr) return;
        for (size_t p = 0; p < pages_.size(); ++p) {
            if (pages_[p] == nullptr) continue;
            if (hipMemUnmap(static_cast<char *>(base_) + (size_t)gen_ * gen_stride_ + p * page_, page_) == hipSuccess)
                (void)hipMemRelease(pages_[p]);
            GpuMemoryTally::add(name_, -(int64_t)page_);
        }
        pages_.clear();
        mapped_ = 0;
        if (!shared_) (void)hipMemAddressFree(base_, reserved_);  // a shared reservation is its owner's to free
        base_ = nullptr;
        reserved_ = 0;
    }
    void swap(PagedDeviceBuffer &o) noexcept {
        std::swap(base_, o.base_);
        std::swap(reserved_, o.reserved_);
        std::swap(page_, o.page_);
        std::swap(dev_, o.dev_);
        std::swap(pages_, o.pages_);
        std::swap(mapped_, o.mapped_);
        std::swap(name_, o.name_);
        std::swap(shared_, o.shared_);
        std::swap(offset_, o.offset_);
        std::swap(res_bytes_, o.res_bytes_);
        std::swap(gen_stride_, o.gen_stride_);
        std::swap(gen_, o.gen_);
        std::swap(gen_cap_, o.gen_cap_);
        std::swap(gen_used_, o.gen_used_);
    }

    void *base_ = nullptr;
    size_t reserved_ = 0, page_ = 0, mapped_ = 0;
    int dev_ = 0;
    std::vector<hipMemGenericAllocationHandle_t> pages_;  // one a page of the range; null: unmapped
    std::string name_;
    bool shared_ = false;  // carved from a VmmReservation: its address range is the reservation's to free
    size_t offset_ = 0, res_bytes_ = 0;  // the carve's place in the shared reservation (the generations fit check)
    size_t gen_stride_ = 0;              // bytes between two generations of this array (0: generations not enabled)
    uint32_t gen_ = 0, gen_cap_ = 1;     // the current generation, the reservation's capacity of them
    bool gen_used_ = false;              // anything mapped (or attempted) in the current generation
};

}  // namespace strix
