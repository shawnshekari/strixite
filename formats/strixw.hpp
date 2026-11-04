#pragma once

// strixw: the converted weights format the serving path loads.
// Header + index + data; every tensor already in its kernel's layout. StrixwWriter writes a planned set of
// tensors component by component (streaming: the converter never holds more than one tensor), then fills
// in the hashes; StrixwFile reads and validates one (mmap, like SafetensorsFile - the pinned bulk load for
// serving comes with the target runtime).

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strix {

enum class StrixwEncoding : uint8_t { Q4RowMajor = 1, Q4ChunkMajor = 2, BF16 = 3, F32 = 4, I64 = 5, Q8RowMajor = 6,
                                    Q8ChunkMajor = 7, Q6RowMajor = 8, Q5RowMajor = 9 };
enum class StrixwRole : uint8_t { Data = 0, Q = 1, Scale = 2, Min = 3 };
const char *strixw_encoding_name(StrixwEncoding e);
const char *strixw_role_name(StrixwRole r);

// v2 added Q8RowMajor, v3 Q8ChunkMajor, v4 Q6RowMajor and Q5RowMajor; the
// reader also accepts v1-v3 files (subsets).
constexpr uint32_t kStrixwVersion = 4, kStrixwOldestVersion = 1;
constexpr size_t kStrixwHeaderBytes = 128, kStrixwTensorAlign = 4096, kStrixwComponentAlign = 256;

struct StrixwPart {
    std::string source;  // checkpoint tensor name
    int64_t row_offset = 0, rows = 0;
};

struct StrixwComponent {
    StrixwRole role = StrixwRole::Data;
    uint64_t offset = 0, bytes = 0, hash = 0;  // offset: absolute, in the file
};

struct StrixwTensor {
    std::string name;
    StrixwEncoding encoding = StrixwEncoding::F32;
    std::vector<int64_t> shape;
    int64_t group_size = 0, experts = 0;
    std::vector<StrixwPart> parts;
    std::vector<StrixwComponent> components;  // roles + byte sizes from strixw_components(); offsets/hashes filled in
    int64_t numel() const;
    uint64_t bytes() const;  // sum of component bytes
};

// The components an encoding has, with their exact byte sizes for t's shape / group size (offset and hash
// zero). Throws if the shape doesn't suit the encoding (e.g. Q4 needs [N, K] with K % G == 0).
std::vector<StrixwComponent> strixw_components(const StrixwTensor &t);

// 64-bit hash of a byte range (4 interleaved multiply-xorshift lanes; not a standard hash - defined here).
uint64_t strix_hash64(const void *p, size_t n);

// The same hash fed in pieces: n is the TOTAL length (the lanes are seeded by it), update() takes any split,
// final() wants all of it. The prompt cache's writer needs this for a paged entry: its state bytes live in
// two places (the entry's small-state buffer and the parked pages) and the file's
// hash must be exactly the value strix_hash64 gives for those bytes assembled - strix_hash64 is this class
// over one update, so the two can never disagree.
class StrixHasher {
public:
    explicit StrixHasher(size_t n);
    void update(const void *data, size_t len);
    uint64_t final();  // throws unless every byte was fed
    size_t bytes_fed() const { return fed_; }

private:
    void consume_block(const uint8_t *p);  // one 32-byte block, four lanes
    size_t total_, fed_ = 0;
    uint64_t h_[4] = {0, 0, 0, 0};
    std::vector<uint8_t> carry_;  // < 32 bytes: the block boundaries are part of the hash
};

class StrixwWriter {
public:
    // Plans the whole file: offsets for every component of every tensor (components are taken from
    // strixw_components, so callers set name/encoding/shape/group_size/experts/parts only). Creates path
    // (truncating) and writes the header + index with zero hashes.
    StrixwWriter(const std::string &path, std::map<std::string, std::string> meta, std::vector<StrixwTensor> tensors);
    ~StrixwWriter();
    StrixwWriter(const StrixwWriter &) = delete;
    StrixwWriter &operator=(const StrixwWriter &) = delete;

    const std::vector<StrixwTensor> &tensors() const { return tensors_; }
    uint64_t file_bytes() const { return file_bytes_; }
    // Writes one component's bytes (exactly its planned size) and records its hash. Each once.
    void write(size_t tensor, StrixwRole role, const void *data, size_t bytes);
    // Checks every component was written, rewrites the index (hashes) and header, fdatasyncs, closes.
    void finish();

private:
    std::string path_;
    int fd_ = -1;
    std::map<std::string, std::string> meta_;
    std::vector<StrixwTensor> tensors_;
    std::vector<std::vector<bool>> written_;
    uint64_t index_bytes_ = 0, data_offset_ = 0, file_bytes_ = 0;
    bool finished_ = false;
    std::vector<uint8_t> encode_index() const;
    void write_header_and_index();
};

class StrixwFile {
public:
    explicit StrixwFile(const std::string &path);  // maps and validates (header, index, every record)
    ~StrixwFile();
    StrixwFile(const StrixwFile &) = delete;
    StrixwFile &operator=(const StrixwFile &) = delete;

    const std::string &path() const { return path_; }
    uint64_t file_bytes() const { return file_bytes_; }
    uint64_t data_offset() const { return data_offset_; }
    const std::map<std::string, std::string> &meta() const { return meta_; }
    const std::string &meta(const std::string &key) const;  // throws, naming the file, if absent
    const std::vector<StrixwTensor> &tensors() const { return tensors_; }
    const StrixwTensor *find(const std::string &name) const;
    const StrixwTensor &get(const std::string &name) const;  // throws if absent
    const void *data(const StrixwTensor &t, StrixwRole role) const;  // throws if t has no such component
    // Recomputes every component hash (threads in parallel); throws naming the first mismatch.
    void verify_hashes(int threads) const;

private:
    std::string path_;
    void *map_ = nullptr;
    uint64_t file_bytes_ = 0, data_offset_ = 0;
    std::map<std::string, std::string> meta_;
    std::vector<StrixwTensor> tensors_;
    std::map<std::string, size_t> by_name_;
};

}  // namespace strix
