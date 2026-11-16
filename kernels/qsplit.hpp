#pragma once

// Row layout of the Q4-based expert weights as the expert kernels read it: per row K/2 bytes of 4-bit low codes
// (formats/q4.hpp's nibble order), then - for Q5 (formats/q5.hpp) - K/8 bytes of the codes' 5th bit. HB = the high
// bits per code: 0 (Q4; the high-bit code compiles away) or 1 (Q5). Q5 rows need K % 128 == 0 (16-byte aligned).

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

template <int HB>
__host__ __device__ constexpr int64_t qsplit_row_bytes(int64_t K) {
    static_assert(HB == 0 || HB == 1, "Q4 (HB 0) or Q5 (HB 1)");
    return K / 2 + K * HB / 8;
}

// The high bits of the 32 codes of a row starting at k (k % 32 == 0), already shifted into code bit 4.
template <int HB>
struct QHigh32 {
    uint32_t w = 0;
    __device__ static QHigh32 load(const uint8_t *row, int64_t K, int64_t k) {
        QHigh32 h;
        if constexpr (HB == 1) h.w = *reinterpret_cast<const uint32_t *>(row + K / 2 + k / 8);
        return h;
    }
    __device__ uint32_t at(int i) const {
        if constexpr (HB == 1) return ((w >> i) & 1u) << 4;
        else return 0u;
    }
};

// Of 8 Q5 codes' 5th bits h8 (code i's at bit i; bits 8+ ignored), the even codes' (parity 0) or the odd codes'
// (parity 1) as bit 4 of bytes 0..3 - code 2j + parity's in byte j, where the F16 decode holds code 2j + parity's low
// nibble (kernels/wmma_gemm.hpp wmma_q5_word_to_f16). (h8 >> parity) & 0x55 holds them at bits 2j; times 0x410410
// (bits 4, 10, 16, 22) moves bit 2j to 8j + 4. The other products land on 2i + 6j + 4 with i != j, never on a kept
// bit (i + 3j = 4j has no other solution in 0..3); the three positions two products share (10, 16, 22) carry into
// 11, 17, 23 only. The product stays below 2^29.
__host__ __device__ constexpr uint32_t q5_bit4_spread(uint32_t h8, int parity) {
    return (((h8 >> parity) & 0x55u) * 0x410410u) & 0x10101010u;
}

}  // namespace strix::kernels
