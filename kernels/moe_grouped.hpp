#pragma once

// MoE experts for multi-token forwards (prefill, MTP verify): tokens grouped by expert, so each active expert's Q4
// weights are read once per tile of up to kMoeTileRows slots instead of once per token. The decode kernels
// (kernels/linear_q4.hpp linear_q4_experts_gather / _combine) read an expert's rows once per (token, slot): at
// T = 512 the 5120 slots of a layer touch ~all 512 experts ~10 times each, 55% of prefill time (rocprofv3). Same semantics as those kernels - FP32 math, one rounding where the output is stored - only the
// summation order differs:
//   gather:  y[m, a, :] = x[m, :] * W[ids[m, a]]^T                      x [M, K], y [M, A, N]
//   combine: y[m, :] = sum_a coef[m, a] * (h[m, a, :] * W[ids[m, a]]^T)  h [M, A, K], y [M, N]
//            (the per-slot dots kept FP32 in a workspace, then summed over a in order, rounded once).
//
// Three steps per MoE layer: moe_group_routes (once; both projections share the grouping), then the grouped
// gather and combine. An id outside [0, E) sets err like the decode kernels (err[0] = 1, err[1] = the id, err[2] =
// its slot m * A + a; the first flag wins; check_expert_error() throws) and that slot's outputs - for combine,
// its token's row - are NaN, never an out-of-bounds read. MoeMath::WmmaBf16 needs K % 64 == 0 (the target's K
// are 2560 and 640).

#include "kernels/norm.hpp"  // Act
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::kernels {

// How the grouped GEMMs compute: FP32 on the vector units (the engine's default semantics), or on the matrix units
// with x rounded to BF16, the weights exact (scaled codes) and FP32 accumulation (kernels/wmma_gemm.hpp - a
// lower-precision path, behind a switch).
// WmmaF16 (Q4 / Q5 experts): x as F16 (exact from BF16 in range), the weights dequantized
// to F16 in registers (code * scale + min, one F16 rounding; codes 0..31 exact), FP32 accumulation over the whole K.
// WmmaF16 picks one of three kernels with the same math (bit-identical outputs) by the call's token count M:
// WmmaF16Narrow - 64-slot tiles, each wave decoding its own 16 weight rows in registers -, WmmaF16Wide - the dense
// GEMMs' tile structure, 128-slot x 128-row blocks with each weight tile dequantized once into LDS and read by the
// waves sharing it - or WmmaF16Wide256 - 128-slot x 256-row blocks, 4 x 4 output tiles a wave -, the wide ones faster
// once the experts' tiles are full (moe_grouped.hip, kWideMinM*). Narrow / Wide / Wide256 force one (tests, benches).
enum class MoeMath { F32, WmmaBf16, WmmaF16, WmmaF16Narrow, WmmaF16Wide, WmmaF16Wide256 };

constexpr int kMoeTileRows = 16;       // slots per tile (one expert's rows; a weight pass serves them all)
constexpr int kMoeWmmaTileRows = 64;   // the WMMA kernels' tiles: up to 4 16-row WMMA tiles per weight pass
constexpr int kMoeWmmaWideRows = 128;  // MoeMath::WmmaF16Wide's tiles: up to 8 16-row WMMA tiles per weight pass
constexpr int64_t kMoeMaxExperts = 1023;  // the grouping runs in one 1024-thread block: E + 1 buckets
constexpr int64_t kMoeMaxSlots = 1 << 20;

// Bytes of grouping workspace for M tokens x A slots over E experts.
size_t moe_route_workspace_bytes(int64_t M, int64_t A, int64_t E);

// Groups the M*A slots of ids [M, A] (device, int32) by expert into ws (moe_route_workspace_bytes(M, A, E) bytes,
// device): per expert its slots (in an order set by atomics - no output depends on it), cut into tiles of up to
// kMoeTileRows (the FP32 kernels) and, separately, of up to kMoeWmmaTileRows (the WMMA kernels).
void moe_group_routes(const int32_t *ids, int64_t M, int64_t A, int64_t E, void *ws, size_t ws_bytes, uint32_t *err,
                      hipStream_t stream);

// Gather over grouped routes (ws from moe_group_routes with the same ids, M, A, E). w: E experts stacked
// [E*N, K]. x, y in the activation dtype; x 16-byte aligned; y must not overlap x.
void linear_q4_experts_gather_grouped(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream);

// The gather with SwiGLU fused (MoeMath::WmmaF16 kinds): h [M*A, N/2] = silu(gate) * up of the stacked gate_up
// rows (each expert's N rows: N/2 gate rows, then N/2 up rows), from the FP32 accumulators, rounded once to the
// activation dtype - in place of the gather into gu [M*A, N] and kernels/swiglu. h must not overlap x. h_f16 (BF16
// activations): h stored as F16 bits of its BF16 rounding (what the F16 down kernels stage from BF16 h, round toward
// zero), for linear_q4_experts_combine_grouped with h_f16 - the down's x conversion done once, bit-identical.
void linear_q4_experts_gather_grouped_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                             size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                             hipStream_t stream, bool h_f16 = false);

// The prefill MoE sublayer's last step folded into the combine's reduce: y[m, :] += sigmoid(gate_logit[m * gate_ld]) *
// s[m, :] on the stored y, as kernels/moe_router.hpp moe_shared_add computes it - bit-identical to the combine then
// moe_shared_add, without y's extra write and read. A non-finite gate logit sets err as moe_shared_add does (err[0] = 1,
// err[1] = the token, err[2] = gate_index) and makes that token's row NaN.
struct MoeSharedAdd {
    const void *s = nullptr;            // the shared expert's output [M, N], the activation dtype, not overlapping y
    const float *gate_logit = nullptr;  // token m's gate logit at gate_logit[m * gate_ld]
    int64_t gate_ld = 0;
    uint32_t gate_index = 0;
    uint32_t *err = nullptr;
};


// Combine over grouped routes. partial: M*A*N FP32 scratch (device) - MoeMath::WmmaF16 stores its partials as BF16
// in the first half of it (moe_grouped.hip, put_partial). ids: the same ids (the reduction checks
// them per token); coef [M, A] F32. h 16-byte aligned; y must not overlap h or partial. h_f16 (BF16 activations, a
// MoeMath::WmmaF16 kind): h holds F16 bits (linear_q4_experts_gather_grouped_swiglu's h_f16).
void linear_q4_experts_combine_grouped(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                       const MoeSharedAdd *shared = nullptr, bool h_f16 = false);

// The same with Q5 experts (formats/q5.hpp rows, K % 128 == 0) - one kernel template with the Q4 ones (every MoeMath).
void linear_q5_experts_gather_grouped(const void *x, const Q5DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream);
void linear_q5_experts_gather_grouped_swiglu(const void *x, const Q5DeviceView &w, int64_t E, const void *ws,
                                             size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                             hipStream_t stream, bool h_f16 = false);
void linear_q5_experts_combine_grouped(const void *h, const Q5DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                       const MoeSharedAdd *shared = nullptr, bool h_f16 = false);

}  // namespace strix::kernels
