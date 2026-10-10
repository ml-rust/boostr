// Shared ground of the tensor-core MMQ family (`quant_mmq_<fmt>_q8_1_mma*`):
// the warp size, the unaligned load, the dp4a fallback, the feature-major
// tile constants and the dynamic shared-memory symbol. Every header under
// `mmq_mma/` includes it; it is not a compilation unit of its own. Each
// weight format compiles as its own translation unit, `mmq_mma/<fmt>.cu`,
// and is its own module `quant_mmq_mma_<fmt>`.
#pragma once

#include <cuda_fp16.h>
#include <type_traits>

#include "../mma_int8.cuh"

#define WARP_SIZE 32

// 2-byte-aligned 4-byte load (quant blocks are not always 4-byte aligned).
static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

// ---------------------------------------------------------------------------
// Quantized weight x Q8_1 -> f32 (MMQ), feature-major decomposition,
// specialized per weight format and batch size.
//
// Computes the same product as the token-major `quant_mmq_q8_0_q8_1_mma`
// (`q8_0.cu`), but with the two MMA operand roles swapped: the WEIGHT is
// operand A (16 output features per fragment) and the ACTIVATION is operand B
// (8 tokens per fragment). The block owns an MMQ_Y-feature x MMQ_X-token
// output tile, and the grid axes are transposed relative to that kernel.
//
// Four things follow, and they are the point of this variant:
//   - The weight tile is staged 256 k-values deep per outer iteration, so a
//     weight row is read in one wide pass instead of eight narrow ones. For a
//     256-element format that is exactly one super-block per pass.
//   - Shared memory is row-major (feature-major / token-major) with the block
//     scales inline in the same row, so a fragment gather is one strided read
//     and the scale it needs is in the same row.
//   - The activation tile is refilled once per 128 k, giving four barriers per
//     256 k-values rather than one pair per 32.
//   - Activations arrive in the repacked layout `quantize_f32_q8_1_mmq` emits,
//     k-group-major and token-minor, so staging one is a flat copy. `ntok` is
//     that layout's token stride; `y_packed` is its base.
//
// `MMQ_X` is the token tile, one compiled entry point per value. A small batch
// wants a small `MMQ_X`: it cuts both the activation tile and the accumulator
// register count, and it stops the grid from launching blocks whose token
// columns are entirely out of range. Host-side selection lives in Rust.
//
// `MMQ_Y` is the feature tile, a second template axis. Every body is generic
// in it; the block's warp count follows it, one warp per 16-feature MMA
// minitile row, so a warp's share of the tile is the same at every `MMQ_Y`.
// The default is `MMQF_Y_DEFAULT`, compiled for every format. A narrower
// tile doubles the feature-tile count of a fixed N, which is what a small-M,
// small-N product needs: at the default tile such a product launches fewer
// blocks than the device has SMs, and most of the device sits idle for the
// whole K walk. The narrow tile stages the same activation tile for half the
// features, so it trades activation reuse for that lost parallelism. At its
// wider token tiles it also compiles a second cadence, `_y64g`
// (`mmqf_accumulate_group`, selected by the `GROUP` template axis). Its
// blocks hold a whole 256-k activation group in the shared memory the
// halved weight tile frees. That halves the barriers per group. It is
// compiled only for the formats whose small-N shapes reach the starved
// regime, and the host picks it only there.
//
// Each `MMQ_X` compiles four entry points: `_x<N>` is tile-parallel (one block
// per output tile, K in one range), `_ms_x<N>` is tile-parallel over several
// K ranges in one block, and `_sk_x<N>` plus `_fixup_x<N>` are the split-K
// pair, for when the tile count is too small to fill the device. The last
// three walk K as the same `splits` ranges (`mmqf_split_range`) and sum the
// range partials in range order, and `_x<N>` is the one-range case of that
// sum, so for a fixed `splits` every output element receives the same float
// sequence from any schedule, at any M and any tile position. The host picks
// `splits` from K, N and the device alone. None of them is bit-identical to
// the pre-sm_80 dp4a kernel. A non-default feature tile inserts `_y<Y>`
// before `_x<N>` in all four names.
//
// Shared memory is DYNAMIC. The largest variants exceed the 48 KB static
// limit, so the launcher must opt in with
// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` before the launch and
// request `4 * (MMQ_Y * FMT::X_STRIDE + H * MMQ_X * (MMQF_Y_STRIDE +
// FMT::Y_SCRATCH))` bytes, where `H` is the number of 128-k halves the
// activation tile holds: two for a `GROUP` cadence entry point, else one.
// ---------------------------------------------------------------------------

// Feature tile the plain `MMQ_FM_KERNEL` list compiles: output features per
// block (MMA "rows", operand A). Bodies take the tile as `MMQ_Y`; only
// `MMQ_FM_KERNEL` in `entry_points.cuh` reads this define.
#define MMQF_Y_DEFAULT 128
// The narrow feature tile. Besides the halved weight tile it offers a second
// activation cadence at its wider token tiles: a `GROUP` entry point stages
// a whole 256-k activation group at once (see `mmqf_accumulate_group`). Its
// activation tile is twice the default tile's, and the host's
// shared-memory request must say so.
#define MMQF_Y_NARROW 64
// Single-warp feature tile: one 16-feature minitile per block, compiled at
// the token tiles a decode batch selects (`MMQ_FM_KERNEL_Y16`).
#define MMQF_Y_SMALL 16
// Narrowest token tile that compiles the full-group cadence. The doubled
// activation tile costs block co-residency, which repays itself only where
// per-group activation staging dominates the group: the wide token tiles.
// Below this the narrow tile compiles the halves cadence alone. Which of the
// two a wide token tile runs is the host's call, by grid size. Under one
// block per SM nothing co-resides and the halved barriers win. At or above
// it the halves cadence keeps the overlap the group cadence would lose.
#define MMQF_GROUP_CADENCE_X 40
// One warp per 16-feature MMA minitile row of the weight tile, so the warp
// count is fixed by the feature tile and every per-warp row count in the
// staging loops is the same at every `MMQ_Y`. Threads follow the warps; the
// launch's block size must equal `MMQF_THREADS_OF(MMQ_Y)`.
#define MMQF_WARPS_OF(Y) ((Y) / 16)
#define MMQF_THREADS_OF(Y) (MMQF_WARPS_OF(Y) * WARP_SIZE)

// Warp blocking. `granularity` sets how many MMA feature minitiles one warp
// owns (`ntx`) and therefore how the warps split the output tile: with ntx=1
// they stack on features, one minitile each, with ntx=2 they form a
// (warps/2)x2 grid and each warp's B fragment feeds two `mma` calls. `MMQ_X`
// must be a multiple of the granularity, which is what makes the variant list
// 8..40 step 8 and 48..128 step 16.
#define MMQF_GRAN(X) ((X) >= 48 ? 16 : 8)
#define MMQF_NTX(X) (MMQF_GRAN(X) / 8)
#define MMQF_NJ(X) ((X) / (8 * MMQF_NTX(X)))  // token groups per warp

// Word offset of the second 128-k half within a staged weight row.
#define MMQF_HALF_W 32
// `MMQF_ITER_B` (blocks per 256-k staging group) and `mmqf_split_range`,
// shared with the single-token kernel in `quant_mmq_gemv1.cu`.
#include "../mmq/split_range.cuh"
// Independent global loads a staging thread issues before the first shared
// store. Trades registers for memory-level parallelism; the staging loops stall
// on global latency, so this is the knob that moves that stall.
//
// It shrinks as MMQ_X grows because the accumulator is MMQ_X/2 floats and is
// live across the whole kernel: at the widest token tile there is no room left
// to hold a deep batch, and a batch that spills to local memory costs more than
// the latency it hides. Must divide the weight tile's row-iteration count.
#define MMQF_STAGE_BATCH(X) ((X) >= 96 ? 2 : ((X) >= 48 ? 4 : ((X) <= MMQF_SMALL_X ? 16 : 8)))
// Widest token tile that stages both 128-k activation halves of a group at
// once on every feature tile, and on the formats with a `Regs`/`load`/`store`
// split also walks K with the next group's words already in registers
// (`mmqf_accumulate_pipelined`). Above it the accumulator leaves no room for
// the prefetch. The host doubles the activation term of the shared-memory
// request up to this tile: must match `SMALL_X` in
// `src/quant/cuda/quant_matmul/mmq_feat_major/tiling/geometry.rs`.
#define MMQF_SMALL_X 32

// Activation row: 4 header words, then 32 quant words (128 k-values). Each
// header word covers one 32-value sub-block, `half` scale in bits 0..15 and
// int16 quant sum in bits 16..31 — see `quantize_f32_q8_1_mmq` in
// `quant_act.cu`. This is exactly one repacked record, which is what lets the
// tile be flat-copied.
#define MMQF_Y_DS 0
#define MMQF_Y_QS 4
#define MMQF_Y_STRIDE 36
// Per-token ints in the optional activation scratch region, which sits
// straight after each staged 128-k half's activation tile. One int per
// 32-value sub-block of that half. Only a format whose minimum term changes
// every 16 elements requests it — see `mmqf_stage_y_sums` and `MmqfQ2K`.
#define MMQF_Y_SCRATCH 4

static_assert(MMQF_Y_QS + 32 == MMQF_Y_STRIDE, "Wrong activation row stride.");

// Copied verbatim from `quant_gemv.cu`, including the `__CUDA_ARCH__ >= 610`
// guard, so the row-sum reductions in `mmqf_stage_y_sums` and in
// `quant_mmq_q4_k_q8_1_mma` match the dp4a kernel exactly.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

extern __shared__ int mmqf_smem[];

// Bounds handling in every format policy's `stage` (`formats/*.cuh`) is
// deliberate and is the reason those functions look the way they do.
//
// A bound must be folded into the ADDRESS, never into a branch around the load.
// The feature-row and token-row bounds are warp-uniform, so a guard on them
// compiles to a real branch, and a branch between unrolled iterations stops
// their loads batching: each iteration then waits on its own load before the
// next is issued. Clamping the index instead keeps every load independent, so
// a whole group is in flight at once.
//
// A clamped row re-reads the last valid row; the masked write-back discards it.
// A clamped k-block is staged but never consumed, because the tail limits how
// many k-steps `FMT::vec_dot` runs. `CLAMP_K` is false for whole 256-k groups,
// which is every iteration but the last of a ragged K, so the common path has
// no clamp at all.
//
// Each loop issues a group of independent loads into registers FIRST, then
// stores the group to shared. Interleaving load and store per iteration would
// serialize on the load.

