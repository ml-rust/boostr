// IQ2_XS weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_sc16.cuh"
#include "q6_k.cuh"

// Per-entry decoder for IQ2_XS. Entry `e` is one little-endian u16 at
// `qs + 2 * e`: its low 9 bits index `IQ2XS_GRID` (512 points) and its top 7
// bits index `KSIGNS`.
//
// The block is 74 bytes and `qs` starts at byte 2, so the entry's address is
// even and the 2-byte load is aligned. This is a 2-byte read, not a 4-byte one,
// so `load_int_ua` does not apply; the 32 lanes of a warp read 64 contiguous
// bytes.
struct MmqfIQ2XSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    typedef unsigned short Regs;
    unsigned long long off_q;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        return *reinterpret_cast<const unsigned short*>(row + off_q);
    }
    __device__ __forceinline__ void decode(
        const Regs& q, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2XS_GRID[(unsigned int)q & 511u];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = KSIGNS[(unsigned int)q >> 9];
    }
};

// IQ2_XS weight format policy, same contract as `MmqfQ80`.
//
// IQ2_XS is a 256-element block of 74 bytes: `f16 d`@0, `qs[64]`@2 read as 32
// little-endian u16, then `scales[8]`@66. Each u16 packs a 9-bit index into
// `IQ2XS_GRID` (512 points, eight magnitude bytes each) in its low bits and a
// 7-bit index into `KSIGNS` above them. Reading `qs` as packed 2-bit magnitudes
// still yields finite, plausibly scaled numbers, which is why both tables live
// in exactly one place (`iq_grid.cuh`) for every CUDA kernel.
//
// Staging EXPANDS the grid point to signed int8 and folds the sign in, through
// `mmqf_stage_iq_grid`, so the staged quant lanes are Q8_0's and IQ2_XXS's.
//
// SCALE GRANULARITY. This is where IQ2_XS parts company with IQ2_XXS. Its
// 4-bit scale is packed two per `scales` byte and `packed_scale` indexes it at
// `entry / 2`, so it changes every SIXTEEN elements, not every 32. A single
// `mma_m16n8k32_s8` cannot express one 32-k step under that, so the staged row
// is Q6_K's — 64 quant words then 16 f32 `d * (0.5 + s) * 0.25` — and
// `vec_dot` is `mmqf_vec_dot_sc16`, which runs two `mma_m16n8k16_s8` calls per
// 32-k step.
//
// ALIGNMENT. The block is 74 bytes, so a row base (`supers * 74`) and every
// block base inside it are only 2-byte aligned. There is no 4-byte read here:
// the entry is one u16 at an even offset and the scale is one byte, so
// `load_int_ua` does not apply. `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2XS {
    // On-disk block: f16 scale, 64 bytes of packed index/sign u16, 8 bytes of
    // packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 74;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offset of `scales[8]` within the block.
    static constexpr int SCALES_OFF = 66;

    // Staged weight row: Q6_K's, int for int. 64 quant words (256 k-values),
    // then 16 f32 group scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "IQ2_XS must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_xs` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_xs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `entry = e / 8`, `j = e % 8`. Its u16 is at `qs + 2 * entry`, `j` picks
    // the magnitude byte of the grid entry and the sign bit, and its scale is
    // group `entry / 2`. Lane `l` owns entry `l` and fills staged words `2l`
    // and `2l + 1`, so staged word `w` sits in scale group `w / 4` — the
    // property `mmqf_vec_dot_sc16` relies on.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ2XSEntry dec;
        dec.off_q = off_blk + 2 + (unsigned long long)lane * 2;
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        mmqf_stage_iq2_packed_scales<MMQ_Y, MMQ_X, X_DF, X_STRIDE, SCALES_OFF>(
            weight, s_x, i_max, rstride, feat0, off_blk
        );
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K and Q3_K: once staging
    // has expanded the grid point and folded the sign in, the rows are
    // indistinguishable, and the per-16 scale arithmetic is the same.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};
