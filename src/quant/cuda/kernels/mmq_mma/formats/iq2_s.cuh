// IQ2_S weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_sc16.cuh"
#include "q6_k.cuh"

// Per-entry decoder for IQ2_S, the one divergence in the group. Entry `e`
// takes its low 8 index bits from `qs[e]` and two more from the field
// `2 * (e % 4)` of `qh[e / 4]`, selecting among 1024 grid points, and its sign
// byte is `signs[e]` DIRECTLY — explicit bits, with no `KSIGNS` indirection.
//
// All three reads are single bytes, so no alignment requirement arises and
// `load_int_ua` does not apply. Across a warp the `qs` and `signs` reads each
// cover 32 contiguous bytes and the `qh` read 8, so all three coalesce.
struct MmqfIQ2SEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        unsigned int qs;
        unsigned int sign;
        unsigned int qh;
    };
    unsigned long long off_qs;
    unsigned long long off_sign;
    unsigned long long off_qh;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = row[off_qs];
        r.sign = row[off_sign];
        r.qh = row[off_qh];
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2S_GRID[r.qs | (((r.qh >> qh_shift) & 0x03u) << 8)];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = r.sign;
    }
};

// IQ2_S weight format policy, same contract as `MmqfQ80`.
//
// IQ2_S is a 256-element block of 82 bytes: `f16 d`@0, `qs[32]`@2,
// `signs[32]`@34, `qh[8]`@66, `scales[8]`@74. Entry `e` takes its low 8 index
// bits from `qs[e]` and two more from the field `2 * (e % 4)` of `qh[e / 4]`,
// selecting among the 1024 points of `IQ2S_GRID`.
//
// SIGNS. This is the divergence from IQ2_XXS and IQ2_XS. The sign byte is
// `signs[e]` directly — eight explicit bits per entry — with no `KSIGNS`
// indirection. `MmqfIQ2SEntry` absorbs that difference; everything downstream
// of it, including the branchless fold in `mmqf_stage_iq_grid`, is unchanged.
//
// SCALE GRANULARITY. Identical to IQ2_XS: a 4-bit scale per SIXTEEN elements,
// packed two per `scales` byte. So the staged row is Q6_K's and `vec_dot` is
// `mmqf_vec_dot_sc16`, and the scale pass is the one IQ2_XS uses, at the
// format's own `scales` offset.
//
// ALIGNMENT. The block is 82 bytes, so a row base (`supers * 82`) and every
// block base inside it are only 2-byte aligned. There is no 4-byte read here:
// `qs`, `signs`, `qh` and `scales` are all read one byte at a time, so
// `load_int_ua` does not apply. `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2S {
    // On-disk block: f16 scale, 32 index bytes, 32 sign bytes, 8 index-high
    // bytes, 8 bytes of packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 82;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields an entry reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int SIGNS_OFF = 34;
    static constexpr int QH_OFF = 66;
    static constexpr int SCALES_OFF = 74;

    // Staged weight row: Q6_K's, int for int, same as IQ2_XS.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "IQ2_S must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_s` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_s_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `entry = e / 8`, `j = e % 8`. Its index byte is at `qs + entry`, its two
    // index-high bits at field `2 * (entry % 4)` of `qh[entry / 4]`, its sign
    // byte at `signs + entry`, and its scale is group `entry / 2`. Lane `l`
    // owns entry `l` and fills staged words `2l` and `2l + 1`, so staged word
    // `w` sits in scale group `w / 4` — the property `mmqf_vec_dot_sc16`
    // relies on.
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
        MmqfIQ2SEntry dec;
        dec.off_qs = off_blk + QS_OFF + lane;
        dec.off_sign = off_blk + SIGNS_OFF + lane;
        dec.off_qh = off_blk + QH_OFF + lane / 4;
        dec.qh_shift = 2 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        mmqf_stage_iq2_packed_scales<MMQ_Y, MMQ_X, X_DF, X_STRIDE, SCALES_OFF>(
            weight, s_x, i_max, rstride, feat0, off_blk
        );
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K, Q3_K and IQ2_XS: once
    // staging has expanded the grid point and folded the explicit sign in, the
    // rows are indistinguishable, and the per-16 scale arithmetic is the same.
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
