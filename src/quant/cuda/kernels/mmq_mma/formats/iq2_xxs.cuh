// IQ2_XXS weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// Per-entry decoder for IQ2_XXS. Entry `e` sits in group `e / 4` at sub-index
// `e % 4`: its group's index word is at `qs + 8 * (e / 4)` and the group's
// `aux` word 4 bytes further on. `e % 4` picks the byte of the index word and
// the 7-bit `KSIGNS` field of `aux`.
//
// The block is 66 bytes, so a row base and every block base inside it are only
// 2-byte aligned, and `qs`@2 with them. Both 4-byte reads therefore go through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS, which
// poisons the context for every later launch on it.
struct MmqfIQ2XXSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        int ind;
        int aux;
    };
    unsigned long long off_ind;
    unsigned int sub;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.ind = load_int_ua(row + off_ind);
        r.aux = load_int_ua(row + off_ind + 4);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2XXS_GRID[((unsigned int)r.ind >> (8 * sub)) & 0xFFu];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = KSIGNS[((unsigned int)r.aux >> (7 * sub)) & 0x7Fu];
    }
};

// IQ2_XXS weight format policy, same contract as `MmqfQ80`.
//
// The family's first GRID-INDEXED format. IQ2_XXS is a 256-element block of 66
// bytes: `f16 d`@0 then `qs[64]`@2, read as eight pairs of u32. The first u32
// of a pair holds four 8-bit indices into `IQ2XXS_GRID`, whose entry expands to
// EIGHT magnitude bytes; the second holds the group's 4-bit scale in its top
// nibble over four 7-bit indices into `KSIGNS`, one sign bit per expanded
// component. Reading `qs` as packed 2-bit magnitudes still yields finite,
// plausibly scaled numbers, which is why both tables live in exactly one place
// (`iq_grid.cuh`) for every CUDA kernel.
//
// Staging EXPANDS the grid point to signed int8 and folds the sign in, so the
// staged row is Q8_0's unchanged — signed quants in the int8 lanes plus one f32
// scale per 32-element sub-block — and `vec_dot` is Q8_0's. The grid's only
// magnitudes are 8, 25 and 43, so a negated component still fits an int8 lane.
// A group of 32 elements is exactly one sub-block, which is the granularity
// `mmqf_vec_dot_d` already indexes at.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.25` to it
// perturbs every sub-block's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// ALIGNMENT. The block is 66 bytes, so a row base (`supers * 66`) and every
// block base inside it are only 2-byte aligned, and `qs`@2 with it. Every
// 4-byte read of `qs` therefore goes through `load_int_ua`; a plain `int` load
// raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every
// later launch on it. The `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2XXS {
    // On-disk block: one f16 scale then 64 bytes of grid indices, scales and
    // sign-table indices.
    static constexpr int BLOCK_BYTES = 66;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 sub-block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 sub-block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ2_XXS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_xxs` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_xxs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`. Its group's index
    // word is at `qs + 8 * group` and its `aux` word 4 bytes further on; `sub`
    // picks the byte of the index word and the 7-bit `KSIGNS` field of `aux`;
    // `j` picks the magnitude byte of the grid entry and the sign bit.
    //
    // A lane therefore owns one (group, sub) pair: `group = lane / 4`,
    // `sub = lane % 4`, which covers all 8 groups with 4 lanes each and needs
    // TWO global loads per lane. Since `entry = 4 * group + sub = lane`, that
    // is the one-entry-per-lane map `mmqf_stage_iq_grid` runs, and this
    // format's whole share of it is `MmqfIQ2XXSEntry`. Under that map a staged
    // word `w` belongs to group `w / 8`, which is exactly the scale index
    // `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

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
        MmqfIQ2XXSEntry dec;
        dec.off_ind = off_blk + 2 + (unsigned long long)(lane / 4) * 8;
        dec.sub = lane % 4;
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's `aux` word are at row-invariant offsets, so the first loop is
        // pure loads and the 4-bit scale is unpacked in the second, on
        // registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + 2 + (unsigned long long)sb * 8 + 4;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        __half d_h[SROWS];
        int aux[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            aux[u] = load_int_ua(row + off_sc);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // Top nibble of `aux` is the group's 4-bit scale `s`, and the
            // sub-block scale is `d * (0.5 + s) * 0.25`.
            const float s = (float)((unsigned int)aux[u] >> 28);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (0.5f + s) * 0.25f;
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once staging has expanded
    // the grid point and folded the sign in, the two rows are
    // indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
