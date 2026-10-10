// IQ3_XXS weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// Per-sub-group decoder for IQ3_XXS, the family's first 4-COMPONENT grid. A
// sub-group needs TWO grid points: sub-group `s` is elements `8s .. 8s+7`,
// element `8s+j` takes its magnitude from `IQ3XXS_GRID[qs[2s]]` component `j`
// for `j < 4` and from `IQ3XXS_GRID[qs[2s+1]]` component `j - 4` above that.
// The CPU dequantizer reaches the same two bytes as
// `qs[group * 8 + sub * 2 (+1)]` with `group = s / 4` and `sub = s % 4`, and
// `(s / 4) * 8 + (s % 4) * 2` is `2s`.
//
// Its sign byte is `KSIGNS[(aux >> (7 * sub)) & 0x7F]`, where `aux` is the
// group's u32 at `scales + 4 * (s / 4)`.
//
// The two index bytes are adjacent and `qs`@2 keeps `2 + 2s` even, so they are
// read as ONE aligned u16 — low byte `qs[2s]`, high byte `qs[2s+1]` on a
// little-endian device. The block is 98 bytes, so a row base and every block
// base inside it are only 2-byte aligned; the `aux` read is 4 bytes and
// therefore goes through `load_int_ua`. A plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it.
struct MmqfIQ3XXSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 4;
    struct Regs {
        unsigned int qs;
        int aux;
    };
    unsigned long long off_qs;
    unsigned long long off_aux;
    unsigned int sub;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = *reinterpret_cast<const unsigned short*>(row + off_qs);
        r.aux = load_int_ua(row + off_aux);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        mag[0] = IQ3XXS_GRID[r.qs & 0xFFu];
        mag[1] = IQ3XXS_GRID[r.qs >> 8];
        signs = KSIGNS[((unsigned int)r.aux >> (7 * sub)) & 0x7Fu];
    }
};

// IQ3_XXS weight format policy, same contract as `MmqfQ80`.
//
// IQ3_XXS is a 256-element block of 98 bytes: `f16 d`@0, `qs[64]`@2, then
// `scales[32]`@66 read as eight little-endian u32, one per 32-element group.
// Each group's u32 carries its 4-bit scale in the TOP nibble over four 7-bit
// indices into `KSIGNS`, one sign byte per 8-element sub-group.
//
// GRID WIDTH. This is the family's first 4-COMPONENT grid: each `qs` byte
// indexes one of the 256 points of `IQ3XXS_GRID`, and a point expands to FOUR
// magnitude bytes, not eight. So an 8-element sign sub-group spans TWO
// consecutive `qs` bytes — the low one for elements 0..3 of the sub-group, the
// high one for 4..7 — where the IQ2 formats spent one grid point on the whole
// sub-group. `MmqfIQ3XXSEntry` absorbs that; the staging skeleton is the same
// one at both widths, since a sub-group is two staged words either way.
//
// Staging EXPANDS the two grid points to signed int8 and folds the sign in, so
// the staged row is Q8_0's unchanged and `vec_dot` is Q8_0's. The grid's
// magnitudes run 4..62, so a negated component still fits an int8 lane.
//
// SCALE GRANULARITY. The 4-bit scale is one per `aux` word and an `aux` word
// covers one group of 32 elements, which is exactly the granularity
// `mmqf_vec_dot_d` already indexes at. So this format stages 8 f32 rather than
// Q6_K's 16 and needs no wider row. The scale is `d * (0.5 + s) * 0.5` — the
// trailing factor is 0.5, not the IQ2 family's 0.25.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.5` to it
// perturbs every group's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// ALIGNMENT. The block is 98 bytes, so a row base (`supers * 98`) and every
// block base inside it are only 2-byte aligned. Every 4-byte read of an `aux`
// word — in the quant pass and in the scale pass — therefore goes through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS,
// which poisons the context for every later launch on it. The `f16 d`@0 is
// 2-byte aligned, which is `alignof(__half)`, and is read directly, as is the
// u16 index pair at the even offset `2 + 2 * lane`.
struct MmqfIQ3XXS {
    // On-disk block: one f16 scale, 64 grid-index bytes, 32 bytes of packed
    // scales and sign-table indices.
    static constexpr int BLOCK_BYTES = 98;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offset of `scales[32]` within the block.
    static constexpr int SCALES_OFF = 66;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 group scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 group scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ3_XXS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq3_xxs` in
    // `src/quant/cpu/kernels/dequant_iq3.rs` and matching the per-block decoder
    // `iq3_xxs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`. Lane `l` owns the
    // sign sub-group `s = 4 * group + sub = l`: its two index bytes are
    // `qs[2l]` and `qs[2l + 1]`, its sign byte is
    // `KSIGNS[(aux >> (7 * (l % 4))) & 0x7F]` for the `aux` word at
    // `scales + 4 * (l / 4)`, and it fills staged words `2l` and `2l + 1`.
    // Under that map a staged word `w` belongs to group `w / 8`, which is
    // exactly the scale index `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
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
        MmqfIQ3XXSEntry dec;
        dec.off_qs = off_blk + 2 + (unsigned long long)lane * 2;
        dec.off_aux = off_blk + SCALES_OFF + (unsigned long long)(lane / 4) * 4;
        dec.sub = lane % 4;
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's `aux` word are at row-invariant offsets, so the first loop is
        // pure loads and the 4-bit scale is unpacked in the second, on
        // registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + SCALES_OFF + (unsigned long long)sb * 4;

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
            // group scale is `d * (0.5 + s) * 0.5`.
            const float s = (float)((unsigned int)aux[u] >> 28);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (0.5f + s) * 0.5f;
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once staging has expanded
    // the two grid points and folded the sign in, the two rows are
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
