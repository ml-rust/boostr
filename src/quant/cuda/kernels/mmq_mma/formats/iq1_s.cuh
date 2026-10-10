// IQ1_S weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_dm.cuh"
#include "q4_1.cuh"
#include "q4_k.cuh"

// Per-sub-group decoder for IQ1_S, the family's only SIGNED grid. Its grid
// width is the IQ2 formats': one 8-component point of `IQ1_GRID` supplies a
// whole 8-element sub-group, as a `u64` split into two magnitude words.
//
// Sub-group `s` is elements `8s .. 8s+7`, which is the CPU dequantizer's
// `(group, sub)` pair with `group = s / 4` and `sub = s % 4`. Its index byte is
// `qs[group * 4 + sub]`, which is `qs[s]`; its three high index bits are field
// `3 * (s % 4)` of the group's little-endian u16 `qh[s / 4]`. Together they
// form an 11-bit index into the 2048-point grid.
//
// SIGNS. There are none to fold: `IQ1_GRID` stores SIGNED bytes in
// `{-1, 0, 1}` and IQ1_S has no sign table, so `SIGNED_GRID` is true and
// `signs` is left untouched. The delta IQ1_S adds to every component is NOT
// handled here — it is a per-group affine term carried by the staged
// scale/min pair; see `MmqfIQ1S`.
//
// The index byte is a single byte and the `qh` field is a u16 at the even
// offset `34 + 2 * (s / 4)` inside a 50-byte block, so neither read is 4 bytes
// and `load_int_ua` does not apply. Across a warp the `qs` reads cover 32
// contiguous bytes and the `qh` reads 16, so both coalesce.
struct MmqfIQ1SEntry {
    // `IQ1_GRID`'s components are already signed, so the shared skeleton stores
    // the decoded words unchanged.
    static constexpr bool SIGNED_GRID = true;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        unsigned int qs;
        unsigned int qh;
    };
    unsigned long long off_qs;
    unsigned long long off_qh;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = row[off_qs];
        r.qh = *reinterpret_cast<const unsigned short*>(row + off_qh);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ1_GRID[r.qs | (((r.qh >> qh_shift) & 0x07u) << 8)];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        (void)signs;
    }
};

// IQ1_S weight format policy, same contract as `MmqfQ80`.
//
// IQ1_S is a 256-element block of 50 bytes: `f16 d`@0, `qs[32]`@2, `qh[16]`@34
// read as eight little-endian u16, one per 32-element group. Sub-group `s`
// (8 elements) takes its low 8 index bits from `qs[s]` and three more from
// field `3 * (s % 4)` of `qh[s / 4]`, an 11-bit index into the 2048-point
// `IQ1_GRID`. That group's u16 also carries a 3-bit scale at bits 12..14 and
// the group's delta sign at bit 15.
//
// THE AFFINE VALUE. This is what separates IQ1_S from every other format in
// the family. Its dequantized value is
//
//     dl * (g + delta),   dl = d * (2 * sc + 1),   delta = +/- 0.125
//
// with `g` an already-signed grid component. That is an AFFINE transform of
// the grid point, not a scale times an int8, so one scaled int32 dot cannot
// express it. Expanding a 32-element group's contribution against that group's
// activations `a`:
//
//     sum_j a_j * dl * (g_j + delta) = dl * dot(a, g) + dl * delta * sum(a)
//
// Both terms are per 32-element group, which is exactly the shape
// `mmqf_vec_dot_dm` already computes for Q4_K, Q5_K, Q4_1 and Q5_1: one
// `(scale, min)` f32 pair per 32 elements, the first against the int32 dot and
// the second against the activation block sum. So IQ1_S stages the Q4_K row
// and needs NO new `vec_dot` — it stores `(dl, dl * delta)` where those
// formats store their `(d, m)`.
//
// The rewrite is EXACT because the block sum `mmqf_vec_dot_dm` multiplies by
// is the int16 the activation producer stored in the header word's high half,
// which is the true sum of that sub-block's 32 int8 quants. Folding `delta`
// into a `half` field on the activation side instead would not be.
//
// THE MIN SIGN. `mmqf_vec_dot_dm` always ADDS `pair.y * (activation scale *
// block sum)`. IQ1_S's delta term is additive, so it stores `+dl * delta` —
// Q4_1's and Q5_1's convention, not Q4_K's negated `-dmin * m`. The delta's
// own sign, bit 15 of the group's u16, is folded into the stored value at
// staging.
//
// SCALE GRANULARITY. `dequant_iq1_s` computes `dl` and `delta` once per
// `group` and holds both across all four of that group's sub-blocks, so one
// pair covers `4 * 8 = 32` elements. A 256-k staging group therefore carries
// exactly 8 pairs, which is the granularity `mmqf_vec_dot_dm` indexes with
// `kw / 8`.
//
// SIGNS. `IQ1_GRID` stores signed components and IQ1_S has no sign table, so
// there is nothing to fold; `MmqfIQ1SEntry` sets `SIGNED_GRID`.
//
// ALIGNMENT. The block is 50 bytes, so a row base (`supers * 50`) and every
// block base inside it are only 2-byte aligned. This format issues NO 4-byte
// read: `qs` is read one byte at a time and `qh` as a u16 at the even offset
// `34 + 2 * group`. So `load_int_ua` does not apply to it. The `f16 d`@0 is
// 2-byte aligned, which is `alignof(__half)`, and is read directly.
struct MmqfIQ1S {
    // On-disk block: one f16 scale, 32 index bytes, then 16 bytes holding the
    // eight per-group u16 of index-high fields, scale and delta sign.
    static constexpr int BLOCK_BYTES = 50;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // The delta term is per 32 elements, the granularity the activation
    // record's own block sum already carries, so no per-16 split and no
    // activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields a sub-group reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int QH_OFF = 34;

    // Staged weight row: Q4_K's, int for int. 64 quant words (256 k-values, one
    // SIGNED grid component per int8 lane), then 8 `float2` holding
    // `(dl, dl * delta)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ4K::X_QS && X_DM == MmqfQ4K::X_DM && X_STRIDE == MmqfQ4K::X_STRIDE,
        "IQ1_S must stage into the Q4_K row; the two share `mmqf_vec_dot_dm`."
    );
    static_assert(
        X_QS == MmqfQ41::X_QS && X_DM == MmqfQ41::X_DM && X_STRIDE == MmqfQ41::X_STRIDE,
        "IQ1_S takes Q4_1's ADDITIVE min convention, so it stages Q4_1's row too."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq1_s` in
    // `src/quant/cpu/kernels/dequant_iq1.rs` and matching the per-block decoder
    // `iq1_s_dequant_block` in `iq_dequant.cuh`. Element `e` is
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`, so its sub-group is
    // `s = e / 8 = group * 4 + sub`. Lane `l` owns sub-group `l`: index byte
    // `qs[l]`, `qh` word `l / 4`, index-high shift `3 * (l % 4)`. It fills
    // staged words `2l` and `2l + 1`, so a staged word `w` belongs to the
    // 32-element group `w / 8` — exactly the pair index `mmqf_vec_dot_dm`
    // reads at `(X_DM / 2) + kw / 8`.
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
        MmqfIQ1SEntry dec;
        dec.off_qs = off_blk + QS_OFF + lane;
        dec.off_qh = off_blk + QH_OFF + (unsigned long long)(lane / 4) * 2;
        dec.qh_shift = 3 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale/min pass: eight pairs per row, so a warp covers four rows. `d`
        // and the group's u16 are at row-invariant offsets, so the first loop
        // is pure loads and every bitfield is unpacked in the second, on
        // registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group within the block
        const unsigned int rsub = lane / 8;
        const unsigned long long off_h = off_blk + QH_OFF + (unsigned long long)sb * 2;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        __half d_h[SROWS];
        unsigned int h[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            h[u] = *reinterpret_cast<const unsigned short*>(row + off_h);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // Bits 12..14 hold the 3-bit scale, bit 15 the delta sign.
            const float dl = __half2float(d_h[u]) * (2.0f * (float)((h[u] >> 12) & 7u) + 1.0f);
            const float delta = (h[u] & 0x8000u) == 0 ? IQ1_DELTA : -IQ1_DELTA;
            // `+dl * delta`, never negated: `mmqf_vec_dot_dm` ADDS `pair.y`
            // times the activation block sum and this format's delta term is
            // additive. Both components are f32 for the same parity reason as
            // Q4_K — no `half` round-trip on a term the GEMM/GEMV bound is
            // sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + sb] = make_float2(dl, dl * delta);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K, Q5_K, Q4_1 and Q5_1:
    // once staging has expanded the signed grid point and split the affine
    // value into `(dl, dl * delta)`, the row and the two-term arithmetic are
    // Q4_1's.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
