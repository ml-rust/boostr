// IQ3_S weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../iq_grid.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// Per-sub-group decoder for IQ3_S: a 4-COMPONENT grid like IQ3_XXS, reached by
// a 9-bit index and paired with explicit sign bits.
//
// Sub-group `s` is elements `8s .. 8s+7`. Element `e` sits at grid entry
// `e / 4` component `e % 4`, so the sub-group's two entries are `2s` and
// `2s+1`. Entry `n` takes its low 8 index bits from `qs[n]` and its NINTH from
// bit `n % 8` of `qh[n / 8]`, selecting among the 512 points of `IQ3S_GRID`.
// Both `2s` and `2s+1` land in the same `qh` byte `qh[s / 4]`, at bit
// positions `2 * (s % 4)` and one above it, so ONE byte read serves the pair.
//
// Its sign byte is `signs[s]` DIRECTLY — eight explicit bits, one per element
// of the sub-group, with no `KSIGNS` indirection.
//
// The two index bytes are adjacent and `qs`@2 keeps `2 + 2s` even, so they are
// read as ONE aligned u16. The block is 110 bytes, so a row base and every
// block base inside it are only 2-byte aligned — but this format issues NO
// 4-byte read at all, here or in its scale pass, so `load_int_ua` does not
// apply to it.
struct MmqfIQ3SEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 4;
    struct Regs {
        unsigned int qs;
        unsigned int qh;
        unsigned int sign;
    };
    unsigned long long off_qs;
    unsigned long long off_qh;
    unsigned long long off_sign;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = *reinterpret_cast<const unsigned short*>(row + off_qs);
        r.qh = row[off_qh];
        r.sign = row[off_sign];
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        mag[0] = IQ3S_GRID[(r.qs & 0xFFu) | (((r.qh >> qh_shift) & 1u) << 8)];
        mag[1] = IQ3S_GRID[(r.qs >> 8) | (((r.qh >> (qh_shift + 1)) & 1u) << 8)];
        signs = r.sign;
    }
};

// IQ3_S weight format policy, same contract as `MmqfQ80`.
//
// IQ3_S is a 256-element block of 110 bytes: `f16 d`@0, `qs[64]`@2, `qh[8]`@66,
// `signs[32]`@74, `scales[4]`@106. Grid entry `n` takes its low 8 index bits
// from `qs[n]` and a NINTH from bit `n % 8` of `qh[n / 8]`, selecting among the
// 512 points of `IQ3S_GRID`, four magnitude bytes each. It shares IQ3_XXS's
// 4-component grid width, so an 8-element sign sub-group again spans two
// consecutive entries and `MmqfIQ3SEntry` reads both.
//
// SIGNS. Explicit bits, as in IQ2_S: element `e` takes bit `e % 8` of
// `signs[e / 8]`, so sub-group `s` takes the whole byte `signs[s]`, with no
// `KSIGNS` indirection.
//
// SCALE GRANULARITY. One 4-bit scale per 32-element group, eight of them
// packed two per byte across `scales[4]` — the granularity `mmqf_vec_dot_d`
// already indexes at, so the staged row is Q8_0's and not Q6_K's. Note that
// the packing is IQ2_XS's while the granularity is NOT: `scales` holds 4 bytes
// here, not 8, and the index is the 32-element group, not the pair of entries.
//
// SCALE FORM. `d * (1 + 2 * s)`, which differs in FORM from every other IQ
// format's `d * (0.5 + s) * c`. It is not an affine rewrite of one: at `s = 0`
// it is `d`, not `0.5 * d * c`.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`,
// for the same parity reason as the rest of the family.
//
// ALIGNMENT. The block is 110 bytes, so a row base (`supers * 110`) and every
// block base inside it are only 2-byte aligned. This format issues NO 4-byte
// read: the index pair is one u16 at the even offset `2 + 2 * lane`, and `qh`,
// `signs` and `scales` are read one byte at a time. So `load_int_ua` does not
// apply to it. The `f16 d`@0 is 2-byte aligned, which is `alignof(__half)`,
// and is read directly.
struct MmqfIQ3S {
    // On-disk block: one f16 scale, 64 index bytes, 8 index-high bytes, 32
    // sign bytes, 4 bytes of packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields a sub-group reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int QH_OFF = 66;
    static constexpr int SIGNS_OFF = 74;
    static constexpr int SCALES_OFF = 106;

    // Staged weight row: Q8_0's, byte for byte, same as IQ3_XXS.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 group scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ3_S must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq3_s` in
    // `src/quant/cpu/kernels/dequant_iq3.rs` and matching the per-block decoder
    // `iq3_s_dequant_block` in `iq_dequant.cuh`. Element `e` sits at grid entry
    // `e / 4`, component `e % 4`, sign byte `signs[e / 8]` bit `e % 8`. Lane
    // `l` owns the sign sub-group `s = e / 8 = l`: its two entries are `2l` and
    // `2l + 1`, whose index bytes are `qs[2l]` and `qs[2l + 1]` and whose ninth
    // bits are bits `2 * (l % 4)` and `2 * (l % 4) + 1` of the single byte
    // `qh[l / 4]` — both entries fall in that byte because `(2l) / 8` and
    // `(2l + 1) / 8` are both `l / 4`. Its sign byte is `signs[l]`, and it
    // fills staged words `2l` and `2l + 1`. Under that map a staged word `w`
    // belongs to the 32-element group `w / 8`, which is exactly the scale index
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
        MmqfIQ3SEntry dec;
        dec.off_qs = off_blk + QS_OFF + (unsigned long long)lane * 2;
        dec.off_qh = off_blk + QH_OFF + lane / 4;
        dec.off_sign = off_blk + SIGNS_OFF + lane;
        dec.qh_shift = 2 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_Y, MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's scale byte are at row-invariant offsets, so the first loop is
        // pure loads and the nibble is unpacked in the second, on registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + SCALES_OFF + sb / 2;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        __half d_h[SROWS];
        unsigned int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            sc[u] = row[off_sc];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // Two groups share a byte, low nibble first, and the group scale is
            // `d * (1 + 2 * s)` — this format's own form.
            const float s = (float)((sc[u] >> (4 * (sb % 2))) & 0x0Fu);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (1.0f + 2.0f * s);
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0 and IQ3_XXS: once staging
    // has expanded the two grid points and folded the explicit sign in, the
    // rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
