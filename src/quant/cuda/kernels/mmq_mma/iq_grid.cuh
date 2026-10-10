// Staging shared by the grid-indexed IQ weight formats: the quant pass
// `mmqf_stage_iq_grid` and the packed-scale pass
// `mmqf_stage_iq2_packed_scales`. Each format's per-entry decoder lives with
// its policy in `formats/`. Not a compilation unit of its own.
#pragma once

#include "common.cuh"
#include "../iq_dequant.cuh"

// ── Shared staging for the grid-indexed IQ formats ─────────────────────────
//
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S have the same interior. A
// 256-element block is 32 SUB-GROUPS of 8 elements; every sub-group names 8
// magnitude bytes of a codebook grid plus, for five of the six, one byte of
// eight sign bits, one bit per magnitude. Only the way a sub-group reaches its
// magnitudes and its sign byte differs, so that is the only thing `DEC`
// carries.
//
// SIGN SOURCE. `DEC::SIGNED_GRID` says whether the grid's components are
// already signed. The IQ2 and IQ3 grids store magnitudes and a separate sign
// byte selects the sign, so those decoders set it false and the fold below
// runs. The IQ1 grid stores SIGNED bytes and has no sign table at all, so
// IQ1_S sets it true, leaves `signs` alone, and the staged word is the decoded
// word unchanged.
//
// GRID WIDTH. The IQ2 grids are 8-component: one point supplies the whole
// sub-group, as a `u64` split into two magnitude words. The IQ3 grids are
// 4-component: TWO consecutive points supply the sub-group, one `u32`
// magnitude word each. `DEC::GRID_COMPONENTS` names which, and it is checked
// here rather than assumed. Either way a sub-group is exactly TWO staged
// words, so the map below is one skeleton at both widths and `DEC::decode`
// hands back the sub-group's two magnitude words already in element order.
//
// A lane owns one sub-group: `sub = lane` covers all 32 with a single global
// load group and no inner loop. Sub-group `s` is elements `8s .. 8s+7`, which
// is exactly the staged words `2s` and `2s+1`, in element order, which is the
// order an int8 lane quadruple is read back in. So a staged word `w` holds
// elements `4w .. 4w+3` for all five formats, and both of the family's scale
// maps fall out of that — `w / 8` is the 32-element sub-block
// `mmqf_vec_dot_d` indexes at, `w / 4` the 16-element group
// `mmqf_vec_dot_sc16` indexes at. Sign bits `0..3` belong to the first staged
// word and `4..7` to the second, which is `iq_sign_mask4`'s nibble argument.
//
// `DEC` supplies the per-format delta, split in two so the two-loop staging
// shape survives: `DEC::load(row)` issues the sub-group's global loads and
// NOTHING else, `DEC::decode` does the grid and sign lookups afterwards. A
// table lookup placed between two loads would separate a load from the next
// load's issue and serialize the group.
template <int MMQ_Y, int MMQ_X, int X_QS, int X_STRIDE, class DEC>
static __device__ __forceinline__ void mmqf_stage_iq_grid(
    const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int i_max,
    unsigned long long rstride, unsigned int feat0, const DEC& dec
) {
    static_assert(
        DEC::GRID_COMPONENTS == 8 || DEC::GRID_COMPONENTS == 4,
        "A sign sub-group is 8 elements: one 8-component point or two 4-component points."
    );
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Sub-group `lane` occupies staged words `2 * lane` and `2 * lane + 1`.
    const unsigned int w_lo = (threadIdx.x % WARP_SIZE) * 2;

    constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
    constexpr int ROWS = MMQ_Y / WARPS;
    constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
    static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
    for (int g = 0; g < ROWS / BATCH; ++g) {
        typename DEC::Regs v[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
            v[u] = dec.load(weight + (feat0 + min(i, i_max)) * rstride);
        }
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
            unsigned int mag[2];
            // A signed-grid decoder never writes this; initialized so the
            // unused value is never read from an uninitialized register.
            unsigned int signs = 0;
            dec.decode(v[u], mag, signs);
            // `__vsub4(mag ^ mask, mask)` is a per-byte negate-if: a byte whose
            // mask is 0xFF becomes `(255 - mag) - 255 = -mag`, and a byte whose
            // mask is 0 is untouched. No borrow crosses a lane. The largest
            // magnitude in any of the unsigned grids is 62, so a negated
            // component still fits an int8 lane; the signed IQ1 grid's
            // components are already in `{-1, 0, 1}`.
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                if constexpr (DEC::SIGNED_GRID) {
                    s_x[i * X_STRIDE + X_QS + w_lo + t] = (int)mag[t];
                } else {
                    const unsigned int m = iq_sign_mask4((unsigned char)signs, t);
                    s_x[i * X_STRIDE + X_QS + w_lo + t] = (int)__vsub4(mag[t] ^ m, m);
                }
            }
        }
    }
}

// Shared scale pass for IQ2_XS and IQ2_S, which pack their scales identically:
// a `scales[8]` field of 4-bit values, one per 16-element group, two groups to
// a byte, each giving `d * (0.5 + s) * 0.25`. `SCALES` is the byte offset of
// that field within the block — 66 for IQ2_XS, 74 for IQ2_S — and is the only
// thing that differs.
//
// Sixteen scales per row, so a warp covers two rows. Group `j` (0..15) covers
// entries `2j` and `2j + 1`, which is the `k = entry / 2` of `packed_scale` in
// `src/quant/cpu/kernels/dequant_iq2.rs`: its byte is `scales[j / 2]` and its
// nibble `j % 2`. Staged word `w` sits in group `w / 4`, which is what
// `mmqf_vec_dot_sc16` indexes at.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.25` to it
// perturbs every group's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// `d` and the scale byte are at row-invariant offsets, so the first loop is
// pure loads and the nibble is unpacked in the second, on registers.
template <int MMQ_Y, int MMQ_X, int X_DF, int X_STRIDE, int SCALES>
static __device__ __forceinline__ void mmqf_stage_iq2_packed_scales(
    const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int i_max,
    unsigned long long rstride, unsigned int feat0, unsigned long long off_blk
) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    float* s_xdf = (float*)s_x;
    const unsigned int j = lane % 16;  // 16-element group within the block
    const unsigned int rsub = lane / 16;

    constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
    constexpr int SROWS = MMQ_Y / (WARPS * 2);
    __half d_h[SROWS];
    unsigned int sc[SROWS];
#pragma unroll
    for (int u = 0; u < SROWS; ++u) {
        const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
        const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
        // Both block sizes are even, so a block base is 2-byte aligned, which
        // is `alignof(__half)`, and `d`@0 is read directly. The scale is a
        // single byte and carries no alignment requirement of its own.
        d_h[u] = *reinterpret_cast<const __half*>(blk);
        sc[u] = blk[SCALES + j / 2];
    }
#pragma unroll
    for (int u = 0; u < SROWS; ++u) {
        const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
        const float s = (float)((sc[u] >> (4 * (j % 2))) & 0x0Fu);
        s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h[u]) * (0.5f + s) * 0.25f;
    }
}
