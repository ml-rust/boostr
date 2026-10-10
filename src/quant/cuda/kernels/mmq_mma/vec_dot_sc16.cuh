// `mmqf_vec_dot_sc16`, the consumer every weight format whose scale changes
// every 16 elements forwards to. Not a compilation unit of its own.
#pragma once

#include "common.cuh"

// Shared `vec_dot` for the formats whose scale changes every 16 elements.
// Q6_K, Q3_K, Q2_K, IQ2_XS and IQ2_S differ ONLY in how `stage` unpacks quants
// and in whether the format carries a minimum term; once staged, their weight
// rows are the same shape — quants in the int8 lanes plus one scale record per
// 16-element group — and the MMA structure is identical, so all five formats'
// `vec_dot` forward here. IQ2_XS and IQ2_S reach that granularity from the
// other side: their 4-bit scale is packed two to a `scales` byte, one per two
// grid entries, and a grid entry is eight elements. The offsets are template
// parameters rather than hard constants so a format that shifts its row layout
// still reuses this body.
//
// Consumes one staged 128-k half, four 32-k steps, each split into two
// 16-k `mma_m16n8k16_s8` calls because the scale changes at 16. Words
// `kw .. kw+3` are the low half and take scale `kw / 4`; words
// `kw+4 .. kw+7` are the high half and take scale `kw / 4 + 1`.
//
// `MIN` selects the scale record and the minimum term, and nothing else:
//
//   MIN == false (Q6_K, Q3_K,  `X_DF` holds 16 f32 `d * scale_j`. The int16
//                 IQ2_XS,      block sum in the HIGH half of the activation
//                 IQ2_S)       header word is never read; only the `half`
//                              activation scale in the low half is.
//
//   MIN == true  (Q2_K)        `X_DF` holds 16 `float2`
//                              `(d * sc_j, -dmin * m_j)`. Each 16-k half takes
//                              one extra multiply-add against the activation's
//                              sum over ITS OWN 16 elements — see the split
//                              derived by `mmqf_stage_y_sums`.
//
// The k-step loop is outermost so only one A fragment pair and its two
// scale pairs are live at a time. ggml hoists a `scA[ntx][ne/2][8]`
// register array across the whole tile instead; that costs more registers
// than this kernel's accumulator leaves free.
template <int MMQ_X, bool FULL, bool MIN, int X_QS, int X_DF, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_sc16(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    const float* s_xdf = (const float*)s_x;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        // Weight is operand A at the 16x16 shape: 16 feature rows x 16
        // k-values, two ints per lane. `mma_a16_*` is the k=16 map and is
        // NOT interchangeable with `mma_a_*`.
        int A_lo[NTX][2];
        int A_hi[NTX][2];
        float sc_lo[NTX][2];
        float sc_hi[NTX][2];
        float mn_lo[NTX][2];
        float mn_hi[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const unsigned int r = (ir + mma_a16_i(l)) * X_STRIDE + X_QS + kw;
                A_lo[n][l] = s_x[r + mma_a16_j(l)];
                A_hi[n][l] = s_x[r + 4 + mma_a16_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, so two rows cover
            // the whole accumulator fragment. D keeps the 16x8 shape at
            // both `mma` widths, so this map is the same one the k=32
            // formats use.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned int row = (ir + mma_d_i(2 * h)) * X_STRIDE;
                if constexpr (MIN) {
                    // One `float2` per 16-element group: `(d * sc, -dmin * m)`.
                    // `X_DF` and `X_STRIDE` are both even for this layout, so
                    // each pair is one `ld.shared.v2.f32`.
                    const float2* s_xdm = (const float2*)s_x;
                    const float2 lo = s_xdm[(row + X_DF) / 2 + kw / 4];
                    const float2 hi = s_xdm[(row + X_DF) / 2 + kw / 4 + 1];
                    sc_lo[n][h] = lo.x;
                    mn_lo[n][h] = lo.y;
                    sc_hi[n][h] = hi.x;
                    mn_hi[n][h] = hi.y;
                } else {
                    const unsigned int r = row + X_DF + kw / 4;
                    sc_lo[n][h] = s_xdf[r];
                    sc_hi[n][h] = s_xdf[r + 1];
                }
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            // Activation is operand B at the 8x16 shape: 8 tokens x 16
            // k-values, one int per lane, shared by all NTX minitiles.
            int B_lo[1];
            int B_hi[1];
            {
                const unsigned int r =
                    (jt + mma_b16_i(0)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8;
                B_lo[0] = s_y[r + mma_b16_j(0)];
                B_hi[0] = s_y[r + 4 + mma_b16_j(0)];
            }
            // One header word per token: `half` scale in bits 0..15, int16
            // quant sum in bits 16..31. Only the scale is read unless `MIN`.
            // The producer rounds `d` through `half`, so reading the low half
            // back as float is an exact round-trip. `mma_d_j(l)` takes one
            // value per `l % 2`, so two words cover the fragment. None of
            // these reads depends on the feature minitile, so they all stay
            // outside the `n` loop below.
            float da[2];
            float dsum_lo[2];
            float dsum_hi[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int ds =
                    s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
                da[l] = __half2float(__ushort_as_half((unsigned short)(ds & 0xFFFF)));
                if constexpr (MIN) {
                    // The header's high half is the EXACT int16 sum of this
                    // token's 32 quants for this step; `s_ys` holds the exact
                    // int sum of the FIRST 16 of the same 32. Both are integer
                    // sums over the same values, so `s32 - s_a` is the sum of
                    // the last 16 exactly — an integer identity, not an
                    // approximation. The activation scale is per token and is
                    // folded in here, outside the minitile loop below.
                    //
                    // The sums live in their own region straight after the
                    // activation tile, at a fixed offset from it, so the format
                    // contract does not have to carry a fourth pointer.
                    const int* s_ys = s_y + MMQ_X * MMQF_Y_STRIDE;
                    const int s32 = ds >> 16;
                    const int s_a = s_ys[(jt + mma_d_j(l)) * MMQF_Y_SCRATCH + ks];
                    dsum_lo[l] = da[l] * (float)s_a;
                    dsum_hi[l] = da[l] * (float)(s32 - s_a);
                }
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D_lo[4] = {0, 0, 0, 0};
                int D_hi[4] = {0, 0, 0, 0};
                mma_m16n8k16_s8(D_lo, A_lo[n], B_lo);
                mma_m16n8k16_s8(D_hi, A_hi[n], B_hi);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // Each 16-k half is exact in int32, so its own
                    // `d * scale` applies once; the activation scale is
                    // common to both halves and factors out.
                    acc[jj][n][l] += (sc_lo[n][l / 2] * (float)D_lo[l]
                                      + sc_hi[n][l / 2] * (float)D_hi[l])
                                     * da[l % 2];
                    if constexpr (MIN) {
                        acc[jj][n][l] += mn_lo[n][l / 2] * dsum_lo[l % 2]
                                         + mn_hi[n][l / 2] * dsum_hi[l % 2];
                    }
                }
            }
        }
    }
}
