// `mmqf_vec_dot_d`, the consumer every one-scale weight format forwards to.
// Not a compilation unit of its own.
#pragma once

#include "common.cuh"

// Shared `vec_dot` for the formats whose weight term is a scale alone. Q8_0,
// Q4_0, Q5_0, IQ4_NL, IQ4_XS, IQ2_XXS, PQ2_0, Q2_0, Q1_0 and PTQ1_0 differ
// ONLY in how `stage` unpacks quants; once staged, their weight row is the
// same shape — signed quants in the int8 lanes plus eight f32 scales, one
// per 32 elements — and their arithmetic is identical, so all ten formats'
// `vec_dot` forward here. IQ4_XS and IQ2_XXS are the 256-element blocks among them: their scale
// granularity is still 32 elements, so a staged 256-k group takes the same
// eight scales. The four lowbit formats hold one scale per 64 or 128
// elements, and `stage` writes it into each 32-element slot it covers.
// The three offsets are template parameters rather than hard constants so a
// format that shifts its row layout still reuses this body.
//
// Consumes one staged 128-k half. `k00` is the word offset of that half inside
// a weight row; the activation tile always holds the half at word 0. `i0` is
// the warp's feature base, `jb` its token base within a group.
//
// `FULL` is the whole-half case (four 32-k steps); the tail instantiation stops
// early rather than running steps whose scales are zero.
//
// The k-step loop is outermost so only one A fragment per minitile and its two
// scales are live at a time.
//
// One scalar term per fragment element: the weight scale, the activation
// scale, and the exact int32 dot. There is no minimum term, which is what
// separates this body from `mmqf_vec_dot_dm` (`vec_dot_dm.cuh`).
template <int MMQ_X, bool FULL, int X_QS, int X_DS, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_d(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    const float* s_xd = (const float*)s_x;
    const half2* s_yds = (const half2*)s_y;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        // Weight is operand A: 16 feature rows x 32 k-values per minitile.
        int A[NTX][4];
        float dw[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                A[n][l] = s_x[(ir + mma_a_i(l)) * X_STRIDE + X_QS + kw + mma_a_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, so two weight scales
            // cover the whole accumulator fragment.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                dw[n][h] = s_xd[(ir + mma_d_i(2 * h)) * X_STRIDE + X_DS + kw / 8];
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            // Activation is operand B: 8 tokens x 32 k-values, shared by all
            // NTX minitiles.
            int B[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8 + mma_b_j(l)];
            }
            // `mma_d_j(l)` takes one value per `l % 2`. `MMQF_Y_STRIDE`/`MMQF_Y_DS`
            // are int counts and a header word is one int, so the index is
            // unchanged. The scale is the LOW half of the header word; the high
            // half is the int16 quant sum, which these formats have no use for
            // and never touch. The producer already rounds `d` through `half`, so
            // reading the low half back as float is an exact round-trip,
            // numerically identical to the old f32-scale layout.
            float da[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                da[l] = __low2float(s_yds[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks]);
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(D, A[n], B);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // `da` is the activation scale, `dw` the weight scale. The
                    // int32 result is exact, so both apply once per 32-element
                    // block.
                    acc[jj][n][l] += (float)D[l] * da[l % 2] * dw[n][l / 2];
                }
            }
        }
    }
}
