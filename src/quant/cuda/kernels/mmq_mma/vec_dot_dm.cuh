// `mmqf_vec_dot_dm`, the consumer every scale-and-minimum weight format
// forwards to. Not a compilation unit of its own.
#pragma once

#include "common.cuh"

// Shared `vec_dot` for the formats carrying a scale and a minimum per
// 32-element block: Q4_K, Q5_K, Q4_1 and Q5_1. They differ ONLY in how `stage`
// unpacks quants and in the SIGN of the minimum it folds in; once staged,
// their weight row is the same shape — unsigned quants in the int8 lanes plus
// eight `(scale_j, min_j)` f32 pairs — and their arithmetic is identical, so
// all four formats' `vec_dot` forward here. The K-quants store
// `(d * sc_j, -dmin * m_j)` because their value is `d * sc * q - dmin * m`;
// Q4_1 and Q5_1 store `(d, +m)` because theirs is `d * q + m`. This body
// always ADDS `pair.y * (activation scale * block sum)`. The three offsets are template parameters rather than hard
// constants so a format that shifts its row layout still reuses this body.
//
// Consumes one staged 128-k half; the loop shape matches `MmqfQ80::vec_dot`
// exactly, four 32-k steps of one `mma_m16n8k32_s8` per (token group,
// minitile). Two scalar terms per fragment element instead of one:
//
//   acc += pair.x * d_act * int32 dot
//   acc += pair.y * d_act * int32 block sum
//
// The second is the minimum correction. Its block sum is the sum of that
// token's 32 activation quants for this 32-k step, which the producer
// already computed and stored EXACTLY as the int16 in bits 16..31 of the
// activation header word. It is read here, not recomputed: an all-ones
// `mma` also yields it exactly, but costs one extra tensor-core issue per
// (token group, k-step), and a `half` `d * sum(q)` field is too coarse for
// the GEMM/GEMV parity bound. |sum| <= 32 * 128 = 4096 fits int16, so the
// stored value equals the MMA's to the bit.
//
// The sum is per token, so it indexes exactly like the scale `da[l % 2]`:
// both come from the same header word at row `mma_d_j(l)`, and neither
// depends on the feature minitile.
template <int MMQ_X, bool FULL, int X_QS, int X_DM, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_dm(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    // The weight pair is f32. The activation header stays a packed 32-bit
    // word, which is the producer's record format for every weight format,
    // and is read here through `s_y` directly.
    const float2* s_xdm = (const float2*)s_x;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        int A[NTX][4];
        float2 dmw[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                A[n][l] = s_x[(ir + mma_a_i(l)) * X_STRIDE + X_QS + kw + mma_a_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, and `kw / 8` is the
            // sub-block index the staged pair belongs to.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                dmw[n][h] = s_xdm[((ir + mma_d_i(2 * h)) * X_STRIDE + X_DM) / 2 + kw / 8];
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            int B[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8 + mma_b_j(l)];
            }
            // One header word per token, holding both scalars this format
            // needs: `half` scale in bits 0..15, int16 quant sum in bits
            // 16..31. `mma_d_j(l)` takes one value per `l % 2`, so two
            // words cover the whole accumulator fragment. The producer
            // rounds `d` through `half`, so reading the low half back as
            // float is an exact round-trip; the high half is sign-extended
            // by an arithmetic shift of the signed word, then widened,
            // which is exact for every attainable value.
            float da[2];
            float sum_i[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int ds = s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
                da[l] = __half2float(__ushort_as_half((unsigned short)(ds & 0xFFFF)));
                sum_i[l] = (float)(ds >> 16);
            }

            // The activation scale times the block sum is per token, so it
            // is formed once here rather than once per minitile inside the
            // `n` loop below.
            float dsum[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                dsum[l] = da[l] * sum_i[l];
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(D, A[n], B);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[jj][n][l] += dmw[n][l / 2].x * da[l % 2] * (float)D[l];
                    acc[jj][n][l] += dmw[n][l / 2].y * dsum[l % 2];
                }
            }
        }
    }
}
