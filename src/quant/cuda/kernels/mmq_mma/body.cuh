// One-range entry-point body of the feature-major MMQ family, `mmqf_body`,
// and the output writes it shares with the bodies in `split.cuh`. Not a
// compilation unit of its own.
#pragma once

#include "accumulate.cuh"
#include "common.cuh"

// The accumulator is (feature, token)-indexed; the output stays
// (token, feature)-indexed.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_write_dst(
    float* __restrict__ output, const float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4],
    unsigned int M, unsigned int N, unsigned int tok0, unsigned int feat0, unsigned int i0,
    unsigned int jb
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const unsigned int f = feat0 + i0 + n * 16 + mma_d_i(l);
                const unsigned int t = tok0 + jj * (NTX * 8) + jb + mma_d_j(l);
                if (t < M && f < N) output[(unsigned long long)t * N + f] = acc[jj][n][l];
            }
        }
    }
}

// Writes a partial tile to this block's workspace slot. The slot is a dense
// MMQ_X x MMQ_Y tile in (token, feature) order, so no bounds check applies;
// the fixup pass clips when it folds the slot into the output.
template <int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_write_fixup(
    float* __restrict__ ws, const float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4],
    unsigned int i0, unsigned int jb
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const unsigned int i = i0 + n * 16 + mma_d_i(l);
                const unsigned int j = jj * (NTX * 8) + jb + mma_d_j(l);
                ws[j * MMQ_Y + i] = acc[jj][n][l];
            }
        }
    }
}

// Tile-parallel path: one block per output tile, whole K in one range. This
// is the `splits == 1` schedule; it is its own entry point, not a one-trip
// case of `mmqf_ms_body`, because the multi-range loop's presence alone
// changes the register allocation of the straight-line walk.
template <class FMT, int MMQ_Y, int MMQ_X, bool GROUP>
static __device__ __forceinline__ void mmqf_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N, unsigned int ntok
) {
    constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
    static_assert(MMQ_X % MMQF_GRAN(MMQ_X) == 0, "MMQ_X violates the granularity rule.");
    static_assert(WARPS % NTX == 0, "Warps do not split into NTX token columns.");
    static_assert((WARPS / NTX) * NTX * 16 == MMQ_Y, "Warps do not cover the features.");

    int* s_x = mmqf_smem;
    int* s_y = mmqf_smem + MMQ_Y * FMT::X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Q8_1 activation blocks per row. Always 32 elements, whatever the weight
    // format packs, because every MMQ path quantizes activations to Q8_1. The
    // k-step count, the split ranges and `MMQF_ITER_B` are all counted in
    // these, so a 256-element weight block converts inside `FMT::stage`.
    const unsigned int bpr = K / 32;

    // The warps form a (WARPS/NTX) x NTX grid over the output tile: a warp
    // owns NTX consecutive 16-feature minitiles and every NTX-th token group.
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;

    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQ_Y;

    float acc[NJ][NTX][4];
    mmqf_accumulate<FMT, MMQ_Y, MMQ_X, GROUP>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr,
                                              tok0, feat0, i0, jb, 0, bpr);
    mmqf_write_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);
}
