// Split-K range bodies for the feature-major MMQ family, included by
// `quant_mmq_mma.cu` after `mmqf_accumulate`, `mmqf_write_dst` and
// `mmqf_write_fixup` are defined; it is not a compilation unit of its own.
//
// Three entry-point bodies live here: `mmqf_ms_body` (one block per output
// tile, the K ranges back to back), `mmqf_sk_body` (one block per (tile,
// range), partials to a workspace) and `mmqf_fixup_body` (adds the workspace
// partials onto the range-0 store). All three cut K with `mmqf_split_range`
// and sum the range partials in range order, so for a fixed range count
// every output element receives the same float sequence from any of them,
// at any M and any tile position.
#pragma once

// `mmqf_split_range`, the one K-range rule; the single-token kernel in
// `quant_mmq_gemv1.cu` reads the same header.
#include "mmq/split_range.cuh"

// Adds a partial tile into the output. The same lane owns the same element
// on every split, so the read-modify-write needs no atomics; the sum for
// each element is formed in split order, which is what the fixup pass does
// with the split-K workspace.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_add_dst(
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
                if (t < M && f < N) {
                    float* p = output + (unsigned long long)t * N + f;
                    *p = *p + acc[jj][n][l];
                }
            }
        }
    }
}

// Widest token tile whose multi-range body keeps the running total in a
// second register accumulator. Above it the total lives in the output and
// each range past the first is a read-modify-write: those tiles already sit
// near the register ceiling, and a second copy of `acc` would spill.
#define MMQF_REG_TOTAL_MAX_X 64

// Multi-range tile-parallel path: one block per output tile, whole K walked
// as `splits` consecutive ranges (`splits >= 2`). Range 0's partial becomes
// the total; each later range is added to it. Per element that is the sum
// `((p0 + p1) + p2) + ...` in range order, the same float sequence the
// split-K launch and its fixup form, so the two schedules give the same bits
// and the host may pick either by grid size alone. Where the total is a
// register array the output is written once; see `MMQF_REG_TOTAL_MAX_X` for
// where it is the output itself.
template <class FMT, int MMQ_Y, int MMQ_X, bool GROUP>
static __device__ __forceinline__ void mmqf_ms_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N, unsigned int ntok, unsigned int splits
) {
    constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
    static_assert(WARPS % NTX == 0, "Warps do not split into NTX token columns.");

    int* s_x = mmqf_smem;
    int* s_y = mmqf_smem + MMQ_Y * FMT::X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int bpr = K / 32;
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;
    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQ_Y;

    float acc[NJ][NTX][4];
    if constexpr (MMQ_X <= MMQF_REG_TOTAL_MAX_X) {
        // Zeroed so the range-0 select below reads a defined value; the
        // select, not an add, carries range 0 in, so a signed zero survives.
        float tot[NJ][NTX][4];
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
            for (int n = 0; n < NTX; ++n) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    tot[jj][n][l] = 0.0f;
                }
            }
        }
        for (unsigned int s = 0; s < splits; ++s) {
            unsigned int kb0, kb1;
            mmqf_split_range(bpr, splits, s, kb0, kb1);
            mmqf_accumulate<FMT, MMQ_Y, MMQ_X, GROUP>(y_packed, weight, s_x, s_y, acc, ntok, N,
                                                      bpr, tok0, feat0, i0, jb, kb0, kb1);
#pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
                for (int n = 0; n < NTX; ++n) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        tot[jj][n][l] = (s == 0) ? acc[jj][n][l] : tot[jj][n][l] + acc[jj][n][l];
                    }
                }
            }
        }
        mmqf_write_dst<MMQ_X>(output, tot, M, N, tok0, feat0, i0, jb);
    } else {
        for (unsigned int s = 0; s < splits; ++s) {
            unsigned int kb0, kb1;
            mmqf_split_range(bpr, splits, s, kb0, kb1);
            mmqf_accumulate<FMT, MMQ_Y, MMQ_X, GROUP>(y_packed, weight, s_x, s_y, acc, ntok, N,
                                                      bpr, tok0, feat0, i0, jb, kb0, kb1);
            if (s == 0) {
                mmqf_write_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);
            } else {
                mmqf_add_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);
            }
        }
    }
}

// Split-K path: grid (token tiles, feature tiles, splits). Block `(x, y, s)`
// runs split `s` of tile `(x, y)` over the range `mmqf_split_range` gives.
// Split 0 stores its partial to the output; every later split writes a dense
// partial tile to `workspace[tile][s - 1]`, and the fixup pass adds them in
// split order. It exists because the tile-parallel grid starves the device
// when the tile count is small relative to it; the split count is the host's
// and never depends on the tile count, so the sum each element receives is
// the same at every M and at every tile position.
template <class FMT, int MMQ_Y, int MMQ_X, bool GROUP>
static __device__ __forceinline__ void mmqf_sk_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ workspace,
    unsigned int M, unsigned int K, unsigned int N, unsigned int ntok
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    int* s_x = mmqf_smem;
    int* s_y = mmqf_smem + MMQ_Y * FMT::X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Activation blocks, as in `mmqf_body` — never the weight block count.
    const unsigned int bpr = K / 32;
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;

    const unsigned int splits = gridDim.z;
    const unsigned int s = blockIdx.z;
    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQ_Y;

    unsigned int kb0, kb1;
    mmqf_split_range(bpr, splits, s, kb0, kb1);

    float acc[NJ][NTX][4];
    mmqf_accumulate<FMT, MMQ_Y, MMQ_X, GROUP>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr,
                                              tok0, feat0, i0, jb, kb0, kb1);
    if (s == 0) {
        mmqf_write_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);
        return;
    }
    const unsigned long long tile = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long slot = tile * (splits - 1) + (s - 1);
    mmqf_write_fixup<MMQ_Y, MMQ_X>(workspace + slot * (MMQ_X * MMQ_Y), acc, i0, jb);
}

// Folds the split-K partials into the output. Launched with grid (token
// tiles, feature tiles), so the tile index is the block's own; `splits` is
// the split-K launch's `gridDim.z`. Each element takes its partials in split
// order onto the value split 0 stored, the same sequence the fused
// tile-parallel body forms in place.
//
// The workspace is read with a flat, coalesced map rather than the MMA
// accumulator map the writer used: the slot is a dense tile, so the two maps
// only have to agree on the layout, not on which lane holds which element.
template <int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_fixup_body(
    float* __restrict__ output, const float* __restrict__ workspace, unsigned int M,
    unsigned int N, unsigned int splits
) {
    constexpr int THREADS = MMQF_THREADS_OF(MMQ_Y);
    constexpr int NEL = MMQ_X * MMQ_Y / THREADS;
    static_assert(NEL * THREADS == MMQ_X * MMQ_Y, "Fixup tile is ragged.");

    if (splits < 2) {
        return;
    }

    const unsigned long long tile = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const float* ws = workspace + tile * (splits - 1) * (MMQ_X * MMQ_Y);
    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQ_Y;

#pragma unroll
    for (int e = 0; e < NEL; ++e) {
        const unsigned int idx = e * THREADS + threadIdx.x;
        const unsigned int t = tok0 + idx / MMQ_Y;
        const unsigned int f = feat0 + idx % MMQ_Y;
        if (t < M && f < N) {
            float* p = output + (unsigned long long)t * N + f;
            float o = *p;
            for (unsigned int s = 1; s < splits; ++s) {
                o += ws[(unsigned long long)(s - 1) * (MMQ_X * MMQ_Y) + idx];
            }
            *p = o;
        }
    }
}
