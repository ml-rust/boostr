// The K walk of the feature-major MMQ family: the three cadences and
// `mmqf_accumulate`, which picks among them. Not a compilation unit of its
// own.
#pragma once

#include "common.cuh"
#include "stage_y.cuh"

// Two-half cadence: the default feature tile's, and the narrow tile's
// non-`GROUP` entry points. The activation tile holds one 128-k half at a
// time. Each 256-k group is two stage/sync/vec_dot rounds sharing one
// staged weight group: four barriers per group, and the second half's
// activation load waits for the first half's consumers.
//
// `[kb0_start, full_stop)` is whole groups; `[full_stop, kb0_stop)` is the
// ragged tail, non-empty only for `RAGGED_K` formats.
template <class FMT, int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_accumulate_halves(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop,
    unsigned int full_stop
) {
    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        FMT::template stage<MMQ_Y, MMQ_X, false>(weight, s_x, N, bpr, feat0, b0);
        mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, b0);
        __syncthreads();
        mmqf_stage_y_sums_if<FMT, MMQ_Y, MMQ_X>(s_y);

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        __syncthreads();

        mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, b0 + 4);
        __syncthreads();
        mmqf_stage_y_sums_if<FMT, MMQ_Y, MMQ_X>(s_y);

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, 4);
        __syncthreads();
    }

    // `bpr` and the range bounds are uniform across the block, so every barrier
    // below is still reached by all threads.
    if constexpr (FMT::RAGGED_K) {
        if (full_stop < kb0_stop) {
            const unsigned int nb = kb0_stop - full_stop;
            FMT::template stage<MMQ_Y, MMQ_X, true>(weight, s_x, N, bpr, feat0, full_stop);
            mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, full_stop);
            __syncthreads();
            mmqf_stage_y_sums_if<FMT, MMQ_Y, MMQ_X>(s_y);

            FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);

            if (nb > 4) {
                __syncthreads();
                mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, full_stop + 4);
                __syncthreads();
                mmqf_stage_y_sums_if<FMT, MMQ_Y, MMQ_X>(s_y);
                FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, nb - 4);
            }
        }
    }
}

// Full-group cadence, the narrow feature tile's `GROUP` entry points at its
// wide token tiles. The activation tile holds both 128-k halves of a
// 256-k group. Each half is a complete record slice plus its own scratch
// region, `MMQF_Y_GROUP_INTS` ints apart. `vec_dot` and `mmqf_stage_y_sums`
// see the same per-half layout as the two-half cadence, through a shifted
// base pointer. One stage, one barrier, both halves' `vec_dot`, one
// barrier: two barriers per group and one exposed round trip. That is
// against the halves cadence's four barriers and three round trips. The
// narrow tile's halved weight tile pays for the doubled activation tile.
// The host's `smem_bytes` doubles the activation term for this tile and
// must stay in step.
//
// The ragged tail stages only the halves it will consume: the k-group after
// the last one is not guaranteed to exist in the repacked buffer.
#define MMQF_Y_GROUP_INTS(X, SCRATCH) ((X) * (MMQF_Y_STRIDE + (SCRATCH)))

template <class FMT, int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_accumulate_group(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop,
    unsigned int full_stop
) {
    static_assert(
        FMT::Y_SCRATCH == 0 || FMT::Y_SCRATCH == MMQF_Y_SCRATCH, "Unexpected scratch width."
    );
    constexpr int HALF = MMQF_Y_GROUP_INTS(MMQ_X, FMT::Y_SCRATCH);
    int* s_y1 = s_y + HALF;

    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        FMT::template stage<MMQ_Y, MMQ_X, false>(weight, s_x, N, bpr, feat0, b0);
        mmqf_stage_y<MMQ_Y, MMQ_X, 2, HALF>(y_packed, s_y, ntok, tok0, b0);
        __syncthreads();
        if constexpr (FMT::Y_SCRATCH > 0) {
            mmqf_stage_y_sums<MMQ_Y, MMQ_X>(s_y, s_y + MMQ_X * MMQF_Y_STRIDE);
            mmqf_stage_y_sums<MMQ_Y, MMQ_X>(s_y1, s_y1 + MMQ_X * MMQF_Y_STRIDE);
            __syncthreads();
        }

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        FMT::template vec_dot<MMQ_X, true>(s_x, s_y1, acc, i0, jb, MMQF_HALF_W, 4);
        __syncthreads();
    }

    // `bpr` and the range bounds are uniform across the block, so every barrier
    // below is still reached by all threads.
    if constexpr (FMT::RAGGED_K) {
        if (full_stop < kb0_stop) {
            const unsigned int nb = kb0_stop - full_stop;
            FMT::template stage<MMQ_Y, MMQ_X, true>(weight, s_x, N, bpr, feat0, full_stop);
            if (nb > 4) {
                mmqf_stage_y<MMQ_Y, MMQ_X, 2, HALF>(y_packed, s_y, ntok, tok0, full_stop);
            } else {
                mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, full_stop);
            }
            __syncthreads();
            if constexpr (FMT::Y_SCRATCH > 0) {
                mmqf_stage_y_sums<MMQ_Y, MMQ_X>(s_y, s_y + MMQ_X * MMQF_Y_STRIDE);
                if (nb > 4) {
                    mmqf_stage_y_sums<MMQ_Y, MMQ_X>(s_y1, s_y1 + MMQ_X * MMQF_Y_STRIDE);
                }
                __syncthreads();
            }

            FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);
            if (nb > 4) {
                FMT::template vec_dot<MMQ_X, false>(s_x, s_y1, acc, i0, jb, MMQF_HALF_W, nb - 4);
            }
        }
    }
}

// Pipelined walk for the token tiles at or below `MMQF_SMALL_X`, on formats
// with the `Regs`/`load`/`store` split. Both 128-k halves of the activation
// group are staged at once, as on the group cadence, and the NEXT group's
// weight and activation words are loaded into registers before this group's
// barrier, so their global latency overlaps this group's `vec_dot` and both
// barriers instead of following them. Same stages, same `vec_dot` calls in
// the same order as the other two cadences, so the same bits.
template <class FMT, int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_accumulate_pipelined(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop,
    unsigned int full_stop
) {
    static_assert(FMT::Y_SCRATCH == 0, "The pipelined walk stages no activation scratch.");
    constexpr int HALF = MMQF_Y_GROUP_INTS(MMQ_X, 0);
    int* s_y1 = s_y + HALF;

    typename FMT::template Regs<MMQ_Y, MMQ_X> rx;
    mmqf_y_regs<MMQ_Y, MMQ_X, 2> ry;
    if (kb0_start < full_stop) {
        mmqf_load_y<MMQ_Y, MMQ_X, 2>(y_packed, ntok, tok0, kb0_start, ry);
        FMT::template load<MMQ_Y, MMQ_X>(weight, N, bpr, feat0, kb0_start, rx);
    }
    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        mmqf_store_y<MMQ_Y, MMQ_X, 2, HALF>(s_y, ry);
        FMT::template store<MMQ_Y, MMQ_X>(s_x, rx);
        if (b0 + MMQF_ITER_B < full_stop) {
            mmqf_load_y<MMQ_Y, MMQ_X, 2>(y_packed, ntok, tok0, b0 + MMQF_ITER_B, ry);
            FMT::template load<MMQ_Y, MMQ_X>(weight, N, bpr, feat0, b0 + MMQF_ITER_B, rx);
        }
        __syncthreads();

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        FMT::template vec_dot<MMQ_X, true>(s_x, s_y1, acc, i0, jb, MMQF_HALF_W, 4);
        __syncthreads();
    }

    // The ragged tail is staged synchronously, as on the group cadence.
    if constexpr (FMT::RAGGED_K) {
        if (full_stop < kb0_stop) {
            const unsigned int nb = kb0_stop - full_stop;
            FMT::template stage<MMQ_Y, MMQ_X, true>(weight, s_x, N, bpr, feat0, full_stop);
            if (nb > 4) {
                mmqf_stage_y<MMQ_Y, MMQ_X, 2, HALF>(y_packed, s_y, ntok, tok0, full_stop);
            } else {
                mmqf_stage_y<MMQ_Y, MMQ_X, 1, 0>(y_packed, s_y, ntok, tok0, full_stop);
            }
            __syncthreads();
            FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);
            if (nb > 4) {
                FMT::template vec_dot<MMQ_X, false>(s_x, s_y1, acc, i0, jb, MMQF_HALF_W, nb - 4);
            }
        }
    }
}

// Runs the k-block range `[kb0_start, kb0_stop)` of one output tile into `acc`,
// which it also zeroes. `kb0_start` is always a multiple of MMQF_ITER_B;
// `kb0_stop` is too, unless it is `bpr`, whose last partial 256-k group takes
// the early-stop path.
//
// The leading barrier makes the function self-contained: a caller may run
// several tiles back to back without draining the previous tile's shared reads
// itself. The walk over a group is picked here: token tiles up to
// `MMQF_SMALL_X` take `mmqf_accumulate_pipelined` on the formats that split
// their staging, else `mmqf_accumulate_group`; wider tiles take
// `mmqf_accumulate_group` when `GROUP` is set, else `mmqf_accumulate_halves`.
// All three issue the same `vec_dot` calls in the same order, so they form
// the same bits.
template <class FMT, int MMQ_Y, int MMQ_X, bool GROUP>
static __device__ __forceinline__ void mmqf_accumulate(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    // Holds for every format: any multiple of 8 puts consecutive weight rows in
    // the same 128-bit segment and serializes the strided fragment gathers.
    static_assert(FMT::X_STRIDE % 8 == 4, "Wrong weight row padding.");
    static_assert(
        !GROUP || (MMQ_Y == MMQF_Y_NARROW && MMQ_X >= MMQF_GROUP_CADENCE_X),
        "The group cadence is compiled only at the narrow tile's wide token tiles."
    );

#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[jj][n][l] = 0.0f;
            }
        }
    }

    __syncthreads();

    // The legacy formats (Q8_0, Q4_0) guarantee K only as a multiple of one
    // 32-element block, so the final group of a tile can hold fewer than
    // MMQF_ITER_B blocks. A format that
    // guarantees whole 256-k groups sets `RAGGED_K` false, which folds this to
    // `kb0_stop` and drops the tail instantiations below.
    const unsigned int full_stop =
        (FMT::RAGGED_K && kb0_stop == bpr) ? (bpr & ~(MMQF_ITER_B - 1u)) : kb0_stop;

    if constexpr (MMQ_X <= MMQF_SMALL_X && mmqf_pipelined<FMT>::value) {
        mmqf_accumulate_pipelined<FMT, MMQ_Y, MMQ_X>(
            y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb, kb0_start,
            kb0_stop, full_stop
        );
    } else if constexpr (GROUP || MMQ_X <= MMQF_SMALL_X) {
        mmqf_accumulate_group<FMT, MMQ_Y, MMQ_X>(
            y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb, kb0_start,
            kb0_stop, full_stop
        );
    } else {
        mmqf_accumulate_halves<FMT, MMQ_Y, MMQ_X>(
            y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb, kb0_start,
            kb0_stop, full_stop
        );
    }
}
