// Entry-point macros of the feature-major MMQ family. A format's translation
// unit (`<fmt>.cu`) includes its policy from `formats/`, then this header,
// then expands `MMQ_FM_KERNELS` and, for a narrow-tile format,
// `MMQ_FM_KERNELS_NARROW`. Not a compilation unit of its own.
#pragma once

#include "body.cuh"
#include "common.cuh"
#include "split.cuh"

// One entry point per (weight format, feature tile, token tile), for each of
// the four roles: `_x<X>` (one range), `_ms_x<X>` (several ranges in one
// block), `_sk_x<X>` (one range per block) and `_fixup_x<X>`. `NAME` is the
// format's name in the symbol, which the host looks up by string. `TAG` is
// the tiling infix, placed just before `_x<X>`. It is empty for the default
// tile, `_y<Y>` for the narrow and single-warp tiles' halves cadence, and
// `_y<Y>g` for the narrow tile's group cadence (`GROUP`). The host picks the
// variant; the rule it must follow is fixed by the `MMQ_X % granularity`
// assert in `mmqf_body`. The launch bounds follow the feature tile: fewer
// warps per block at the narrow tile, so the compiler may spend more
// registers per thread there without losing residency. The fixup pass reads
// no activation tile, so both cadences' fixup entry points are the same
// body. The host still names the one matching its split-K launch.
#define MMQ_FM_KERNEL_AT(FMT, NAME, Y, GROUP, TAG, X)                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS_OF(Y))                       \
        void quant_mmq_##NAME##_q8_1_mma##TAG##_x##X(                                 \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            unsigned int M, unsigned int K, unsigned int N, unsigned int ntok         \
        ) {                                                                           \
        mmqf_body<FMT, Y, X, GROUP>(y_packed, weight, output, M, K, N, ntok);         \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS_OF(Y))                       \
        void quant_mmq_##NAME##_q8_1_mma_ms##TAG##_x##X(                              \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            unsigned int M, unsigned int K, unsigned int N, unsigned int ntok,        \
            unsigned int splits                                                       \
        ) {                                                                           \
        mmqf_ms_body<FMT, Y, X, GROUP>(                                               \
            y_packed, weight, output, M, K, N, ntok, splits                           \
        );                                                                            \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS_OF(Y))                       \
        void quant_mmq_##NAME##_q8_1_mma_sk##TAG##_x##X(                              \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            float* __restrict__ workspace, unsigned int M, unsigned int K,            \
            unsigned int N, unsigned int ntok                                         \
        ) {                                                                           \
        mmqf_sk_body<FMT, Y, X, GROUP>(                                               \
            y_packed, weight, output, workspace, M, K, N, ntok                        \
        );                                                                            \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS_OF(Y))                       \
        void quant_mmq_##NAME##_q8_1_mma_fixup##TAG##_x##X(                           \
            float* __restrict__ output, const float* __restrict__ workspace,          \
            unsigned int M, unsigned int N, unsigned int splits                       \
        ) {                                                                           \
        mmqf_fixup_body<Y, X>(output, workspace, M, N, splits);                       \
    }

// The default feature tile, compiled for every format at every token tile.
// The empty `TAG` keeps these symbols' names unchanged.
#define MMQ_FM_KERNEL(FMT, NAME, X) MMQ_FM_KERNEL_AT(FMT, NAME, MMQF_Y_DEFAULT, false, , X)

// The narrow feature tile for the CTA-starved small-M, small-N regime. Only
// the formats whose small-N shapes reach that regime compile it, and only at
// the token tiles a small M selects, to bound compile time; the host never
// asks for a (format, token tile) pair outside this list. From
// `MMQF_GROUP_CADENCE_X` up each token tile compiles both cadences; the grid
// size decides between them on the host.
#define MMQ_FM_KERNEL_Y64(FMT, NAME, X) \
    MMQ_FM_KERNEL_AT(FMT, NAME, MMQF_Y_NARROW, false, _y64, X)
#define MMQ_FM_KERNEL_Y64_BOTH(FMT, NAME, X) \
    MMQ_FM_KERNEL_Y64(FMT, NAME, X)          \
    MMQ_FM_KERNEL_AT(FMT, NAME, MMQF_Y_NARROW, true, _y64g, X)

// Single-warp feature tile for the decode regime (M <= 16): four times the
// blocks of the narrow tile at the same per-warp work, so a small-N product
// spreads over every SM.
#define MMQ_FM_KERNEL_Y16(FMT, NAME, X) \
    MMQ_FM_KERNEL_AT(FMT, NAME, MMQF_Y_SMALL, false, _y16, X)

// The default feature tile at every token tile, the set every format
// compiles. Must match `VARIANTS` in
// `src/quant/cuda/quant_matmul/mmq_feat_major/tiling/geometry.rs`.
#define MMQ_FM_KERNELS(FMT, NAME)  \
    MMQ_FM_KERNEL(FMT, NAME, 8)    \
    MMQ_FM_KERNEL(FMT, NAME, 16)   \
    MMQ_FM_KERNEL(FMT, NAME, 24)   \
    MMQ_FM_KERNEL(FMT, NAME, 32)   \
    MMQ_FM_KERNEL(FMT, NAME, 40)   \
    MMQ_FM_KERNEL(FMT, NAME, 48)   \
    MMQ_FM_KERNEL(FMT, NAME, 64)   \
    MMQ_FM_KERNEL(FMT, NAME, 80)   \
    MMQ_FM_KERNEL(FMT, NAME, 96)   \
    MMQ_FM_KERNEL(FMT, NAME, 112)  \
    MMQ_FM_KERNEL(FMT, NAME, 128)

// The narrow and single-warp feature tiles. Q4_K, Q5_K and Q6_K are the
// formats a K-quant mix puts on the small-N projections that starve the
// default tile, and only they expand this, each with its `narrow_tile`
// descriptor flag set. The narrow token tiles stop at 64 because a batch
// wide enough to need more already launches enough tiles at the default
// feature tile. Must match `NARROW_VARIANTS` and `SMALL_VARIANTS` in
// `src/quant/cuda/quant_matmul/mmq_feat_major/tiling/geometry.rs`.
#define MMQ_FM_KERNELS_NARROW(FMT, NAME)    \
    MMQ_FM_KERNEL_Y64(FMT, NAME, 8)         \
    MMQ_FM_KERNEL_Y64(FMT, NAME, 16)        \
    MMQ_FM_KERNEL_Y64(FMT, NAME, 24)        \
    MMQ_FM_KERNEL_Y64(FMT, NAME, 32)        \
    MMQ_FM_KERNEL_Y64_BOTH(FMT, NAME, 40)   \
    MMQ_FM_KERNEL_Y64_BOTH(FMT, NAME, 48)   \
    MMQ_FM_KERNEL_Y64_BOTH(FMT, NAME, 64)   \
    MMQ_FM_KERNEL_Y16(FMT, NAME, 8)         \
    MMQ_FM_KERNEL_Y16(FMT, NAME, 16)
