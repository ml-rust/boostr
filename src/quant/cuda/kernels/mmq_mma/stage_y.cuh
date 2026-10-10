// Activation staging of the feature-major MMQ family: the flat copy of the
// repacked Q8_1 record, the per-16 sums Q2_K's minimum term reads, and the
// register split the pipelined walk uses. Not a compilation unit of its own.
#pragma once

#include "common.cuh"

// Stages 128 k-values of the activation tile as a FLAT COPY. The repacked
// layout indexes records k-group-major, token-minor, so the `MMQ_X` records a
// token tile needs are contiguous and the shared row IS the record: 4 header
// words (`half` scale, int16 quant sum) then 32 quant words. No per-element index math, no token bound, no
// k-block bound — the repack already zeroed the padded token slots and the
// padded k-blocks.
//
// `b0` is always a multiple of 4, so `b0 / 4` is the k-group index.
//
// `HALVES` is how many consecutive k-groups one call stages: one for the
// two-half cadence, two for the full-group cadence. Consecutive k-groups
// are `ntok` records apart in the repacked layout, and `HALF_STRIDE` ints
// apart in shared. Each half is its own flat copy. Both halves' loads
// are issued before either half's stores, so the second half's global
// latency overlaps the first's rather than following it.
//
// The copy is spread over the block's threads, so a narrower feature tile,
// with fewer warps, takes more copy iterations per thread for the same
// activation tile: the tile is per token, not per feature.
template <int MMQ_Y, int MMQ_X, int HALVES, int HALF_STRIDE>
static __device__ __forceinline__ void mmqf_stage_y(
    const int* __restrict__ y_packed, int* __restrict__ s_y, unsigned int ntok,
    unsigned int tok0, unsigned int b0
) {
    const int* src =
        y_packed + ((unsigned long long)(b0 / 4) * ntok + tok0) * MMQF_Y_STRIDE;
    const unsigned long long src_half = (unsigned long long)ntok * MMQF_Y_STRIDE;

    constexpr int THREADS = MMQF_THREADS_OF(MMQ_Y);
    constexpr int TOTAL = MMQ_X * MMQF_Y_STRIDE;
    constexpr int CITER = (TOTAL + THREADS - 1) / THREADS;
    constexpr int CBATCH = CITER < MMQF_STAGE_BATCH(MMQ_X) ? CITER : MMQF_STAGE_BATCH(MMQ_X);
    constexpr int CGROUPS = (CITER + CBATCH - 1) / CBATCH;

#pragma unroll
    for (int g = 0; g < CGROUPS; ++g) {
        int v[HALVES][CBATCH];
        // `g`, `h` and `u` are compile-time here, so both range tests fold
        // away when the copy divides evenly.
#pragma unroll
        for (int h = 0; h < HALVES; ++h) {
#pragma unroll
            for (int u = 0; u < CBATCH; ++u) {
                const int e = g * CBATCH + u;
                if (e >= CITER) {
                    continue;
                }
                const unsigned int l = (unsigned int)(e * THREADS) + threadIdx.x;
                if (TOTAL % THREADS == 0 || l < (unsigned int)TOTAL) {
                    v[h][u] = src[h * src_half + l];
                }
            }
        }
#pragma unroll
        for (int h = 0; h < HALVES; ++h) {
#pragma unroll
            for (int u = 0; u < CBATCH; ++u) {
                const int e = g * CBATCH + u;
                if (e >= CITER) {
                    continue;
                }
                const unsigned int l = (unsigned int)(e * THREADS) + threadIdx.x;
                if (TOTAL % THREADS == 0 || l < (unsigned int)TOTAL) {
                    s_y[h * HALF_STRIDE + l] = v[h][u];
                }
            }
        }
    }
}

// Derives the per-16 activation sums the Q2_K minimum term needs, once per
// staged 128-k activation tile, into the scratch region after that tile.
//
// WHY THIS EXISTS. The activation record's header word carries an exact int16
// sum of each 32-value sub-block, which is the right granularity for Q4_K and
// Q5_K. Q2_K's minimum changes every 16 elements, so a per-32 sum is the wrong
// granularity: the two halves of a 32-value window carry different minima.
//
// Only ONE extra number per (token, sub-block) is needed, not two: this pass
// stores `s_a`, the sum of the FIRST 16 quants, and `vec_dot` recovers the
// second as `s32 - s_a`. Both are integer sums over the same 32 int8 values,
// so that identity is exact in integer arithmetic — the derived half is as
// exact as the stored one.
//
// COST. It runs once per activation tile and is independent of the feature
// rows, so all MMQ_Y features amortize it. That is what separates it from
// deriving the sums with an all-ones `mma` inside `vec_dot`, which costs one
// extra tensor-core issue per (token group, k-step, minitile).
//
// llama.cpp instead gives Q2_K its own activation layout
// (`MMQ_Q8_1_DS_LAYOUT_D2S6` in `ggml-cuda/quantize.cu`), which coarsens the
// activation scale from 32 values to 64, stores the per-16 sums as `half`
// rather than exactly, and still needs an all-ones `mma` for the last quarter
// of each tile. This kernel keeps one format-neutral, exact activation record
// for every format instead.
template <int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_stage_y_sums(
    const int* __restrict__ s_y, int* __restrict__ s_ys
) {
    constexpr int THREADS = MMQF_THREADS_OF(MMQ_Y);
    constexpr int TOTAL = MMQ_X * MMQF_Y_SCRATCH;

#pragma unroll
    for (int e = 0; e < (TOTAL + THREADS - 1) / THREADS; ++e) {
        const unsigned int v = (unsigned int)(e * THREADS) + threadIdx.x;
        if (TOTAL % THREADS == 0 || v < (unsigned int)TOTAL) {
            // Token-major, matching the `MMQF_Y_SCRATCH` ints-per-token
            // framing the host's shared-memory request uses.
            const unsigned int j = v / MMQF_Y_SCRATCH;   // token
            const unsigned int ks = v % MMQF_Y_SCRATCH;  // 32-value sub-block
            const int* q = s_y + j * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8;
            // Four quant words are 16 int8 values, which is one Q2_K scale
            // group. The reduction is integer and therefore exact:
            // |sum| <= 16 * 128 = 2048.
            int s_a = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                s_a = dp4a(0x01010101, q[w], s_a);
            }
            s_ys[v] = s_a;
        }
    }
}

// Runs `mmqf_stage_y_sums` for the formats that ask for it, with the barrier
// that publishes it. Compiles to nothing for every other format.
template <class FMT, int MMQ_Y, int MMQ_X>
static __device__ __forceinline__ void mmqf_stage_y_sums_if(int* __restrict__ s_y) {
    if constexpr (FMT::Y_SCRATCH > 0) {
        static_assert(FMT::Y_SCRATCH == MMQF_Y_SCRATCH, "Unexpected scratch width.");
        mmqf_stage_y_sums<MMQ_Y, MMQ_X>(s_y, s_y + MMQ_X * MMQF_Y_STRIDE);
        __syncthreads();
    }
}

// Whether `FMT` has the `Regs`/`load`/`store` split the pipelined walk needs.
// Formats without it walk the small token tiles on the group cadence.
template <class FMT, class = void>
struct mmqf_pipelined : std::false_type {};
template <class FMT>
struct mmqf_pipelined<FMT, std::void_t<decltype(FMT::PIPELINED)>>
    : std::integral_constant<bool, FMT::PIPELINED> {};

// Activation words one thread copies per staged group, when the copy is one
// batch (`mmqf_stage_y`'s `CGROUPS == 1`). The pipelined walk keeps them in
// registers across the previous group's consumption.
template <int MMQ_Y, int MMQ_X, int HALVES>
struct mmqf_y_regs {
    static constexpr int THREADS = MMQF_THREADS_OF(MMQ_Y);
    static constexpr int TOTAL = MMQ_X * MMQF_Y_STRIDE;
    static constexpr int CITER = (TOTAL + THREADS - 1) / THREADS;
    int v[HALVES][CITER];
};

template <int MMQ_Y, int MMQ_X, int HALVES>
static __device__ __forceinline__ void mmqf_load_y(
    const int* __restrict__ y_packed, unsigned int ntok, unsigned int tok0, unsigned int b0,
    mmqf_y_regs<MMQ_Y, MMQ_X, HALVES>& r
) {
    using R = mmqf_y_regs<MMQ_Y, MMQ_X, HALVES>;
    const int* src =
        y_packed + ((unsigned long long)(b0 / 4) * ntok + tok0) * MMQF_Y_STRIDE;
    const unsigned long long src_half = (unsigned long long)ntok * MMQF_Y_STRIDE;
#pragma unroll
    for (int h = 0; h < HALVES; ++h) {
#pragma unroll
        for (int e = 0; e < R::CITER; ++e) {
            const unsigned int l = (unsigned int)(e * R::THREADS) + threadIdx.x;
            if (R::TOTAL % R::THREADS == 0 || l < (unsigned int)R::TOTAL) {
                r.v[h][e] = src[h * src_half + l];
            }
        }
    }
}

template <int MMQ_Y, int MMQ_X, int HALVES, int HALF_STRIDE>
static __device__ __forceinline__ void mmqf_store_y(
    int* __restrict__ s_y, const mmqf_y_regs<MMQ_Y, MMQ_X, HALVES>& r
) {
    using R = mmqf_y_regs<MMQ_Y, MMQ_X, HALVES>;
#pragma unroll
    for (int h = 0; h < HALVES; ++h) {
#pragma unroll
        for (int e = 0; e < R::CITER; ++e) {
            const unsigned int l = (unsigned int)(e * R::THREADS) + threadIdx.x;
            if (R::TOTAL % R::THREADS == 0 || l < (unsigned int)R::TOTAL) {
                s_y[h * HALF_STRIDE + l] = r.v[h][e];
            }
        }
    }
}
