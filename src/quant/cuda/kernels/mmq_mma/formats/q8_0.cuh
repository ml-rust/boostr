// Q8_0 weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_d.cuh"

// Weight format policy. The format-generic machinery (`accumulate.cuh`,
// `body.cuh`, `split.cuh`) reaches the weight tile only through one of these:
// the on-disk block geometry, the staged weight-row layout, and the two
// functions that touch weight data. Q8_0, Q4_0,
// Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS,
// IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, PQ2_0, Q2_0, Q1_0 and PTQ1_0 are the
// instantiations.
struct MmqfQ80 {
    // On-disk block: one f16 scale then 32 int8 quants.
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values), then 8 f32 block
    // scales, then padding. The padding makes the stride an odd multiple of 4
    // ints, so consecutive rows start in different 128-bit segments and the
    // strided fragment gathers in `vec_dot` hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");

    // Stages 256 k-values (8 blocks) of the weight tile. One warp owns one
    // feature row per step and steps by the warp count; within the row a lane maps
    // to (block, 4-k word) and issues two loads, for the block at `b0` and the one
    // 128 k-values further along. Independent of MMQ_X.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 8;   // Q8_0 block within a 128-k half
        const unsigned int kqsx = lane % 8;  // 4-k word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q8_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Both block offsets are loop-invariant, so the addressing collapses to one
        // add per row.
        const unsigned int blk_lo = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned int blk_hi = CLAMP_K ? min(b0 + 4 + kbx, bpr - 1) : b0 + 4 + kbx;
        // The quants start at byte 2 of a 34-byte block, so only 2-byte alignment
        // holds and each word is two 16-bit loads.
        const unsigned long long off_lo = (unsigned long long)blk_lo * BLOCK_BYTES + 2 + kqsx * 4;
        const unsigned long long off_hi = (unsigned long long)blk_hi * BLOCK_BYTES + 2 + kqsx * 4;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int ROWS = MMQ_Y / WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v_lo[BATCH];
            int v_hi[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v_lo[u] = load_int_ua(row + off_lo);
                v_hi[u] = load_int_ua(row + off_hi);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                s_x[i * X_STRIDE + X_QS + lane] = v_lo[u];
                s_x[i * X_STRIDE + X_QS + MMQF_HALF_W + lane] = v_hi[u];
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four rows.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blk));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, which is shared with Q4_0: the staged row
    // layout and the one-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
