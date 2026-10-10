// Q4_1 weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_dm.cuh"
#include "q4_k.cuh"

// Q4_1 weight format policy, same contract as `MmqfQ80`.
//
// Q4_1 is a legacy 32-element format, not a K-quant: `f16 d`@0, `f16 m`@2,
// then 16 bytes holding 32 unsigned 4-bit quants@4. The value is `d * q + m`.
//
// THE MIN SIGN. Q4_1 stages into Q4_K's row and shares `mmqf_vec_dot_dm`,
// which adds `pair.y * (activation scale * exact int16 block sum)`. Q4_K's
// value is `d * sc * q - dmin * m`, so Q4_K stores `-dmin * m_j` in `pair.y`.
// Q4_1 is ADDITIVE, so it stores `+m` there. Same `vec_dot`, opposite sign
// folded at staging — this is the one place in the family where the stored
// minimum is not negated.
//
// One `(d, m)` pair covers a whole 32-element block, so a 256-k staging group
// carries exactly 8 pairs, which is the granularity `mmqf_vec_dot_dm` already
// indexes with `kw / 8`.
//
// ALIGNMENT. The block is 20 bytes, a multiple of 4, so a row base
// (`bpr * 20`) and every block base inside it are 4-byte aligned, and `qs`@4
// and the `d`/`m` pair@0 land on multiples of 4. Plain `int` and `half2` loads
// are legal; unlike Q4_0 and Q5_0 this format needs no `load_int_ua`.
struct MmqfQ41 {
    // On-disk block: f16 scale, f16 minimum, 16 nibble-packed quant bytes.
    static constexpr int BLOCK_BYTES = 20;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q4_K's, int for int. 64 quant words (256 k-values,
    // one unsigned 0..15 value per int8 lane), then 8 `float2` holding
    // `(d, m)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ4K::X_QS && X_DM == MmqfQ4K::X_DM && X_STRIDE == MmqfQ4K::X_STRIDE,
        "Q4_1 must stage into the Q4_K row; the two share `mmqf_vec_dot_dm`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile, on Q4_0's lane map: a
    // block holds 4 source ints and produces 8 staged words, so the warp's 32
    // lanes cover a whole 256-k group with ONE global load each — lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`). Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q4_1` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q4_1` in `src/quant/cpu/kernels/dequant_simple.rs`: within one
    // 32-element block, element `j` (0..15) is the LOW nibble of `qs[j]` and
    // element `j + 16` is the HIGH nibble of the same byte. So the int at
    // `qs + 4*w` (w = 0..3) carries elements `4w..4w+3` in its low nibbles and
    // `4w+16..4w+19` in its high nibbles, placing the two staged words at
    // `8*blk + w` and 4 further along. This is Q4_0's map, NOT Q4_K's: a
    // legacy block is a single 32-element run, while Q4_K pairs sub-blocks
    // across a shared 32-byte run.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q4_1's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 4 + kqsx * 4;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
        const unsigned int w_lo = kbx * 8 + kqsx;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int ROWS = MMQ_Y / WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                // Unsigned 0..15 sits inside the signed int8 range the `mma`
                // takes, so no bias is applied; the minimum term below is what
                // carries the format's asymmetry, exactly as in Q4_K.
                s_x[i * X_STRIDE + X_QS + w_lo] = v[u] & 0x0F0F0F0F;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = (v[u] >> 4) & 0x0F0F0F0F;
            }
        }

        // Scale/min pass: eight pairs per row, so a warp covers four rows.
        float2* s_xdm = (float2*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        half2 dm[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            // `d` and `m` are adjacent f16 at byte 0 of a 4-aligned block, so
            // the pair is one aligned 4-byte load.
            dm[u] = *reinterpret_cast<const half2*>(blkp);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // `+m`, not `-m`: see THE MIN SIGN above. Stored as f32 for the
            // same parity reason as Q4_K — no half round-trip on the term the
            // GEMM/GEMV bound is sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + kbxd] =
                make_float2(__low2float(dm[u]), __high2float(dm[u]));
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K and Q5_K: once staged,
    // the rows and the two-term arithmetic are identical.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
