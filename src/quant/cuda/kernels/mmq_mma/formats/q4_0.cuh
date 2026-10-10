// Q4_0 weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// Q4_0 weight format policy, same contract as `MmqfQ80`.
//
// Q4_0 is a legacy 32-element format like Q8_0, not a K-quant: `f16 d`@0 then
// 16 bytes holding 32 unsigned 4-bit quants, and the value is `d * (q - 8)`.
// Biasing by 8 during staging turns it into exactly Q8_0's staged row —
// signed quants in the int8 lanes plus one f32 scale per 32-element block —
// so the whole row layout and `vec_dot` are Q8_0's, and `stage` is the only
// thing this format defines for itself.
//
// ALIGNMENT. The block is 18 bytes, so a row base (`bpr * 18`) and every block
// base inside it are only 2-byte aligned. Every 4-byte read of `qs` therefore
// goes through `load_int_ua`; a plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it. The `f16 d` at byte 0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfQ40 {
    // On-disk block: one f16 scale then 16 bytes of nibble-packed quants.
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "Q4_0 must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile. A block holds 4
    // source ints and produces 8 staged words, so the warp's 32 lanes cover a
    // whole 256-k group with ONE global load each: lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`). Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q4_0` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_q4_0` in
    // `src/quant/cpu/kernels/dequant_simple.rs`: within one 32-element block,
    // element `j` (0..15) is the LOW nibble of `qs[j]` and element `j + 16` is
    // the HIGH nibble of the same byte. So the int at `qs + 4*w` (w = 0..3)
    // carries elements `4w..4w+3` in its low nibbles and `4w+16..4w+19` in its
    // high nibbles, which places the two staged words at `8*blk + w` and 4
    // further along. This is NOT Q4_K's map: Q4_K pairs sub-blocks across a
    // shared 32-byte run, while a Q4_0 block is a single 32-element run.
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
        // Q4_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        // The quants start at byte 2 of an 18-byte block, so only 2-byte
        // alignment holds and each word is two 16-bit loads.
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 2 + kqsx * 4;
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
                v[u] = load_int_ua(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                // The 8 bias is applied here, per byte, so the staged lanes are
                // signed -8..7 and `vec_dot` is Q8_0's unchanged. Inputs are
                // 0..15, so the subtract never saturates.
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(v[u] & 0x0F0F0F0F, 0x08080808);
                s_x[i * X_STRIDE + X_QS + w_lo + 4] =
                    __vsubss4((v[u] >> 4) & 0x0F0F0F0F, 0x08080808);
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
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blkp));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once the bias is folded
    // in during staging the two rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
