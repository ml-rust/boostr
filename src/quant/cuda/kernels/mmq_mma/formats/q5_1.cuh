// Q5_1 weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_dm.cuh"
#include "q4_1.cuh"

// Q5_1 weight format policy, same contract as `MmqfQ80`.
//
// Q5_1 is Q4_1 with a fifth quant bit: `f16 d`@0, `f16 m`@2, a 32-bit `qh`@4
// holding one bit per element, then 16 low-nibble bytes@8. The value is
// `d * q + m` with `q` the unsigned 5-bit assembly, so the staged row, the
// `+m` sign and `mmqf_vec_dot_dm` are Q4_1's and only the quant assembly
// differs — the same relationship Q5_K has to Q4_K.
//
// ALIGNMENT. The block is 24 bytes, a multiple of 4, so a row base
// (`bpr * 24`) and every block base inside it are 4-byte aligned, and `qs`@8,
// `qh`@4 and the `d`/`m` pair@0 land on multiples of 4. Plain `int` and
// `half2` loads are legal; unlike Q5_0 this format needs no `load_int_ua`.
struct MmqfQ51 {
    // On-disk block: f16 scale, f16 minimum, 32-bit fifth-bit field, 16
    // nibble bytes.
    static constexpr int BLOCK_BYTES = 24;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q4_1's, int for int — 64 quant words (256 k-values,
    // one unsigned 0..31 value per int8 lane), then 8 `float2` holding
    // `(d, m)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ41::X_QS && X_DM == MmqfQ41::X_DM && X_STRIDE == MmqfQ41::X_STRIDE,
        "Q5_1 must stage into the Q4_1 row; the two share `mmqf_vec_dot_dm`."
    );

    // Stages 256 k-values (8 blocks), on Q4_1's lane map: lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`), one `qs` load and one `qh`
    // load each. Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q5_1` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q5_1` in `src/quant/cpu/kernels/dequant_simple.rs`. The nibble
    // half is Q4_1's: the int at `qs + 4*w` carries elements `4w..4w+3` in its
    // low nibbles and `4w+16..4w+19` in its high nibbles, so the two staged
    // words land at `8*blk + w` and 4 further along.
    //
    // The fifth bit of element `j` is bit `j` of the block's single 32-bit
    // `qh`, so this lane needs bits `4w..4w+3` and `4w+16..4w+19`.
    // Pre-shifting by `4*w` leaves those in bits 0..3 and 16..19, and the four
    // masked shifts move bit `t` of that into bit 4 of staged byte `t`. That
    // is llama.cpp's form, kept verbatim: the source bits are packed
    // contiguously inside one nibble rather than spread one per byte, so
    // Q5_K's single `(qh >> sh) & 0x01010101` trick does not apply here.
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
        // Q5_1's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 8 + kqsx * 4;
        const unsigned long long off_qh = (unsigned long long)blk * BLOCK_BYTES + 4;
        const unsigned int w_lo = kbx * 8 + kqsx;
        const unsigned int shq = 4 * kqsx;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int ROWS = MMQ_Y / WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vl[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vl[u] = *reinterpret_cast<const int*>(row + off_qs);
                vh[u] = *reinterpret_cast<const int*>(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const int qh = (int)((unsigned int)vh[u] >> shq);
                // 4 low bits from `qs`, the fifth from `qh`. Unsigned 0..31
                // sits inside the signed int8 range the `mma` takes, so no
                // bias is applied; the minimum term carries the asymmetry.
                int lo = vl[u] & 0x0F0F0F0F;
                lo |= (qh << 4) & 0x00000010;   // element 4w+0 -> byte 0 bit 4
                lo |= (qh << 11) & 0x00001000;  // element 4w+1 -> byte 1 bit 4
                lo |= (qh << 18) & 0x00100000;  // element 4w+2 -> byte 2 bit 4
                lo |= (qh << 25) & 0x10000000;  // element 4w+3 -> byte 3 bit 4
                int hi = (vl[u] >> 4) & 0x0F0F0F0F;
                hi |= (qh >> 12) & 0x00000010;  // element 4w+16 -> byte 0 bit 4
                hi |= (qh >> 5) & 0x00001000;   // element 4w+17 -> byte 1 bit 4
                hi |= (qh << 2) & 0x00100000;   // element 4w+18 -> byte 2 bit 4
                hi |= (qh << 9) & 0x10000000;   // element 4w+19 -> byte 3 bit 4
                s_x[i * X_STRIDE + X_QS + w_lo] = lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = hi;
            }
        }

        // Scale/min pass: byte-for-byte Q4_1's, because the header is the
        // same two f16 at byte 0. Eight pairs per row, so a warp covers four
        // rows.
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
            dm[u] = *reinterpret_cast<const half2*>(blkp);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // `+m`, not `-m`: Q5_1 is additive, see THE MIN SIGN on `MmqfQ41`.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + kbxd] =
                make_float2(__low2float(dm[u]), __high2float(dm[u]));
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_1, Q4_K and Q5_K.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
