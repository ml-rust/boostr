// Q5_0 weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// Q5_0 weight format policy, same contract as `MmqfQ80`.
//
// Q5_0 is a legacy 32-element format like Q4_0: `f16 d`@0, a 32-bit `qh`@2
// holding one fifth bit per element, then 16 bytes holding the low 4 bits of
// 32 quants@6. The value is `d * (q - 16)` with `q` the unsigned 5-bit
// assembly. Biasing by 16 during staging turns it into exactly Q8_0's staged
// row — signed quants in the int8 lanes plus one f32 scale per 32-element
// block — so the whole row layout and `vec_dot` are Q8_0's, and `stage` is the
// only thing this format defines for itself.
//
// ALIGNMENT. The block is 22 bytes, so a row base (`bpr * 22`) and every block
// base inside it are only 2-byte aligned, and `qh`@2 and `qs`@6 inherit that.
// EVERY 4-byte read here therefore goes through `load_int_ua`; a plain `int`
// load raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for
// every later launch on it. The `f16 d` at byte 0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfQ50 {
    // On-disk block: f16 scale, 32-bit fifth-bit field, 16 nibble bytes.
    static constexpr int BLOCK_BYTES = 22;
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
        "Q5_0 must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile, on Q4_0's lane map: a
    // block holds 4 source `qs` ints and produces 8 staged words, so the
    // warp's 32 lanes cover a whole 256-k group with one `qs` load and one
    // `qh` load each — lane maps to (block `lane / 4`, 4-k word `lane % 4`).
    // Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q5_0` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q5_0` in `src/quant/cpu/kernels/dequant_simple.rs`: within one
    // 32-element block, element `j` (0..15) is the LOW nibble of `qs[j]` and
    // element `j + 16` is the HIGH nibble of the same byte, exactly as in
    // Q4_0. So the int at `qs + 4*w` (w = 0..3) carries elements `4w..4w+3` in
    // its low nibbles and `4w+16..4w+19` in its high nibbles, placing the two
    // staged words at `8*blk + w` and 4 further along.
    //
    // The fifth bit of element `j` is bit `j` of the 32-bit `qh`, so the whole
    // block shares one `qh` word and this lane needs bits `4w..4w+3` for its
    // low word and `4w+16..4w+19` for its high one. Pre-shifting by `4*w`
    // leaves those in bits 0..3 and 16..19, and the four masked shifts below
    // move bit `t` of that into bit 4 of staged byte `t` — that is llama.cpp's
    // form, kept verbatim because the source bits are packed contiguously
    // inside one nibble rather than spread one per byte, so Q5_K's single
    // `(qh >> sh) & 0x01010101` trick does not apply here.
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
        // Q5_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        // Byte 6 and byte 2 of a 22-byte block: 2-byte aligned, so both reads
        // are `load_int_ua`.
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 6 + kqsx * 4;
        const unsigned long long off_qh = (unsigned long long)blk * BLOCK_BYTES + 2;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
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
                vl[u] = load_int_ua(row + off_qs);
                vh[u] = load_int_ua(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const int qh = (int)((unsigned int)vh[u] >> shq);
                // 4 low bits from `qs`, the fifth from `qh`, then the 16 bias
                // per byte so the staged lanes are signed -16..15 and
                // `vec_dot` is Q8_0's unchanged. Inputs are 0..31, so the
                // subtract never saturates.
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
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x10101010);
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = __vsubss4(hi, 0x10101010);
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

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0 and Q4_0: once the 16
    // bias is folded in during staging the three rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
