// Q3_K weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../../decode.cuh"
#include "../common.cuh"
#include "../vec_dot_sc16.cuh"
#include "q6_k.cuh"

// Q3_K weight format policy, same contract as `MmqfQ80`.
//
// Q3_K is Q6_K's shape with a narrower quant: the same 16-element scale
// granularity, sixteen signed scales per super-block, and no minimum term. So
// everything but `stage` is shared with Q6_K — the staged row layout, the 16
// f32 `d * scale_j` at `X_DF`, and `mmqf_vec_dot_sc16`.
//
// ALIGNMENT. The super-block is 110 bytes, which is only 2-byte aligned, so a
// row base and every super-block base inside it are 2-byte aligned as well.
// Every 4-byte read of `hmask` or `qs` therefore goes through `load_int_ua`;
// a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the
// context for every later launch on it. The packed `scales` are read one byte
// at a time, which carries no alignment requirement, and the `half` `d` at
// byte 108 is 2-byte aligned and is read directly.
//
// THE QUANT. Three bits: two low bits from `qs`, one high bit from `hmask`,
// and the high bit is INVERTED — a SET bit means do NOT subtract 4. The value
// is `low2 - (hmask_bit ? 0 : 4)`, an int8 in [-4, 3], which is what
// `dequant_q3k` in `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`
// computes. Folding the -4 in during staging is what lets the staged lanes be
// signed and `vec_dot` be Q6_K's unchanged.
struct MmqfQ3K {
    // On-disk super-block: 32 bytes `hmask`, 64 bytes `qs`, 12 packed 6-bit
    // `scales`, then the f16 `d` at byte 108.
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q6_K's, int for int. 64 quant words (256 k-values,
    // one signed value in [-4, 3] per int8 lane), then 16 f32 holding
    // `d * scale_j`, one per 16-element group, then padding.
    //
    // The scale is staged as f32 already multiplied by `d`, never as the raw
    // `int8` and never through `half`: half has an 11-bit significand, and
    // rounding `d * scale` to it perturbs every 16-element group's
    // contribution past the bound the GEMM/GEMV parity tests hold this path
    // to.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "Q3_K must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q3_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_q3k` in
    // `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`. ggml splits a row
    // across 16 threads each emitting 4 words; this family gives a row a whole
    // warp, so lane `l` takes `c = l % 16` and shift level `t = l / 16`, and
    // emits the TWO words for levels `t` and `t + 2`.
    //
    // Lane `l` reads the unaligned int at `qs + 4*c` and the unaligned int at
    // `hmask + 4*(c % 8)`, the latter shifted right by `4 * (c / 8)` so the
    // four `hmask` bits this lane needs sit at bit 0..3 of each byte. Level
    // `t` then takes `qs` bits `2t..2t+1` and `hmask` bit `t`, and the two
    // staged words land at `32*(c/8) + 8*t + c%8` and 16 words further along.
    //
    // Under that map staged word `w` covers elements `4w..4w+3` of the
    // super-block in natural order, exactly as the CPU dequantizer emits them,
    // so element `k` belongs to scale group `k / 16` and staged word `w` takes
    // scale `w / 4` — the property `mmqf_vec_dot_sc16` relies on.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant byte offsets, so the addressing collapses to one add
        // per row. `hmask` starts at byte 0 of the super-block, `qs` at 32.
        const unsigned int c = lane % 16;   // `qs` int within the super-block
        const unsigned int t = lane / 16;   // shift level, 0 or 1
        const unsigned long long off_qs = off_blk + 32 + c * 4;
        const unsigned long long off_hm = off_blk + (c % 8) * 4;
        const unsigned int hsh = 4 * (c / 8);
        const unsigned int w_lo = 32 * (c / 8) + 8 * t + c % 8;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int ROWS = MMQ_Y / WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vq[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vq[u] = load_int_ua(row + off_qs);
                vh[u] = load_int_ua(row + off_hm);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const int hm = vh[u] >> hsh;
                // Two low bits from `qs`, then the `hmask` bit moved to bit 2
                // so a SET bit contributes +4. The -4 bias that follows turns
                // the pair into the signed [-4, 3] value the `mma` takes: a
                // set bit cancels the bias, a clear one leaves the -4. That
                // inversion is the format's, not a sign convention of this
                // kernel.
                const int lo = ((vq[u] >> (2 * t)) & 0x03030303)
                               | (((hm >> t) << 2) & 0x04040404);
                const int hi = ((vq[u] >> (2 * t + 4)) & 0x03030303)
                               | (((hm >> (t + 2)) << 2) & 0x04040404);
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x04040404);
                s_x[i * X_STRIDE + X_QS + w_lo + 16] = __vsubss4(hi, 0x04040404);
            }
        }

        // Scale pass: sixteen per row, so a warp covers two rows. The two
        // packed bytes each lane needs and `d` are at row-invariant offsets,
        // so the first loop is pure loads and the 6-bit unpack happens in the
        // second, on registers.
        float* s_xdf = (float*)s_x;
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;
        const unsigned int o_low = 96 + GGUF_Q3K_SC_LOW_BYTE(j);
        const unsigned int o_high = 96 + GGUF_Q3K_SC_HIGH_BYTE(j);

        constexpr int SROWS = MMQ_Y / (WARPS * 2);
        __half d_h[SROWS];
        unsigned int b_low[SROWS];
        unsigned int b_high[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 108 is even and the super-block base is 2-byte aligned, so
            // the f16 load is aligned. The two scale bytes are single-byte
            // reads and carry no alignment requirement of their own.
            d_h[u] = *reinterpret_cast<const __half*>(blk + 108);
            b_low[u] = blk[o_low];
            b_high[u] = blk[o_high];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            // Already biased by -32 by the decoder, so the scale is signed.
            const int sc = q3k_scale_bytes(b_low[u], b_high[u], (int)j);
            s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h[u]) * (float)sc;
        }
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K: once the -4 bias is
    // folded in during staging the two rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};
