// Q6_K weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_sc16.cuh"

// Q6_K weight format policy, same contract as `MmqfQ80`.
//
// Two properties set this format apart from Q8_0 and Q4_K.
//
// ALIGNMENT. The super-block is 210 bytes, which is only 2-byte aligned, so a
// row base and every super-block base inside it are 2-byte aligned as well.
// Every 4-byte read of `ql`, `qh` or `scales` therefore goes through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS,
// which poisons the context for every later launch on it. The `half` scale at
// byte 208 is 2-byte aligned and is read directly.
//
// SCALE GRANULARITY. The scale changes every 16 elements, so a single
// `mma_m16n8k32_s8` cannot express one 32-k step: its two 16-k halves need
// different scales. Each 32-k step is therefore two `mma_m16n8k16_s8` calls,
// each scaled by its own `d * scale`. This is what the token-major
// `quant_mmq_q6_k_q8_1_mma` kernel in `q6_k.cu` does as well.
struct MmqfQ6K {
    // On-disk super-block: 128 bytes `ql`, 64 bytes `qh`, 16 signed `scales`,
    // then the f16 `d` at byte 208.
    static constexpr int BLOCK_BYTES = 210;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values, one signed 6-bit value
    // biased to -32..31 per int8 lane), then 16 f32 holding `d * scale_j`, one
    // per 16-element group, then padding.
    //
    // The scale is staged as f32 already multiplied by `d`, never as the raw
    // `int8` and never through `half`: half has an 11-bit significand, and
    // rounding `d * scale` to it perturbs every 16-element group's
    // contribution past the bound the GEMM/GEMV parity tests hold this path
    // to. Same stride as Q4_K, so the two share the family's shared-memory
    // request exactly.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q6_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`. Lane `l` reads the unaligned int at `ql + 4*l` and
    // the unaligned int at `qh + 4*(8*(l/16) + l%8)`, and produces TWO staged
    // words: the low nibbles at word `32*(l/16) + l%16` and the high nibbles
    // 16 words further along. `s = (l & 8) >> 2` picks the `qh` 2-bit field
    // for the low-nibble word; the high-nibble word takes `s + 4`. Under that
    // map staged element `k` belongs to scale group `k / 16` for every `k`,
    // so a staged word `w` takes scale `w / 4`.
    // One warp's share of a 256-k group as raw words; see `MmqfQ4K::Regs`.
    template <int MMQ_Y, int MMQ_X>
    struct Regs {
        static constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        static constexpr int ROWS = MMQ_Y / WARPS;
        static constexpr int SROWS = MMQ_Y / (WARPS * 2);
        int ql[ROWS];
        int qh[ROWS];
        __half d_h[SROWS];
        int sc[SROWS];
    };
    static constexpr bool PIPELINED = true;

    template <int MMQ_Y, int MMQ_X>
    static __device__ __forceinline__ void load(
        const unsigned char* __restrict__ weight, unsigned int N, unsigned int bpr,
        unsigned int feat0, unsigned int b0, Regs<MMQ_Y, MMQ_X>& r
    ) {
        using R = Regs<MMQ_Y, MMQ_X>;
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned long long off_blk =
            (unsigned long long)(b0 / (BLOCK_ELEMS / 32)) * BLOCK_BYTES;
        const unsigned long long off_ql = off_blk + lane * 4;
        const unsigned long long off_qh = off_blk + 128 + (8 * (lane / 16) + lane % 8) * 4;
#pragma unroll
        for (int u = 0; u < R::ROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS) + warp;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            r.ql[u] = load_int_ua(row + off_ql);
            r.qh[u] = load_int_ua(row + off_qh);
        }
        const unsigned int j = lane % 16;
        const unsigned int rsub = lane / 16;
#pragma unroll
        for (int u = 0; u < R::SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            r.d_h[u] = *reinterpret_cast<const __half*>(blk + 208);
            r.sc[u] = (int)*reinterpret_cast<const signed char*>(blk + 192 + j);
        }
    }

    template <int MMQ_Y, int MMQ_X>
    static __device__ __forceinline__ void store(
        int* __restrict__ s_x, const Regs<MMQ_Y, MMQ_X>& r
    ) {
        using R = Regs<MMQ_Y, MMQ_X>;
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int w_lo = 32 * (lane / 16) + lane % 16;
        const unsigned int sh = (lane & 8) >> 2;
#pragma unroll
        for (int u = 0; u < R::ROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS) + warp;
            store_quants(s_x, i, w_lo, sh, r.ql[u], r.qh[u]);
        }
        const unsigned int j = lane % 16;
        const unsigned int rsub = lane / 16;
#pragma unroll
        for (int u = 0; u < R::SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS * 2) + warp * 2 + rsub;
            store_scale(s_x, i, j, r.d_h[u], r.sc[u]);
        }
    }

    // One `ql`/`qh` word pair of row `i`: 4 low bits from `ql`, 2 high bits
    // from `qh`, then the -32 bias that turns the unsigned 0..63 value into
    // the signed one the `mma` takes.
    static __device__ __forceinline__ void store_quants(
        int* __restrict__ s_x, unsigned int i, unsigned int w_lo, unsigned int sh, int vl,
        int vh
    ) {
        const int lo = (vl & 0x0F0F0F0F) | (((vh >> sh) & 0x03030303) << 4);
        const int hi = ((vl >> 4) & 0x0F0F0F0F) | (((vh >> (sh + 4)) & 0x03030303) << 4);
        s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x20202020);
        s_x[i * X_STRIDE + X_QS + w_lo + 16] = __vsubss4(hi, 0x20202020);
    }

    static __device__ __forceinline__ void store_scale(
        int* __restrict__ s_x, unsigned int i, unsigned int j, __half d_h, int sc
    ) {
        float* s_xdf = (float*)s_x;
        s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h) * (float)sc;
    }

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
        // per row. `ql` starts at byte 0 of the super-block, `qh` at byte 128.
        const unsigned long long off_ql = off_blk + lane * 4;
        const unsigned long long off_qh = off_blk + 128 + (8 * (lane / 16) + lane % 8) * 4;
        const unsigned int w_lo = 32 * (lane / 16) + lane % 16;
        const unsigned int sh = (lane & 8) >> 2;  // 0 or 2

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
                vl[u] = load_int_ua(row + off_ql);
                vh[u] = load_int_ua(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                store_quants(s_x, i, w_lo, sh, vl[u], vh[u]);
            }
        }

        // Scale pass: sixteen per row, so a warp covers two rows. The scale
        // byte and `d` are at row-invariant offsets, so the first loop is pure
        // loads and the multiply happens in the second, on registers.
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;

        constexpr int SROWS = MMQ_Y / (WARPS * 2);
        __half d_h[SROWS];
        int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 208 is even and the super-block base is 2-byte aligned, so
            // the f16 load is aligned. The scale is a single signed byte, so
            // it carries no alignment requirement of its own.
            d_h[u] = *reinterpret_cast<const __half*>(blk + 208);
            sc[u] = (int)*reinterpret_cast<const signed char*>(blk + 192 + j);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            store_scale(s_x, i, j, d_h[u], sc[u]);
        }
    }

    // Forwards to `mmqf_vec_dot_sc16`, which is shared with Q3_K: the staged
    // row layout and the per-16 scale arithmetic are the same for both.
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
