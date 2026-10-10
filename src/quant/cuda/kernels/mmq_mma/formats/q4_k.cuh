// Q4_K weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../../decode.cuh"
#include "../common.cuh"
#include "../vec_dot_dm.cuh"

// Q4_K weight format policy, same contract as `MmqfQ80`.
//
// The staged row is WIDER than Q8_0's: 64 quant words then 16 ints, because
// the scale/min pair is kept as two f32 rather than one `half2`. Half has an
// 11-bit significand, and `d * sc` rounded to it perturbs every 32-element
// sub-block's contribution by that much; the GEMM path then drifts from the
// GEMV path past the parity bound the backend tests hold both to. The pair is
// read once per fragment and reused across four `mma` results, so the cost of
// exact staged scales is one extra shared word per pair read and 8 extra ints
// per staged row.
//
// Q4_K dequantizes to `d * sc * q - dmin * m` with UNSIGNED 4-bit `q`. The
// minimum term is rank-1 over the k-block: it depends on the weight row and
// the activation block sum, not on the individual quants, so it folds into the
// existing accumulator as one extra multiply-add per fragment element. The
// block sum itself is read straight out of the activation record's header
// word, where the producer stores it as an EXACT int16 — see `vec_dot`.
struct MmqfQ4K {
    // On-disk super-block: f16 d, f16 dmin, 12 packed 6-bit scale/min bytes,
    // then 128 nibble-packed quants.
    static constexpr int BLOCK_BYTES = 144;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole. The ragged tail path is dead for this format and is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values, one unsigned nibble per
    // int8 lane), then 8 `float2` holding `(d * sc_j, -dmin * m_j)`, one per
    // 32-element sub-block, then padding. The `-1` on the minimum is folded in
    // here so the consumer is a plain multiply-add. Offsets are int counts like
    // Q8_0's, so a `float2` spans two of them: 16 ints for the eight pairs.
    // Both `X_DM` and `X_STRIDE` are even, which keeps every pair 8-byte
    // aligned and the access a single `ld.shared.v2.f32`.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q4_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`: lane `l` reads the aligned int at `qs + 4*l`, whose
    // low nibbles are 4 k-values of the EVEN sub-block `2*(l/8)` and whose
    // high nibbles are the same 4 positions of the ODD sub-block `2*(l/8)+1`.
    // A sub-block pair shares one 32-byte run of `qs`; it is not a 16-byte run
    // per sub-block. That places the two staged words at
    // `16*(l/8) + l%8` and 8 further along.
    // One warp's share of a 256-k group as raw words: the quant word of each
    // of its ROWS rows and the header of each of its SROWS scale rows. The
    // pipelined walk loads the next group into these while the current one
    // is consumed; `stage` is the same loads and stores back to back.
    template <int MMQ_Y, int MMQ_X>
    struct Regs {
        static constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        static constexpr int ROWS = MMQ_Y / WARPS;
        static constexpr int SROWS = MMQ_Y / (WARPS * 4);
        int q[ROWS];
        half2 dm[SROWS];
        unsigned int b_j[SROWS];
        unsigned int b_j4[SROWS];
        unsigned int b_jm4[SROWS];
    };
    static constexpr bool PIPELINED = true;

    // Loads a whole (never clamped) group's words for this warp.
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
        const unsigned long long off_qs = off_blk + 16 + lane * 4;
#pragma unroll
        for (int u = 0; u < R::ROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS) + warp;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            r.q[u] = *reinterpret_cast<const int*>(row + off_qs);
        }
        const unsigned int j = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int o_jm4 = j < 4 ? j : j - 4;
#pragma unroll
        for (int u = 0; u < R::SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            r.dm[u] = *reinterpret_cast<const half2*>(blk);
            const unsigned char* sc = blk + 4;
            r.b_j[u] = sc[j];
            r.b_j4[u] = sc[j + 4];
            r.b_jm4[u] = sc[o_jm4];
        }
    }

    // Unpacks a loaded group into the staged row layout `stage` documents.
    template <int MMQ_Y, int MMQ_X>
    static __device__ __forceinline__ void store(
        int* __restrict__ s_x, const Regs<MMQ_Y, MMQ_X>& r
    ) {
        using R = Regs<MMQ_Y, MMQ_X>;
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int w_lo = 16 * (lane / 8) + lane % 8;
#pragma unroll
        for (int u = 0; u < R::ROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS) + warp;
            store_quants(s_x, i, w_lo, r.q[u]);
        }
        const unsigned int j = lane % 8;
        const unsigned int rsub = lane / 8;
#pragma unroll
        for (int u = 0; u < R::SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * R::WARPS * 4) + warp * 4 + rsub;
            store_scale(s_x, i, j, r.dm[u], r.b_j[u], r.b_j4[u], r.b_jm4[u]);
        }
    }

    // One quant word of row `i`: low nibbles at `w_lo`, high nibbles 8 words
    // on. Unsigned 0..15 sits inside the signed int8 range the `mma` takes,
    // so no bias is applied; the minimum term carries the asymmetry.
    static __device__ __forceinline__ void store_quants(
        int* __restrict__ s_x, unsigned int i, unsigned int w_lo, int v
    ) {
        s_x[i * X_STRIDE + X_QS + w_lo] = v & 0x0F0F0F0F;
        s_x[i * X_STRIDE + X_QS + w_lo + 8] = (v >> 4) & 0x0F0F0F0F;
    }

    // Sub-block `j` of row `i`: `(d * sc_j, -dmin * m_j)` as f32, so `d * sc`
    // takes no half round-trip, which is the term the parity bound is
    // sensitive to.
    static __device__ __forceinline__ void store_scale(
        int* __restrict__ s_x, unsigned int i, unsigned int j, half2 dm, unsigned int b_j,
        unsigned int b_j4, unsigned int b_jm4
    ) {
        int scale;
        int minimum;
        q4k_scale_min_bytes(b_j, b_j4, b_jm4, (int)j, &scale, &minimum);
        const float d = __low2float(dm);
        const float dmin = __high2float(dm);
        float2* s_xdm = (float2*)s_x;
        s_xdm[(i * X_STRIDE + X_DM) / 2 + j] = make_float2(d * (float)scale, -dmin * (float)minimum);
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

        // 144 is a multiple of 16 and the quants start at byte 16, so every
        // word here is 4-byte aligned and a plain `int` load is legal.
        const unsigned long long off_qs = off_blk + 16 + lane * 4;
        const unsigned int w_lo = 16 * (lane / 8) + lane % 8;

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
                store_quants(s_x, i, w_lo, v[u]);
            }
        }

        // Scale/min pass: eight pairs per row, so a warp covers four rows.
        // The three packed bytes each lane needs are at row-invariant offsets,
        // so the first loop is pure loads and the 6-bit unpack happens in the
        // second, on registers.
        const unsigned int j = lane % 8;  // sub-block within the super-block
        const unsigned int rsub = lane / 8;
        const unsigned int o_j4 = j + 4;
        // For j < 4 the third byte is unused; re-reading sc[j] keeps the index
        // in range without a branch around the load.
        const unsigned int o_jm4 = j < 4 ? j : j - 4;

        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        half2 dm[SROWS];
        unsigned int b_j[SROWS];
        unsigned int b_j4[SROWS];
        unsigned int b_jm4[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // `d` and `dmin` are adjacent f16 at byte 0 of a 16-aligned block,
            // so the pair is one aligned 4-byte load.
            dm[u] = *reinterpret_cast<const half2*>(blk);
            const unsigned char* sc = blk + 4;  // 12 packed 6-bit scale/min bytes
            b_j[u] = sc[j];
            b_j4[u] = sc[o_j4];
            b_jm4[u] = sc[o_jm4];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            store_scale(s_x, i, j, dm[u], b_j[u], b_j4[u], b_jm4[u]);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, which is shared with Q5_K: the staged
    // row layout and the two-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
