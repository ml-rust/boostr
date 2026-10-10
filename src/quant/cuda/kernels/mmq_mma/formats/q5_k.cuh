// Q5_K weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_dm.cuh"
#include "q4_k.cuh"

// Q5_K weight format policy, same contract as `MmqfQ80`.
//
// Q5_K is Q4_K with a fifth bit. Same super-block header (f16 `d`, f16
// `dmin`, the same 12-byte 6-bit scale/min packing), same eight 32-element
// sub-blocks, same dequant form `d * sc_j * q - dmin * m_j`, same sub-block
// PAIRING over `qs` — even sub-block takes the low nibbles, odd the high
// nibbles of the SAME 32-byte run. Only the quant is wider: unsigned 5-bit,
// four low bits from `qs` plus one bit from the 32-byte `qh` field.
//
// So everything but `stage` is shared with Q4_K: the staged row layout, the
// `(d * sc, -dmin * m)` f32 pairs at `X_DM`, and `mmqf_vec_dot_dm`.
//
// ALIGNMENT. 176 is a multiple of 16, so a row base and every super-block
// base inside it are 16-byte aligned, and `qs`@48, `qh`@16 and the `d`/`dmin`
// pair@0 are all 4-byte aligned. Plain `int` loads are legal here; unlike
// Q6_K this format needs no `load_int_ua`.
struct MmqfQ5K {
    // On-disk super-block: f16 d, f16 dmin, 12 packed 6-bit scale/min bytes,
    // 32 bytes of fifth bits, then 128 nibble-packed quants.
    static constexpr int BLOCK_BYTES = 176;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole. The ragged tail path is dead for this format and is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: identical to Q4_K's — 64 quant words (256 k-values,
    // one unsigned 0..31 value per int8 lane), then 8 `float2` holding
    // `(d * sc_j, -dmin * m_j)`, then padding. Equal stride means Q4_K, Q5_K
    // and Q6_K share the family's shared-memory request exactly.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q5_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`. Write `lane = 8*a + b` with `a = lane / 8` and
    // `b = lane % 8`. Lane `l` reads the aligned int at `qs + 4*l`, whose low
    // nibbles are 4 k-values of the EVEN sub-block `2*a` and whose high
    // nibbles are the same 4 positions of the ODD sub-block `2*a + 1` — the
    // pair shares one 32-byte run of `qs`, exactly as in Q4_K. The two staged
    // words land at `16*a + b` and 8 further along.
    //
    // The fifth bits come from the aligned int at `qh + 4*b`, which is `qh`
    // bytes `4*b .. 4*b+3`. In `qh` the BYTE index is the element within a
    // sub-block and the BIT index is the sub-block number, so the even word
    // takes bit `2*a` of each byte and the odd word bit `2*a + 1`.
    //
    // Both halves of that map were cross-checked element-by-element against
    // `dequant_q5k` in `src/quant/cpu/kernels/dequant_k_quants/q4k_q5k.rs`:
    // staged word `16*a + b` covers k-values `64*a + 4*b .. +3`, which is
    // sub-block `2*a`, elements `4*b .. 4*b+3` — the CPU kernel reads
    // `qs[(j/2)*32 + l]` and `qh[l] >> j` at exactly those indices. Getting
    // either index wrong does not fail loudly; it yields a tensor with the
    // right shape and RMS and the wrong values.
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
        // per row. `qh` starts at byte 16 of the super-block, `qs` at byte 48.
        const unsigned long long off_qs = off_blk + 48 + lane * 4;
        const unsigned long long off_qh = off_blk + 16 + (lane % 8) * 4;
        const unsigned int w_lo = 16 * (lane / 8) + lane % 8;
        const unsigned int sh = 2 * (lane / 8);  // 0, 2, 4 or 6

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
                // 4 low bits from `qs`, the fifth from `qh`. Unsigned 0..31
                // sits inside the signed int8 range the `mma` takes, so no
                // bias is applied; the minimum term is what carries the
                // format's asymmetry, exactly as in Q4_K.
                const int lo =
                    (vl[u] & 0x0F0F0F0F) | (((vh[u] >> sh) & 0x01010101) << 4);
                const int hi =
                    ((vl[u] >> 4) & 0x0F0F0F0F) | (((vh[u] >> (sh + 1)) & 0x01010101) << 4);
                s_x[i * X_STRIDE + X_QS + w_lo] = lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 8] = hi;
            }
        }

        // Scale/min pass: byte-for-byte the same as Q4_K's, because the
        // header is the same. Eight pairs per row, so a warp covers four
        // rows. The three packed bytes each lane needs are at row-invariant
        // offsets, so the first loop is pure loads and the 6-bit unpack
        // happens in the second, on registers.
        float2* s_xdm = (float2*)s_x;
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
            int scale;
            int minimum;
            q4k_scale_min_bytes(b_j[u], b_j4[u], b_jm4[u], (int)j, &scale, &minimum);
            const float d = __low2float(dm[u]);
            const float dmin = __high2float(dm[u]);
            // Stored as f32: no half round-trip on `d * sc`, which is the
            // term the parity bound is sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + j] =
                make_float2(d * (float)scale, -dmin * (float)minimum);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K: once staged, the two
    // formats' rows and arithmetic are identical.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
