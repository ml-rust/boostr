// Q2_K weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../common.cuh"
#include "../vec_dot_sc16.cuh"

// Q2_K weight format policy, same contract as `MmqfQ80`.
//
// The hardest format in the family: it is the only one with BOTH a per-16
// scale granularity and a minimum term. So it takes Q6_K/Q3_K's two 16-k MMAs
// per 32-k step AND Q4_K/Q5_K's rank-1 minimum correction, at twice Q4_K's
// granularity. `mmqf_vec_dot_sc16` carries both under `MIN = true`.
//
// ALIGNMENT. 84 is a multiple of 4, so a row base (`supers * 84`) and every
// super-block base inside it are 4-byte aligned, and `scales`@0, `qs`@16 and
// the `d`/`dmin` pair@80 all land on multiples of 4. Plain `int` and `half2`
// loads are legal; unlike Q6_K and Q3_K this format needs no `load_int_ua`.
//
// THE QUANT. Two bits, UNSIGNED, in 0..3 — no bias is folded in. The format's
// asymmetry is carried entirely by the minimum term, as in Q4_K and Q5_K.
//
// THE MINIMUM. `dequant_q2k` computes `d * (sc[i] & 0x0F) * q - dmin *
// (sc[i] >> 4)` with `i = k / 16`. The minimum does not depend on `q` at all,
// so over a k-block it is rank-1 in the activation's quant SUM, exactly as in
// Q4_K — but over 16 elements, not 32. The activation record's header sum is
// per 32, so `mmqf_stage_y_sums` derives the split once per staged activation
// tile; see that function for why the derived half is exact and why an
// all-ones `mma` is not used.
struct MmqfQ2K {
    // On-disk super-block: 16 packed scale/min bytes, 64 bytes of 2-bit
    // quants, then f16 `d` and f16 `dmin`.
    static constexpr int BLOCK_BYTES = 84;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // The minimum changes every 16 elements, so `vec_dot` needs a per-16
    // activation sum the shared record does not carry.
    static constexpr int Y_SCRATCH = MMQF_Y_SCRATCH;

    // Staged weight row: 64 quant words (256 k-values, one unsigned 0..3 value
    // per int8 lane), then 16 `float2` holding `(d * sc_j, -dmin * m_j)`, one
    // per 16-element group, then padding. That is 32 ints of scale record —
    // twice Q4_K's, because the granularity is twice as fine — which makes
    // this the widest row in the family.
    //
    // The pair is staged as f32 already multiplied by `d`/`dmin`, never
    // through `half`: half has an 11-bit significand, and rounding `d * sc` to
    // it perturbs every 16-element group's contribution past the bound the
    // GEMM/GEMV parity tests hold this path to.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 100;
    static_assert(X_DM + 32 <= X_STRIDE, "Weight row too short: 16 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, derived from `dequant_q2k` in
    // `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs` and structurally the
    // same as Q3_K's. That kernel emits element
    // `y = 128*n + 32*t + 16*h + l` from `qs[32*n + 16*h + l] >> 2*t`, with
    // scale index `y / 16`. Four consecutive elements therefore come from the
    // aligned int at `qs + 4*c` with `c = 8*n + 4*h + m`, and land at staged
    // word `32*(c/8) + 8*t + (c%8)`.
    //
    // This family gives a row a whole warp, so lane `l` takes `c = l % 16` and
    // shift level `t = l / 16`, and emits the TWO words for levels `t` and
    // `t + 2`, the second 16 words further along. Under that map staged word
    // `w` covers elements `4w .. 4w+3` in natural order, so staged word `w`
    // takes scale group `w / 4` — the property `mmqf_vec_dot_sc16` relies on.
    // The map was simulated element-by-element against the CPU dequantizer
    // over a random super-block; a wrong index does not fail loudly, it yields
    // a tensor with the right shape and RMS and the wrong values.
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
        // per row. `scales` starts at byte 0 of the super-block, `qs` at 16.
        const unsigned int c = lane % 16;  // `qs` int within the super-block
        const unsigned int t = lane / 16;  // shift level, 0 or 1
        const unsigned long long off_qs = off_blk + 16 + c * 4;
        const unsigned int w_lo = 32 * (c / 8) + 8 * t + c % 8;

        constexpr int WARPS = MMQF_WARPS_OF(MMQ_Y);
        constexpr int ROWS = MMQ_Y / WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vq[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vq[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                // Unsigned 0..3 sits inside the signed int8 range the `mma`
                // takes, so no bias is applied.
                s_x[i * X_STRIDE + X_QS + w_lo] = (vq[u] >> (2 * t)) & 0x03030303;
                s_x[i * X_STRIDE + X_QS + w_lo + 16] = (vq[u] >> (2 * t + 4)) & 0x03030303;
            }
        }

        // Scale/min pass: sixteen pairs per row, so a warp covers two rows.
        // One scale byte carries both fields — low nibble the scale, high
        // nibble the minimum — so this format needs no 6-bit unpack. The byte
        // and the `d`/`dmin` pair are at row-invariant offsets, so the first
        // loop is pure loads and the multiplies happen in the second, on
        // registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;

        constexpr int SROWS = MMQ_Y / (WARPS * 2);
        half2 dm[SROWS];
        unsigned int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 80 is a multiple of 4 and the super-block base is 4-byte
            // aligned, so the adjacent `d`/`dmin` f16 are one aligned 4-byte
            // load. The scale byte carries no alignment requirement.
            dm[u] = *reinterpret_cast<const half2*>(blk + 80);
            sc[u] = blk[j];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 2) + warp * 2 + rsub;
            const float d = __low2float(dm[u]);
            const float dmin = __high2float(dm[u]);
            // The `-1` on the minimum is folded in here so the consumer is a
            // plain multiply-add.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + j] =
                make_float2(d * (float)(sc[u] & 0x0Fu), -dmin * (float)(sc[u] >> 4));
        }
    }

    // Forwards to `mmqf_vec_dot_sc16` with the minimum term switched on. Q6_K
    // and Q3_K instantiate the same body with `MIN = false`, which drops both
    // the pair load and the correction.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, true, X_QS, X_DM, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};
