// IQ4_XS weight format policy of the feature-major MMQ family. Not a
// compilation unit of its own.
#pragma once

#include "../../decode.cuh"
#include "../common.cuh"
#include "../vec_dot_d.cuh"
#include "q8_0.cuh"

// IQ4_XS weight format policy, same contract as `MmqfQ80`.
//
// IQ4_XS is IQ4_NL's codebook over a 256-element super-block: `f16 d`@0,
// `u16 scales_h`@2, `u8 scales_l[4]`@4, `u8 qs[128]`@8. Eight sub-blocks of 32
// elements each take a 6-bit scale assembled from a `scales_l` nibble and two
// `scales_h` bits, biased by 32: `dl = d * (ls - 32)`. `scales_h` carries the
// high bits of ALL EIGHT sub-blocks (2 bits each = 16), so reading it as one
// byte drops half of them and shifts `scales_l` by one.
//
// A staged 256-k group is exactly one super-block and its scale granularity is
// 32 elements, which is the granularity `mmqf_vec_dot_d` already indexes at.
// With the codebook values signed int8, the staged row is Q8_0's unchanged and
// so is `vec_dot`; only `stage` is this format's own.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (ls - 32)` to it perturbs
// every sub-block's contribution past the bound the GEMM/GEMV parity tests
// hold this path to.
//
// ALIGNMENT. The super-block is 136 bytes, a multiple of 8, so a row base
// (`supers * 136`) and every super-block base inside it are 8-byte aligned.
// `qs`@8 is therefore 4-byte aligned and read with plain `int` loads;
// `scales_h`@2 is 2-byte aligned, which is `alignof(unsigned short)`, and the
// `f16 d`@0 is 2-byte aligned, which is `alignof(__half)`. The `scales_l`
// bytes carry no alignment requirement of their own.
struct MmqfIQ4XS {
    // On-disk super-block: f16 `d`@0, u16 `scales_h`@2, 4 `scales_l`@4, then
    // 128 bytes of nibble-packed codebook indices@8.
    static constexpr int BLOCK_BYTES = 136;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 sub-block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 sub-block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ4_XS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`. `qs` holds 32 ints and the warp has 32 lanes, so each lane
    // issues ONE global load and produces two staged words. Independent of
    // MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_iq4_xs` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_iq4_xs` in
    // `src/quant/cpu/kernels/dequant_iq4.rs`. Lane `l` reads the int at
    // `qs + 4*l`, which is bytes `4*(l % 4) .. 4*(l % 4) + 3` of sub-block
    // `l / 4`. Split-half nibble order puts its low nibbles at elements
    // `32*(l/4) + 4*(l%4) ..+3` and its high nibbles 16 elements further on,
    // so the staged words are `8*(l/4) + l%4` and 4 beyond. Under that map a
    // staged word `w` belongs to sub-block `w / 8`, which is exactly the scale
    // index `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        const unsigned long long off_qs = off_blk + 8 + lane * 4;
        const unsigned int w_lo = 8 * (lane / 4) + lane % 4;

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
                // The codebook lookup sits in the SECOND loop, so it never
                // separates a global load from the next load's issue. Table
                // values are signed int8 already, so the staged lanes need no
                // bias and `vec_dot` is Q8_0's unchanged.
                int q_lo;
                int q_hi;
                gguf_iq4_table_lookup(v[u], &q_lo, &q_hi);
                s_x[i * X_STRIDE + X_QS + w_lo] = q_lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = q_hi;
            }
        }

        // Scale pass: eight per row, so a warp covers four rows. `d`,
        // `scales_h` and the `scales_l` byte are at row-invariant offsets, so
        // the first loop is pure loads and the 6-bit assembly happens in the
        // second, on registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element sub-block
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sl = off_blk + 4 + sb / 2;

        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        __half d_h[SROWS];
        unsigned int sh[SROWS];
        unsigned int sl[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            sh[u] = *reinterpret_cast<const unsigned short*>(row + off_blk + 2);
            sl[u] = row[off_sl];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            // Low 4 bits from the `scales_l` nibble for this sub-block, high 2
            // bits from `scales_h` bits `2*sb`, then the -32 bias.
            const int ls =
                (int)(((sl[u] >> (4 * (sb % 2))) & 0x0Fu) | (((sh[u] >> (2 * sb)) & 0x03u) << 4));
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (float)(ls - 32);
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: the staged row layout
    // and the one-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};
