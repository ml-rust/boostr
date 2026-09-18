// PrismML-fork weight format policies for the feature-major MMQ family:
// PQ2_0, Q2_0 and Q1_0. Included by `quant_mmq_mma.cu` after `MmqfQ80` and
// `mmqf_vec_dot_d` are defined; it is not a compilation unit of its own.
//
// All three keep an f16 `d` at byte 0 and a dense bit-packed `qs` run at
// byte 2, element order byte-major, low bits first (`../prism_dequant.cuh`
// is the layout's one home). One block spans 64 (Q2_0) or 128 (PQ2_0, Q1_0)
// elements under ONE scale, where the legacy 32-element formats carry one
// scale per block. The MMA loop reads signed int8 lanes at `X_QS` and one
// f32 scale per 32 elements at `X_DS`, so staging expands the codes to int8
// and writes the block's `d` into each of the 32-element slots the block
// covers: replicated 4x for PQ2_0 and Q1_0, 2x for Q2_0. The staged row is
// then Q8_0's byte for byte and the whole `vec_dot` is `mmqf_vec_dot_d`.
//
// Value maps:
//   Q1_0        bit `j` of the run, set -> +d, clear -> -d
//   Q2_0/PQ2_0  code `j` (2 bits, low first) -> (code - 1) * d, {-1, 0, 1, 2}
//
// One `stage` body serves all three through `MmqfPrism`; the three structs
// below fix its block geometry and its expansion.
//
// Thread map, as `MmqfQ40::stage`: lane `l` owns 32-element chunk `l / 4`
// of the 256-k group and source word `l % 4` inside it, one warp stages one
// feature row per step and steps by the warp count, and the loads for a
// batch of rows are issued before any store. A Q4_0 word is a 4-byte int
// carrying two staged words 4 apart; here a source word is the 2 bytes (16
// codes) or the 1 byte (8 sign bits) that expand to two CONSECUTIVE staged
// words `8 * (l / 4) + 2 * (l % 4)` and the next, so lanes 0..3 cover a
// chunk's 8 staged words in order.
//
// ALIGNMENT. Blocks are 18 or 34 bytes, so a row base (`supers *
// BLOCK_BYTES`) and every block base inside it are only 2-byte aligned. The
// code-2 formats read their source word as an `unsigned short` at an even
// offset, which is `alignof(unsigned short)`; Q1_0 reads single bytes. The
// `f16 d` at byte 0 is 2-byte aligned, which is `alignof(__half)`, and is
// read directly.

#pragma once

#include "../prism_dequant.cuh"

// The shared body. `CHUNK_BYTES` is a 32-element chunk of the code run: 8
// bytes at 2 bits per element, 4 at 1 bit. `SIGN` picks the sign-bit
// expansion over the code-2 one.
template <int BLOCK_BYTES_, int BLOCK_ELEMS_, bool SIGN>
struct MmqfPrism {
    // On-disk block: one f16 scale then the packed code run.
    static constexpr int BLOCK_BYTES = BLOCK_BYTES_;
    static constexpr int BLOCK_ELEMS = BLOCK_ELEMS_;
    // 32-element activation chunks one weight block covers, and the byte
    // width of one such chunk inside the run.
    static constexpr int CHUNKS = BLOCK_ELEMS / 32;
    static constexpr int CHUNK_BYTES = SIGN ? 4 : 8;
    static_assert(2 + CHUNKS * CHUNK_BYTES == BLOCK_BYTES, "Block bytes do not cover the run.");
    // K is gated on `k % BLOCK_ELEMS == 0`, which is finer than a 256-k
    // group, so a row's last group can hold fewer than MMQF_ITER_B chunks
    // and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256
    // k-values), then 8 f32 chunk scales, then padding that makes the stride
    // an odd multiple of 4 ints so the strided fragment gathers hit all 32
    // banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 chunk scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "The prism formats must stage into the Q8_0 row; they share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 chunks) of the weight tile. `b0` and `bpr`
    // count 32-element activation chunks; chunk `c` sits in weight block
    // `c / CHUNKS` at run offset `(c % CHUNKS) * CHUNK_BYTES`. Independent
    // of MMQ_X.
    template <int MMQ_Y, int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // chunk within the 256-k group
        const unsigned int kqsx = lane % 4;  // source word within that chunk

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / CHUNKS;
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row. A
        // clamped chunk is staged but never consumed.
        const unsigned int chunk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned long long off_qs = (unsigned long long)(chunk / CHUNKS) * BLOCK_BYTES +
                                          GGUF_PRISM_QS_OFFSET + (chunk % CHUNKS) * CHUNK_BYTES +
                                          kqsx * (CHUNK_BYTES / 4);
        const unsigned int w_lo = kbx * 8 + kqsx * 2;

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
                if constexpr (SIGN) {
                    v[u] = (int)row[off_qs];
                } else {
                    v[u] = (int)*reinterpret_cast<const unsigned short*>(row + off_qs);
                }
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * WARPS) + warp;
                // The expansion sits in the SECOND loop, so it never separates
                // a global load from the next load's issue. Both expansions
                // produce signed int8 lanes, so `vec_dot` is Q8_0's unchanged.
                int q_lo;
                int q_hi;
                if constexpr (SIGN) {
                    prism_expand_sign8(v[u], &q_lo, &q_hi);
                } else {
                    prism_expand_code2x8(v[u] & 0xFF, v[u] >> 8, &q_lo, &q_hi);
                }
                s_x[i * X_STRIDE + X_QS + w_lo] = q_lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 1] = q_hi;
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four
        // rows. Every chunk slot reads the `d` of the block it lies in, which
        // is how one block scale is replicated into its CHUNKS slots.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int chunk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)(chunk_d / CHUNKS) * BLOCK_BYTES;

        constexpr int SROWS = MMQ_Y / (WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blk));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once the codes are
    // expanded and the scale replicated, the staged rows are the same.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// PQ2_0: 34 bytes / 128 elements, 2-bit codes; one scale over 4 chunks.
using MmqfPQ20 = MmqfPrism<34, 128, false>;
// Q2_0: 18 bytes / 64 elements, 2-bit codes; one scale over 2 chunks.
using MmqfQ20 = MmqfPrism<18, 64, false>;
// Q1_0: 18 bytes / 128 elements, sign bits; one scale over 4 chunks.
using MmqfQ10 = MmqfPrism<18, 128, true>;

static_assert(MmqfPQ20::CHUNKS == 4 && MmqfQ20::CHUNKS == 2 && MmqfQ10::CHUNKS == 4,
              "Chunk counts follow the block geometry.");
