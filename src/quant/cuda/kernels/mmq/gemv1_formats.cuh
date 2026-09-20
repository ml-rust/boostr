// Weight readers of the single-token (M = 1) MMQ body `gemv1_body.cuh`:
// how one lane pulls one 32-element chunk and one block scale out of the
// staged row of each format. Included by `gemv1_body.cuh` only.
//
// A reader gives `SPAN`, the bytes of one row that hold a 256-k group;
// `scale(row, o, j)`, block scale of chunk `j` (0..7) of the staged group
// as f32; and `read(row, o, j, w)`, chunk `j` as 8 int8x4 words. `o` is the
// span start's byte offset inside the row's first window. Every reader's
// words are signed int8 lanes, the same the tensor-core staging forms.
// `Lane` is what a reader keeps per lane across the K walk, built once by
// `setup(c)` from the lane's chunk slot `c` (0..3) and passed to `read`.
// `PAIRED_WORDS` says whether the words come in element order (word `i`
// holds elements `4i ..`) or in the reader's own order, with `act_pair(ln,
// p)` naming the activation word pair that word pair `p` dots with.
// `CODES_PLUS_ONE` says the words hold `value + 1` as UNSIGNED bytes rather
// than the value as signed bytes; the body then subtracts the chunk's
// activation sum, which the record header carries exactly, so the chunk's
// int dot is the same integer either way.
#pragma once

#include <cuda_fp16.h>

#include "../lowbit_dequant.cuh"
#include "split_range.cuh"

// Chunks per step: one 256-k group.
#define GEMV1_STEP_CHUNKS MMQF_ITER_B

// Returns `v` through a lane's own shuffle slot, which the compiler cannot
// rebuild from `v`'s inputs, so a per-lane constant built from a select
// chain is kept in its register across the K walk instead of being
// re-selected every step.
static __device__ __forceinline__ unsigned int gemv1_pin(unsigned int v) {
    return __shfl_sync(0xFFFFFFFFu, v, threadIdx.x % WARP_SIZE);
}

// The lane state of a reader that keeps none.
struct Gemv1NoLane {};

// The stored `half` bits to f32, the one conversion every scale takes.
static __device__ __forceinline__ float gemv1_half_bits_to_float(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

// `NW` consecutive 32-bit words from byte offset `p` (2-byte aligned) of a
// staged row. Reads `NW + 1` aligned words and funnel-shifts each pair;
// the extra word is inside the row's slot.
template <int NW>
static __device__ __forceinline__ void gemv1_row_words(
    const int* __restrict__ row, unsigned int p, int (&out)[NW]
) {
    const unsigned int* q = reinterpret_cast<const unsigned int*>(row) + p / 4;
    const unsigned int sh = (p & 3) * 8;
    unsigned int prev = q[0];
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        const unsigned int next = q[k + 1];
        out[k] = (int)__funnelshift_r(prev, next, sh);
        prev = next;
    }
}

// The 16 scale bits at byte offset `p` (2-byte aligned) of a staged row.
static __device__ __forceinline__ unsigned short gemv1_row_half(
    const int* __restrict__ row, unsigned int p
) {
    return reinterpret_cast<const unsigned short*>(row)[p / 2];
}

// Q8_0: one f16 scale then 32 int8 quants per 32-element block. The words
// are the raw quant bytes, as `MmqfQ80::stage` writes them.
struct Gemv1Q80 {
    static constexpr unsigned int BLOCK_BYTES = 34;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        return gemv1_half_bits_to_float(gemv1_row_half(row, o + j * BLOCK_BYTES));
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        gemv1_row_words<8>(row, o + j * BLOCK_BYTES + 2, w);
    }
};

// PQ2_0, Q2_0 and Q1_0: f16 `d` at byte 0, a dense code run at byte 2, one
// scale over `BLOCK_ELEMS / 32` chunks. `CHUNK_BYTES` is a chunk's share of
// the run: 8 bytes at 2 bits per element, 4 at 1 bit.
//
// `SIGN` (Q1_0) expands the sign bits to {-1, +1} through
// `lowbit_expand_sign8`; the code-2 formats (PQ2_0, Q2_0) expand each code
// byte pair to {-1, 0, 1, 2} through `lowbit_expand_code2x8`. Both are the
// words `MmqfLowbit::stage` writes.
template <int BLOCK_BYTES_, int BLOCK_ELEMS_, bool SIGN>
struct Gemv1Lowbit {
    static constexpr unsigned int BLOCK_BYTES = BLOCK_BYTES_;
    static constexpr unsigned int CHUNKS = BLOCK_ELEMS_ / 32;
    static constexpr unsigned int CHUNK_BYTES = SIGN ? 4 : 8;
    static constexpr int WORDS = CHUNK_BYTES / 4;
    static_assert(2 + CHUNKS * CHUNK_BYTES == BLOCK_BYTES, "Block bytes do not cover the run.");
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        return gemv1_half_bits_to_float(gemv1_row_half(row, o + (j / CHUNKS) * BLOCK_BYTES));
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const unsigned int blk = o + (j / CHUNKS) * BLOCK_BYTES;
        int v[WORDS];
        gemv1_row_words<WORDS>(row, blk + GGUF_LOWBIT_QS_OFFSET + (j % CHUNKS) * CHUNK_BYTES, v);
        if constexpr (SIGN) {
#pragma unroll
            for (int kqsx = 0; kqsx < 4; ++kqsx) {
                const int byte = (v[0] >> (8 * kqsx)) & 0xFF;
                lowbit_expand_sign8(byte, &w[2 * kqsx], &w[2 * kqsx + 1]);
            }
        } else {
#pragma unroll
            for (int kqsx = 0; kqsx < 4; ++kqsx) {
                const int v16 = (v[kqsx / 2] >> (16 * (kqsx % 2))) & 0xFFFF;
                lowbit_expand_code2x8(v16 & 0xFF, v16 >> 8, &w[2 * kqsx], &w[2 * kqsx + 1]);
            }
        }
    }
};

// PTQ1_0: `qs[24]` at 0, `qh[2]` at 24, f16 `d` at 26; 128 elements packed
// level-major over three runs (`gguf_ptq1_0_trit`). Blocks are 28 bytes and
// the row stride a multiple of 28, so `o` is a multiple of 4 and every
// block word is read whole.
//
// A chunk is 32 elements of one block, and each chunk slot `c` (`j % 4`)
// sees a different shape of the packing:
//
//   c   elements   source                              staged words
//   0    0..32     words 0..3 at levels 0 and 1        w0..3@L0, w0..3@L1
//   1   32..64     words 0..3 at levels 2 and 3        w0..3@L2, w0..3@L3
//   2   64..96     words 0..3 at level 4, then words   w0..3@L4, w4,5@L0,
//                  4..5 at levels 0 and 1              w4,5@L1
//   3   96..128    words 4..5 at levels 2, 3, 4, then  w4,5@L2, w4,5@L3,
//                  the `qh` word at levels 0..3        w4,5@L4, qh@L0,1,
//                                                      qh@L2,3
//
// One instruction stream serves all four: a lane opens six sources (`S0 ..
// S5`, each one block word at one base level through `Ptq10Lanes`) and
// takes the trits of each once, then advances `S2` and `S3` one level and
// takes those (`ptq1_0_lanes_trits`). The block word and the base level of
// each source are per-lane registers; the tail sources of slot 3 use the
// tail lane shape, the others the run shape. Every slot is eight words
// this way, but their order is not element order: `read` returns the words
// in the order `S0 S1 S2 S3 S4 S5 S2' S3'`, and `act_pair` gives the
// chunk's activation word pair each word pair dots with:
//
//   c   S0 S1   S2 S3   S4 S5   S2' S3'      act pairs
//   0   w0,1@L0 w2,3@L0 w0,1@L1 w2,3@L1      0 1 2 3
//   1   w0,1@L2 w2,3@L2 w0,1@L3 w2,3@L3      0 1 2 3
//   2   w0,1@L4 w4,5@L0 w2,3@L4 w4,5@L1      0 2 1 3
//   3   w4,5@L2 w4,5@L3 qh@L01, qh@L23  w4,5@L4   0 1 3 2
//
// The int dot is exact whichever order its products are formed in, so the
// chunk's dot is the same value `MmqfPTQ10::stage` and `mmqf_vec_dot_d`
// form (see the body's header).
struct Gemv1PTQ10 {
    static constexpr unsigned int BLOCK_BYTES = 28;
    static constexpr unsigned int CHUNKS = 4;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");
    // `read` returns the chunk's words in source order; `act_pair` maps
    // them to activation word pairs. The words hold the trit codes `trit +
    // 1` as unsigned bytes.
    static constexpr bool PAIRED_WORDS = true;
    static constexpr bool CODES_PLUS_ONE = true;

    // Per-lane constants of the table above, fixed by the chunk slot.
    struct Lane {
        // Byte offset in the block of `S0`'s word (`S1` is the next), of
        // `S2`'s (`S3` next), of `S4`'s and of `S5`'s.
        unsigned int a, b, d, d1;
        // Base-level multipliers of `S0, S1`, of `S2, S3`, of `S4` and of
        // `S5`: `pow3[level]`, or the tail's two-level pairs.
        unsigned int m01, m23, m4, m5;
        // Lane shape of `S4, S5`: run or tail.
        unsigned int mask45, sel45;
        // Activation word pairs of word pairs 1..3 (pair 0 is always 0).
        unsigned int p1, p2, p3;
    };

    static __device__ __forceinline__ Lane setup(unsigned int c) {
        Lane ln;
        ln.a = gemv1_pin(c == 3 ? 16u : 0u);
        ln.b = gemv1_pin(c < 2 ? 8u : 16u);
        ln.d = gemv1_pin(c == 2 ? 8u : c == 3 ? 24u : 0u);
        ln.d1 = gemv1_pin(ln.d + (c == 3 ? 0u : 4u));
        ln.m01 = gemv1_pin(c == 0 ? 1u : c == 2 ? 81u : 9u);
        ln.m23 = gemv1_pin(c == 0 ? 1u : c == 1 ? 9u : c == 2 ? 1u : 27u);
        ln.m4 = gemv1_pin(c == 0 ? 3u : c == 1 ? 27u : c == 2 ? 81u : 0x00030001u);
        ln.m5 = gemv1_pin(c == 3 ? 0x001B0009u : ln.m4);
        ln.mask45 = gemv1_pin(c == 3 ? 0x000000FFu : 0x00FF00FFu);
        ln.sel45 = gemv1_pin(c == 3 ? 0x4441u : 0x4341u);
        ln.p1 = gemv1_pin(c == 2 ? 2u : 1u);
        ln.p2 = gemv1_pin(c == 2 ? 1u : c == 3 ? 3u : 2u);
        ln.p3 = gemv1_pin(c == 3 ? 2u : 3u);
        return ln;
    }

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        const unsigned int blk = o + (j / CHUNKS) * BLOCK_BYTES;
        return gemv1_half_bits_to_float(gemv1_row_half(row, blk + GGUF_PTQ1_0_D_OFFSET));
    }

    // Activation word pair (0..3, 8 elements each) of word pair `p` of
    // `read`'s output: the table above.
    static __device__ __forceinline__ unsigned int act_pair(const Lane& ln, unsigned int p) {
        return p == 0 ? 0u : p == 1 ? ln.p1 : p == 2 ? ln.p2 : ln.p3;
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, const Lane& ln,
        int (&w)[8]
    ) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(row) + o +
                                 (j / CHUNKS) * BLOCK_BYTES;
        const unsigned int* pa = reinterpret_cast<const unsigned int*>(p + ln.a);
        const unsigned int* pb = reinterpret_cast<const unsigned int*>(p + ln.b);
        const unsigned int* pd = reinterpret_cast<const unsigned int*>(p + ln.d);
        const unsigned int* pd1 = reinterpret_cast<const unsigned int*>(p + ln.d1);
        const Ptq10Lanes s0 = ptq1_0_lanes_open_run(pa[0], ln.m01);
        const Ptq10Lanes s1 = ptq1_0_lanes_open_run(pa[1], ln.m01);
        Ptq10Lanes s2 = ptq1_0_lanes_open_run(pb[0], ln.m23);
        Ptq10Lanes s3 = ptq1_0_lanes_open_run(pb[1], ln.m23);
        const Ptq10Lanes s4 = ptq1_0_lanes_open(pd[0], ln.mask45, ln.sel45, ln.m4);
        const Ptq10Lanes s5 = ptq1_0_lanes_open(pd1[0], ln.mask45, ln.sel45, ln.m5);
        ptq1_0_lanes_mod(s2);
        ptq1_0_lanes_mod(s3);
        w[0] = (int)ptq1_0_lanes_codes_raw(s0);
        w[1] = (int)ptq1_0_lanes_codes_raw(s1);
        w[2] = (int)ptq1_0_lanes_codes_next(s2);
        w[3] = (int)ptq1_0_lanes_codes_next(s3);
        w[4] = (int)ptq1_0_lanes_codes_raw(s4);
        w[5] = (int)ptq1_0_lanes_codes_raw(s5);
        w[6] = (int)ptq1_0_lanes_codes_last(s2);
        w[7] = (int)ptq1_0_lanes_codes_last(s3);
    }
};

using Gemv1PQ20 = Gemv1Lowbit<34, 128, false>;
using Gemv1Q20 = Gemv1Lowbit<18, 64, false>;
using Gemv1Q10 = Gemv1Lowbit<18, 128, true>;
