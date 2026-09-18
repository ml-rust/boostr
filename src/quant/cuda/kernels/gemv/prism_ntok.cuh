// Token-batched dp4a GEMV decode policies for the PrismML-fork formats
//
// Q1_0, Q2_0 and PQ2_0 share one block geometry with each other — an f16
// scale at byte 0, then a dense bit-packed `qs` run, element order byte-major
// low bits first — but not with the legacy 32-element formats: one block
// spans 64 (Q2_0) or 128 (Q1_0, PQ2_0) elements under ONE scale. The batched
// MWR body in `legacy_ntok.cuh` walks K in 32-element chunks, one Q8_1
// activation block per chunk, so these policies decode a 32-element chunk of
// a wider block. `CHUNKS_PER_BLOCK` tells the body how many chunks share a
// block base and scale; the legacy four declare 1.
//
// Value maps (`../prism_dequant.cuh` is the scalar reference, and
// `src/quant/cpu/kernels/dequant_prism.rs` the CPU mirror):
//   Q1_0        bit `j` of the run, set -> +d, clear -> -d
//   Q2_0/PQ2_0  code `j` (2 bits, low first) -> (code - 1) * d, {-1, 0, 1, 2}
// Neither has a minimum term, so `HAS_MIN` is false and the body's block-sum
// correction compiles out.
//
// The int8x4 expansions `prism_expand_code2x8` and `prism_expand_sign8`
// live in `../prism_dequant.cuh`, shared with the feature-major MMQ staging
// in `../mmq/prism_tiles.cuh`. A lane here owns elements `4w..4w+3` and
// `4w+16..4w+19` of its chunk, so the 16-bit (Q2_0) or 8-bit (Q1_0) input
// is assembled from those two positions first. The permutes then land the
// low half in `v_lo` and the high half in `v_hi`, which is what the body
// dot-products against the Q8_1 record's bytes `4 + 4w` and `20 + 4w`.
//
// ALIGNMENT. Q1_0's and Q2_0's blocks are 18 bytes and PQ2_0's 34, so every
// block base is only 2-byte aligned. Q1_0's chunk word goes through
// `load_int_ua`; the code-2 formats read single bytes and need nothing.

#pragma once

#include "common.cuh"
#include "../prism_dequant.cuh"

// ── Per-format decode policies ──────────────────────────────────────────
//
// Contract, as `legacy_ntok.cuh` states it, with `w` widened: `w` is the
// 4-element source word within the BLOCK, 0..4*CHUNKS_PER_BLOCK-1, so chunk
// `w / 4` and word `w % 4` inside it. Output is the same:
//   *d     — the block scale
//   *m     — 0, no minimum term
//   *v_lo  — chunk elements 4(w%4)..+3 as int8x4
//   *v_hi  — chunk elements 4(w%4)+16..+19 as int8x4

// A 32-element chunk of a code-2 run is 8 bytes; word `w4` of it reads byte
// `w4` (elements 4w4..4w4+3) and byte `w4 + 4` (elements 4w4+16..4w4+19).
static __device__ __forceinline__ void prism_decode_code2(
    const unsigned char* __restrict__ blk, int w,
    float* d, float* m, int* v_lo, int* v_hi
) {
    *d = prism_load_d(blk);
    *m = 0.0f;
    const unsigned char* qs = blk + GGUF_PRISM_QS_OFFSET + (w / 4) * 8;
    const int w4 = w % 4;
    prism_expand_code2x8((int)qs[w4], (int)qs[w4 + 4], v_lo, v_hi);
}

// Q1_0: 18 bytes / 128 elements, 4 chunks. A chunk is one 32-bit word of
// sign bits; word `w4` owns bits 4w4..4w4+3 and 4w4+16..4w4+19 of it, packed
// here into one byte for the expansion.
struct PrismQ10 {
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int CHUNKS_PER_BLOCK = 4;
    static constexpr bool HAS_MIN = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        *d = prism_load_d(blk);
        *m = 0.0f;
        // 2-byte aligned block base: unaligned load, see ALIGNMENT above.
        const unsigned int chunk =
            (unsigned int)load_int_ua(blk + GGUF_PRISM_QS_OFFSET + (w / 4) * 4);
        const int sh = (w % 4) * 4;
        const int bits8 = (int)(((chunk >> sh) & 0xFu) | (((chunk >> (16 + sh)) & 0xFu) << 4));
        prism_expand_sign8(bits8, v_lo, v_hi);
    }
};

// Q2_0: 18 bytes / 64 elements, 2 chunks.
struct PrismQ20 {
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int CHUNKS_PER_BLOCK = 2;
    static constexpr bool HAS_MIN = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        prism_decode_code2(blk, w, d, m, v_lo, v_hi);
    }
};

// PQ2_0: 34 bytes / 128 elements, 4 chunks. Q2_0's code space at double the
// run length.
struct PrismPQ20 {
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int CHUNKS_PER_BLOCK = 4;
    static constexpr bool HAS_MIN = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        prism_decode_code2(blk, w, d, m, v_lo, v_hi);
    }
};
