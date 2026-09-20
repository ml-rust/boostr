// Weight readers of the single-token (M = 1) MMQ body `gemv1_body.cuh`:
// how one lane pulls one 32-element chunk and one block scale out of the
// staged row of each format. Included by `gemv1_body.cuh` only.
//
// A reader gives `SPAN`, the bytes of one row that hold a 256-k group;
// `scale(row, o, j)`, block scale of chunk `j` (0..7) of the staged group
// as f32; and `read(row, o, j, w)`, chunk `j` as 8 int8x4 words. `o` is the
// span start's byte offset inside the row's first window. Every reader's
// words are signed int8 lanes, the same the tensor-core staging forms.
#pragma once

#include <cuda_fp16.h>

#include "../lowbit_dequant.cuh"
#include "split_range.cuh"

// Chunks per step: one 256-k group.
#define GEMV1_STEP_CHUNKS MMQF_ITER_B

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

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        return gemv1_half_bits_to_float(gemv1_row_half(row, o + j * BLOCK_BYTES));
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, int (&w)[8]
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

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        return gemv1_half_bits_to_float(gemv1_row_half(row, o + (j / CHUNKS) * BLOCK_BYTES));
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, int (&w)[8]
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
// level-major over three runs (`gguf_ptq1_0_trit`). The 8 elements `e0 ..
// e0 + 8` of one staged word pair read 8 consecutive bytes at one trit
// level, except the `qh` tail (`e0 == 120`), which reads the two `qh` bytes
// at four levels; `ptq1_0_expand4` takes a `(mask, mul)` pair for each
// shape. The offsets and multipliers are `MmqfPTQ10::stage`'s. The tail's
// second word repeats its first: 4 bytes at offset 24 close the block.
// Blocks are 28 bytes and the row stride a multiple of 28, so `o` is a
// multiple of 4 and every group word is read whole.
struct Gemv1PTQ10 {
    static constexpr unsigned int BLOCK_BYTES = 28;
    static constexpr unsigned int CHUNKS = 4;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        const unsigned int blk = o + (j / CHUNKS) * BLOCK_BYTES;
        return gemv1_half_bits_to_float(gemv1_row_half(row, blk + GGUF_PTQ1_0_D_OFFSET));
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, int (&w)[8]
    ) {
        const unsigned int blk = o + (j / CHUNKS) * BLOCK_BYTES;
        const unsigned int* words = reinterpret_cast<const unsigned int*>(row);
#pragma unroll
        for (int kqsx = 0; kqsx < 4; ++kqsx) {
            const unsigned int e0 = 32 * (j % CHUNKS) + 8 * kqsx;
            const bool tail = e0 == 120;
            unsigned int group_off;
            int level;
            if (e0 < 80) {
                group_off = e0 % 16;
                level = (int)(e0 / 16);
            } else if (e0 < 120) {
                group_off = 16;
                level = (int)((e0 - 80) / 8);
            } else {
                group_off = 24;
                level = 0;  // unused: the tail's multipliers are fixed below
            }
            const unsigned int p = ptq1_0_pow3(level);
            const unsigned int mask = tail ? 0x000000FFu : 0x00FF00FFu;
            const unsigned int mul_lo = tail ? 0x00030001u : p;  // levels 0, 1 of qh
            const unsigned int mul_hi = tail ? 0x001B0009u : p;  // levels 2, 3 of qh
            const unsigned int v0 = words[(blk + group_off) / 4];
            const unsigned int v1 = words[(blk + (tail ? group_off : group_off + 4)) / 4];
            w[2 * kqsx] = ptq1_0_expand4(v0, mask, mul_lo);
            w[2 * kqsx + 1] = ptq1_0_expand4(v1, mask, mul_hi);
        }
    }
};

using Gemv1PQ20 = Gemv1Lowbit<34, 128, false>;
using Gemv1Q20 = Gemv1Lowbit<18, 64, false>;
using Gemv1Q10 = Gemv1Lowbit<18, 128, true>;
