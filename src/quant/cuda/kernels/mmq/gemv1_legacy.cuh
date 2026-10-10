// Weight readers of the single-token (M = 1) MMQ body for the legacy
// 32-element blocks and the two IQ4 codebook formats: Q4_0, Q5_0, IQ4_NL,
// IQ4_XS, Q4_1 and Q5_1. Same contract as the readers in
// `gemv1_formats.cuh`, which this file includes. Included by
// `gemv1_chunks.cuh` only.
//
// WORD MAPS. Staged word `w` of a 256-k group holds elements `4w .. 4w + 3`,
// so word `i` of chunk `j` is elements `32j + 4i .. +3`. Every map below is
// the inverse of the corresponding `stage` in `mmq_mma/formats/`, which is
// where the map is derived and cross-checked against the CPU dequantizers.
//
// The four 32-element formats and IQ4_NL share ONE map, the split-half
// nibble order: inside a 32-element block element `e` (0..15) is the LOW
// nibble of `qs[e]` and element `e + 16` the HIGH nibble of the same byte.
// So the source int at `qs + 4 * i` (i = 0..3) carries elements
// `4i .. 4i + 3` in its low nibbles and `4i + 16 .. 4i + 19` in its high
// nibbles, which is words `i` and `i + 4` of the chunk. IQ4_XS is the same
// map per 32-element sub-block of its 256-element super-block.
//
// Q4_0 and Q5_0 fold their bias into the lanes (`-8`, `-16`) so the int dot
// takes signed values and the epilogue is the plain `d` form. Q4_1 and Q5_1
// keep the raw unsigned 0..15 and 0..31 lanes and carry `(d, +m)` into the
// `dm` epilogue, the ADDITIVE sign their dequant form asks for. IQ4_NL and
// IQ4_XS replace the bias with the signed codebook lookup.
#pragma once

#include <cuda_fp16.h>

#include "../decode.cuh"
#include "gemv1_formats.cuh"

// Q4_0: f16 `d`@0 then 16 nibble bytes@2 per 32-element block. The 18-byte
// block stride is only 2-byte aligned, so `o` is even and every word read
// goes through `gemv1_row_words`. The value is `d * (q - 8)`, so the lanes
// carry the `-8` bias and the epilogue is a scale alone.
struct Gemv1Q40 {
    static constexpr unsigned int BLOCK_BYTES = 18;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 2 == 0, "Funnel reads need a 2-byte block stride.");
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_D;
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
        int v[4];
        gemv1_row_words<4>(row, o + j * BLOCK_BYTES + 2, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = __vsubss4(v[i] & 0x0F0F0F0F, 0x08080808);
            w[i + 4] = __vsubss4((v[i] >> 4) & 0x0F0F0F0F, 0x08080808);
        }
    }
};

// The fifth quant bit of Q5_0 and Q5_1: bit `e` of the block's 32-bit `qh`
// belongs to element `e`, so word `i` needs bits `4i .. 4i + 3` for its low
// word and `4i + 16 .. 4i + 19` for its high one. Pre-shifting `qh` by
// `4 * i` leaves those in bits 0..3 and 16..19, and the four masked shifts
// move bit `t` of that into bit 4 of staged byte `t`. The source bits are
// packed contiguously inside one nibble rather than spread one per byte, so
// Q5_K's single `(qh >> sh) & 0x01010101` form does not apply.
static __device__ __forceinline__ void gemv1_q5_fifth_bits(
    int qh, int* lo, int* hi
) {
    *lo |= (qh << 4) & 0x00000010;   // element 4i+0 -> byte 0 bit 4
    *lo |= (qh << 11) & 0x00001000;  // element 4i+1 -> byte 1 bit 4
    *lo |= (qh << 18) & 0x00100000;  // element 4i+2 -> byte 2 bit 4
    *lo |= (qh << 25) & 0x10000000;  // element 4i+3 -> byte 3 bit 4
    *hi |= (qh >> 12) & 0x00000010;  // element 4i+16 -> byte 0 bit 4
    *hi |= (qh >> 5) & 0x00001000;   // element 4i+17 -> byte 1 bit 4
    *hi |= (qh << 2) & 0x00100000;   // element 4i+18 -> byte 2 bit 4
    *hi |= (qh << 9) & 0x10000000;   // element 4i+19 -> byte 3 bit 4
}

// Q5_0: f16 `d`@0, 32-bit `qh`@2, 16 nibble bytes@6 per 32-element block. The
// 22-byte block stride is only 2-byte aligned, so `o` is even and every word
// read goes through `gemv1_row_words`. The value is `d * (q - 16)` over the
// unsigned 5-bit assembly, so the lanes carry the `-16` bias.
struct Gemv1Q50 {
    static constexpr unsigned int BLOCK_BYTES = 22;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 2 == 0, "Funnel reads need a 2-byte block stride.");
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_D;
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
        // `qh`@2 and `qs`@6 are contiguous, so one walk of 5 words covers
        // both: word 0 is `qh`, words 1..4 are the nibble run.
        int v[5];
        gemv1_row_words<5>(row, o + j * BLOCK_BYTES + 2, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qh = (int)((unsigned int)v[0] >> (4 * i));
            int lo = v[i + 1] & 0x0F0F0F0F;
            int hi = (v[i + 1] >> 4) & 0x0F0F0F0F;
            gemv1_q5_fifth_bits(qh, &lo, &hi);
            w[i] = __vsubss4(lo, 0x10101010);
            w[i + 4] = __vsubss4(hi, 0x10101010);
        }
    }
};

// IQ4_NL: f16 `d`@0 then 16 bytes@2 holding 32 4-bit INDICES into the signed
// codebook `KVALUES_IQ4NL`. The 18-byte block stride is only 2-byte aligned,
// so every word read goes through `gemv1_row_words`. The table values are
// signed int8 already, so there is no bias and the epilogue is a scale alone.
struct Gemv1IQ4NL {
    static constexpr unsigned int BLOCK_BYTES = 18;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 2 == 0, "Funnel reads need a 2-byte block stride.");
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_D;
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
        int v[4];
        gemv1_row_words<4>(row, o + j * BLOCK_BYTES + 2, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            gguf_iq4_table_lookup(v[i], &w[i], &w[i + 4]);
        }
    }
};

// IQ4_XS: IQ4_NL's codebook over a 256-element super-block. f16 `d`@0, u16
// `scales_h`@2, 4 bytes `scales_l`@4, then 128 bytes of nibble-packed indices
// @8, 16 bytes per 32-element sub-block. `CHUNKS` is 8, so one step is
// exactly one super-block per feature and chunk `j` is sub-block `j`: its
// four source ints are at `8 + 16 * j`.
//
// One 6-bit scale per 32 elements, which is a per-chunk `dw` of the plain `d`
// epilogue: the low 4 bits from a `scales_l` nibble, the high 2 from
// `scales_h` bits `2j`, then the `-32` bias. `scales_h` carries the high bits
// of ALL EIGHT sub-blocks, so it is read as one 16-bit field, never as a byte.
//
// The 136-byte block stride is a multiple of 8, so `o` is a multiple of 4 and
// every quant word is read whole.
struct Gemv1IQ4XS {
    static constexpr unsigned int BLOCK_BYTES = 136;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");
    // `read` returns the chunk's words in element order, as signed bytes.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_D;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ float scale(
        const int* __restrict__ row, unsigned int o, unsigned int j
    ) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(row) + o;
        const unsigned int sh = gemv1_row_half(row, o + 2);
        const unsigned int sl = b[4 + j / 2];
        const int ls =
            (int)(((sl >> (4 * (j % 2))) & 0x0Fu) | (((sh >> (2 * j)) & 0x03u) << 4));
        return gemv1_half_bits_to_float(gemv1_row_half(row, o)) * (float)(ls - 32);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const int* run = row + (o + 8 + 16 * j) / 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            gguf_iq4_table_lookup(run[i], &w[i], &w[i + 4]);
        }
    }
};

// Q4_1: f16 `d`@0, f16 `m`@2, 16 nibble bytes@4 per 32-element block. The
// value is `d * q + m` with UNSIGNED 4-bit `q`, so the lanes stay raw 0..15
// and the `(d, +m)` pair goes into the `dm` epilogue. That epilogue always
// ADDS its second term, so the stored minimum is `+m`, not `-m`: Q4_1 is the
// additive form, the K-quants are the subtractive one.
//
// The 20-byte block stride is a multiple of 4, so `o` is a multiple of 4,
// every quant word is read whole and the adjacent `d`/`m` pair is one aligned
// `half2` load.
struct Gemv1Q41 {
    static constexpr unsigned int BLOCK_BYTES = 20;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");
    // `read` returns the chunk's words in element order, as unsigned nibbles.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_DM;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* scw, float* minw
    ) {
        const half2 dm = *reinterpret_cast<const half2*>(
            reinterpret_cast<const char*>(row) + o + j * BLOCK_BYTES
        );
        *scw = __low2float(dm);
        *minw = __high2float(dm);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const int* run = row + (o + j * BLOCK_BYTES + 4) / 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = run[i] & 0x0F0F0F0F;
            w[i + 4] = (run[i] >> 4) & 0x0F0F0F0F;
        }
    }
};

// Q5_1: Q4_1 with a fifth quant bit. f16 `d`@0, f16 `m`@2, 32-bit `qh`@4, 16
// nibble bytes@8. The value is `d * q + m` over the unsigned 5-bit assembly,
// so the lanes stay raw 0..31 and the `(d, +m)` pair is Q4_1's. The 24-byte
// block stride is a multiple of 8, so every word is read whole.
struct Gemv1Q51 {
    static constexpr unsigned int BLOCK_BYTES = 24;
    static constexpr unsigned int CHUNKS = 1;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");
    // `read` returns the chunk's words in element order, as unsigned 5-bit
    // values.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_DM;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* scw, float* minw
    ) {
        const half2 dm = *reinterpret_cast<const half2*>(
            reinterpret_cast<const char*>(row) + o + j * BLOCK_BYTES
        );
        *scw = __low2float(dm);
        *minw = __high2float(dm);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        // `qh`@4 and `qs`@8 are contiguous: word 0 is `qh`, words 1..4 are
        // the nibble run.
        const int* run = row + (o + j * BLOCK_BYTES + 4) / 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qh = (int)((unsigned int)run[0] >> (4 * i));
            int lo = run[i + 1] & 0x0F0F0F0F;
            int hi = (run[i + 1] >> 4) & 0x0F0F0F0F;
            gemv1_q5_fifth_bits(qh, &lo, &hi);
            w[i] = lo;
            w[i + 4] = hi;
        }
    }
};
