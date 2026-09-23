// K-quant weight readers of the single-token (M = 1) MMQ body: Q4_K, Q5_K,
// Q6_K, Q3_K and Q2_K. Same contract as the readers in `gemv1_formats.cuh`,
// plus `scales(row, o, j, ln, ...)`, which gives the float constants the
// chunk's epilogue multiplies. All five hold 256 elements per super-block, so
// `CHUNKS` is 8 and one step is exactly one super-block per feature.
//
// A reader returns RAW quant lanes and leaves every sub-scale to the float
// epilogue, as the tensor-core staging does: Q4_K and Q5_K unsigned 4- and
// 5-bit lanes with `(d * scale, -dmin * minimum)` per 32 elements
// (`GEMV1_EPI_DM`), Q6_K signed 6-bit and Q3_K signed 3-bit lanes with
// `d * scale` per 16 elements (`GEMV1_EPI_SC16`), Q2_K unsigned 2-bit lanes
// with both a scale and a minimum per 16 elements (`GEMV1_EPI_SC16_MIN`).
//
// WORD MAPS. Chunk `j` of a super-block is staged words `8j .. 8j + 7`, and
// staged word `w` holds elements `4w .. 4w + 3`, so word `i` of a chunk is
// elements `32j + 4i .. +3`. Each map below is the inverse of the
// corresponding `stage` in `quant_mmq_mma.cu`, which is where the map is
// derived and cross-checked against the CPU dequantizers.
#pragma once

#include <cuda_fp16.h>

#include "../decode.cuh"
#include "gemv1_formats.cuh"

// `(d * scale_j, -dmin * minimum_j)` of sub-block `j` of the super-block at
// byte `blk` of a staged row, the pair `MmqfQ4K::store_scale` writes. The
// header is the same for Q4_K and Q5_K: `d` and `dmin` as adjacent `half` at
// byte 0, then 12 packed 6-bit scale/minimum bytes at byte 4. For `j < 4`
// the third byte is unused; re-reading `sc[j]` keeps the index in range
// without a branch around the load.
static __device__ __forceinline__ void gemv1_q4k_dm(
    const int* __restrict__ row, unsigned int blk, unsigned int j, float* scw, float* minw
) {
    const unsigned char* sc = reinterpret_cast<const unsigned char*>(row) + blk + 4;
    int scale;
    int minimum;
    q4k_scale_min_bytes(sc[j], sc[j + 4], sc[j < 4 ? j : j - 4], (int)j, &scale, &minimum);
    const half2 dm = *reinterpret_cast<const half2*>(reinterpret_cast<const char*>(row) + blk);
    const float d = __low2float(dm);
    const float dmin = __high2float(dm);
    *scw = d * (float)scale;
    *minw = -dmin * (float)minimum;
}

// Q4_K: `half` d and dmin at byte 0, 12 packed 6-bit scale/minimum bytes at
// byte 4, then 128 nibble-packed quants at byte 16 as four 32-byte runs.
//
// Word map. A run holds a sub-block PAIR: the even sub-block in the low
// nibbles and the odd one in the high nibbles of the same bytes. Chunk `j`
// is sub-block `j`, so it reads the 8 consecutive words at `16 + 32 * (j /
// 2)` and takes the low nibbles for even `j`, the high nibbles for odd `j`.
// The row stride is a multiple of 16, so a span starts at the top of its
// window and `o` is 0; whole-word reads are legal for any `o` a multiple
// of 4.
struct Gemv1Q4K {
    static constexpr unsigned int BLOCK_BYTES = 144;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 16 == 0, "A 16-byte block stride keeps the span word aligned.");
    // `read` returns the chunk's words in element order, as unsigned nibbles.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_DM;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* scw, float* minw
    ) {
        gemv1_q4k_dm(row, o, j, scw, minw);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const int* run = row + (o + 16 + 32 * (j / 2)) / 4;
        const unsigned int sh = 4 * (j & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = (run[i] >> sh) & 0x0F0F0F0F;
        }
    }
};

// Q5_K: Q4_K's header, then 32 bytes of fifth bits `qh` at byte 16 and 128
// nibble-packed quants at byte 48.
//
// Word map. The nibble run is Q4_K's, shifted to byte 48. In `qh` the BYTE
// index is the element within a sub-block and the BIT index is the
// sub-block number, so word `i` of chunk `j` takes bit `j` of each byte of
// the `qh` word at byte `16 + 4 * i` — both for the even sub-block's low
// nibbles and for the odd one's high nibbles, since the bit index is the
// sub-block index either way. The 8 `qh` words do not depend on `j`, so a
// step's two chunks per lane read the same ones.
struct Gemv1Q5K {
    static constexpr unsigned int BLOCK_BYTES = 176;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 16 == 0, "A 16-byte block stride keeps the span word aligned.");
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
        gemv1_q4k_dm(row, o, j, scw, minw);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const int* run = row + (o + 48 + 32 * (j / 2)) / 4;
        const int* qh = row + (o + 16) / 4;
        const unsigned int sh = 4 * (j & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = ((run[i] >> sh) & 0x0F0F0F0F) | (((qh[i] >> j) & 0x01010101) << 4);
        }
    }
};

// Q6_K: 128 bytes `ql` at byte 0, 64 bytes `qh` at byte 128, 16 signed
// `scales` at byte 192 and the `half` `d` at byte 208.
//
// Word map. A chunk's two halves take different scales, so the reader gives
// the words in element order and the epilogue splits them 4 + 4. With
// `hh = j / 4` the 128-element half and `hi = (j % 4) >= 2`:
//
//   `ql` words   `16 * hh + 8 * (j % 2) + i`, low nibbles when `hi` is
//                false and high nibbles when it is true
//   `qh` words   `8 * hh + i`, bit pair at `2 * (j & 1) + 4 * hi`
//
// Both runs are 8 consecutive words, so each is one funnel-shifted walk.
// The 210-byte block stride is only 2-byte aligned, so `o` is even and
// every word read goes through `gemv1_row_words`. The `-32` bias turns the
// unsigned 0..63 value into the signed one the int dot takes.
struct Gemv1Q6K {
    static constexpr unsigned int BLOCK_BYTES = 210;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 2 == 0, "Funnel reads need a 2-byte block stride.");
    // `read` returns the chunk's words in element order, as signed 6-bit
    // values.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_SC16;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    // `d * scale` of the chunk's two 16-element groups, `2j` and `2j + 1`,
    // the value `MmqfQ6K::store_scale` writes.
    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* sc_lo,
        float* sc_hi
    ) {
        const signed char* sc = reinterpret_cast<const signed char*>(row) + o + 192;
        const float d_h = gemv1_half_bits_to_float(gemv1_row_half(row, o + 208));
        *sc_lo = d_h * (float)(int)sc[2 * j];
        *sc_hi = d_h * (float)(int)sc[2 * j + 1];
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const unsigned int hh = j / 4;
        const unsigned int nib = (j % 4) >= 2 ? 4u : 0u;
        const unsigned int sh = 2 * (j & 1) + nib;
        int vl[8];
        int vh[8];
        gemv1_row_words<8>(row, o + 64 * hh + 32 * (j & 1), vl);
        gemv1_row_words<8>(row, o + 128 + 32 * hh, vh);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = ((vl[i] >> nib) & 0x0F0F0F0F) | (((vh[i] >> sh) & 0x03030303) << 4);
            w[i] = __vsubss4(v, 0x20202020);
        }
    }
};

// Q3_K: 32 bytes `hmask`@0, 64 bytes `qs`@32, 12 packed 6-bit `scales`@96 and
// the f16 `d`@108. Sixteen signed scales, one per 16 elements, and no minimum
// term, so the epilogue is Q6_K's.
//
// Word map. Staged word `w` takes `qs` word `8 * (w / 32) + w % 8` with the
// 2-bit field at shift `2 * ((w % 32) / 8)`, and `hmask` word `w % 8` at bit
// `(w % 32) / 8 + 4 * (w / 32)`. For chunk `j` (words `8j .. 8j + 7`) that
// collapses to: `qs` words `8 * (j / 4) + i`, one shift `2 * (j % 4)` for the
// whole chunk, and `hmask` words `0 .. 7` at bit `j`. The eight `hmask` words
// do not depend on `j`, so a step's two chunks per lane read the same ones.
//
// THE QUANT. Two low bits from `qs`, one high bit from `hmask`, and the high
// bit is INVERTED: a SET bit means do NOT subtract 4. Moving it to bit 2
// contributes `+4`, so the `-4` bias that follows cancels on a set bit and
// leaves `-4` on a clear one, which is the signed value in [-4, 3].
//
// The 110-byte block stride is only 2-byte aligned, so `o` is even and every
// word read goes through `gemv1_row_words`.
struct Gemv1Q3K {
    static constexpr unsigned int BLOCK_BYTES = 110;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 2 == 0, "Funnel reads need a 2-byte block stride.");
    // `read` returns the chunk's words in element order, as signed values in
    // [-4, 3].
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_SC16;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    // `d * scale` of the chunk's two 16-element groups, `2j` and `2j + 1`,
    // the value `MmqfQ3K::stage` writes.
    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* sc_lo,
        float* sc_hi
    ) {
        const unsigned char* sc = reinterpret_cast<const unsigned char*>(row) + o + 96;
        const float d_h = gemv1_half_bits_to_float(gemv1_row_half(row, o + 108));
        const int g_lo = (int)(2 * j);
        const int g_hi = g_lo + 1;
        *sc_lo = d_h * (float)q3k_scale_bytes(
                           sc[GGUF_Q3K_SC_LOW_BYTE(g_lo)], sc[GGUF_Q3K_SC_HIGH_BYTE(g_lo)], g_lo
                       );
        *sc_hi = d_h * (float)q3k_scale_bytes(
                           sc[GGUF_Q3K_SC_LOW_BYTE(g_hi)], sc[GGUF_Q3K_SC_HIGH_BYTE(g_hi)], g_hi
                       );
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        int vq[8];
        int vh[8];
        gemv1_row_words<8>(row, o + 32 + 32 * (j / 4), vq);
        gemv1_row_words<8>(row, o, vh);
        const unsigned int sh = 2 * (j % 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = ((vq[i] >> sh) & 0x03030303) | (((vh[i] >> j) << 2) & 0x04040404);
            w[i] = __vsubss4(v, 0x04040404);
        }
    }
};

// Q2_K: 16 packed scale/min bytes@0, 64 bytes `qs`@16, f16 `d`@80 and f16
// `dmin`@82. One scale byte per 16 elements carries both fields — low nibble
// the scale, high nibble the minimum — so the epilogue is the only one in the
// family with BOTH a per-16 scale and a minimum term.
//
// Word map. Q3_K's without the `hmask` term: staged word `w` takes `qs` word
// `8 * (w / 32) + w % 8` at shift `2 * ((w % 32) / 8)`, so chunk `j` reads
// `qs` words `8 * (j / 4) + i` at one shift `2 * (j % 4)`.
//
// THE QUANT. Two bits, UNSIGNED, in 0..3. No bias is folded in; the format's
// asymmetry is carried entirely by the minimum term.
//
// The 84-byte block stride is a multiple of 4, so `o` is a multiple of 4,
// every quant word is read whole and the adjacent `d`/`dmin` pair is one
// aligned `half2` load.
struct Gemv1Q2K {
    static constexpr unsigned int BLOCK_BYTES = 84;
    static constexpr unsigned int CHUNKS = 8;
    static constexpr unsigned int SPAN = (GEMV1_STEP_CHUNKS / CHUNKS) * BLOCK_BYTES;
    static_assert(BLOCK_BYTES % 4 == 0, "Whole-word group reads need a 4-byte block stride.");
    // `read` returns the chunk's words in element order, as unsigned 2-bit
    // values.
    static constexpr bool PAIRED_WORDS = false;
    static constexpr bool CODES_PLUS_ONE = false;
    static constexpr Gemv1Epilogue EPILOGUE = GEMV1_EPI_SC16_MIN;
    using Lane = Gemv1NoLane;
    static __device__ __forceinline__ Lane setup(unsigned int) { return Lane{}; }

    // `(d * scale, -dmin * minimum)` of the chunk's two 16-element groups,
    // `2j` and `2j + 1`, the pairs `MmqfQ2K::stage` writes.
    static __device__ __forceinline__ void scales(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, float* sc_lo,
        float* mn_lo, float* sc_hi, float* mn_hi
    ) {
        const unsigned char* sc = reinterpret_cast<const unsigned char*>(row) + o;
        const half2 dm =
            *reinterpret_cast<const half2*>(reinterpret_cast<const char*>(row) + o + 80);
        const float d = __low2float(dm);
        const float dmin = __high2float(dm);
        const unsigned int s_lo = sc[2 * j];
        const unsigned int s_hi = sc[2 * j + 1];
        *sc_lo = d * (float)(s_lo & 0x0Fu);
        *mn_lo = -dmin * (float)(s_lo >> 4);
        *sc_hi = d * (float)(s_hi & 0x0Fu);
        *mn_hi = -dmin * (float)(s_hi >> 4);
    }

    static __device__ __forceinline__ void read(
        const int* __restrict__ row, unsigned int o, unsigned int j, Lane, int (&w)[8]
    ) {
        const int* run = row + (o + 16 + 32 * (j / 4)) / 4;
        const unsigned int sh = 2 * (j % 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = (run[i] >> sh) & 0x03030303;
        }
    }
};
