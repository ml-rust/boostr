// K-quant weight readers of the single-token (M = 1) MMQ body: Q4_K, Q5_K
// and Q6_K. Same contract as the readers in `gemv1_formats.cuh`, plus
// `scales(row, o, j, ln, ...)`, which gives the float constants the chunk's
// epilogue multiplies. The three formats hold 256 elements per super-block,
// so `CHUNKS` is 8 and one step is exactly one super-block per feature.
//
// A reader returns RAW quant lanes and leaves every sub-scale to the float
// epilogue, as the tensor-core staging does: Q4_K and Q5_K unsigned 4- and
// 5-bit lanes with `(d * scale, -dmin * minimum)` per 32 elements
// (`GEMV1_EPI_DM`), Q6_K signed 6-bit lanes with `d * scale` per 16
// elements (`GEMV1_EPI_SC16`).
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
