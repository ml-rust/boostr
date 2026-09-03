// GGUF K-quant scale/min decoders — the ONE place these bit layouts are
// written. Every kernel file that needs Q3_K/Q4_K/Q5_K scales includes this
// instead of restating the packing, so a layout fix lands once.
//
// Format decode only: no GEMV/GEMM tuning constants, no block-size defines,
// no activation helpers. Self-contained — no dependency on WARP_SIZE,
// load_int_ua, or dp4a from the per-kernel-family common.cuh files, since
// quant_gemv.cu and quant_matmul.cu define their own copies of those and a
// second definition here would conflict.

#pragma once

// ── Q4_K / Q5_K scale unpacking ────────────────────────────────────────
// Shared by Q4_K and Q5_K: 12-byte packed 6-bit scales and mins → 8 each

static __device__ __forceinline__ void unpack_q4k_q5k_scales(
    const unsigned char* sc,
    unsigned char* scales,
    unsigned char* mins
) {
    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i] = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
    }
}

// ── Selective scale unpack via 16-bit loads (MWR optimized) ─────────────
// Used by Q4_K and Q5_K MWR kernels. Given chunk index (j_lo/2),
// returns the scale/min pair for j_lo and j_hi sub-blocks.

static __device__ __forceinline__ void unpack_scales_mwr(
    const unsigned char* sc,
    int j_lo,
    unsigned char* scale_lo, unsigned char* scale_hi,
    unsigned char* min_lo, unsigned char* min_hi
) {
    const unsigned short* sc16 = (const unsigned short*)sc;
    const int j = j_lo / 2;
    if (j < 2) {
        unsigned short s0 = sc16[j] & 0x3F3F;
        unsigned short s1 = sc16[j + 2] & 0x3F3F;
        *scale_lo = (unsigned char)(s0);
        *scale_hi = (unsigned char)(s0 >> 8);
        *min_lo = (unsigned char)(s1);
        *min_hi = (unsigned char)(s1 >> 8);
    } else {
        unsigned short s0 = ((sc16[j + 2]) & 0x0F0F) | ((sc16[j - 2] & 0xC0C0) >> 2);
        unsigned short s1 = ((sc16[j + 2] >> 4) & 0x0F0F) | ((sc16[j] & 0xC0C0) >> 2);
        *scale_lo = (unsigned char)(s0);
        *scale_hi = (unsigned char)(s0 >> 8);
        *min_lo = (unsigned char)(s1);
        *min_hi = (unsigned char)(s1 >> 8);
    }
}

// ── Q3_K scale unpacking ────────────────────────────────────────────────
// 12 bytes → 16 signed 6-bit scales

static __device__ __forceinline__ void unpack_q3k_scales(
    const unsigned char* sc_raw,
    signed char* scales
) {
    unsigned int aux[4];
    unsigned char aux_bytes[12];
    for (int i = 0; i < 12; i++) aux_bytes[i] = sc_raw[i];
    memcpy(&aux[0], aux_bytes, 4);
    memcpy(&aux[1], aux_bytes + 4, 4);
    memcpy(&aux[2], aux_bytes + 8, 4);

    unsigned int tmp = aux[2];
    const unsigned int KMASK1 = 0x03030303u;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    unsigned int a0 = aux[0], a1 = aux[1];
    aux[0] = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    aux[2] = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    memcpy(&scales[0],  &aux[0], 4);
    memcpy(&scales[4],  &aux[1], 4);
    memcpy(&scales[8],  &aux[2], 4);
    memcpy(&scales[12], &aux[3], 4);
    for (int i = 0; i < 16; i++)
        scales[i] = (signed char)((unsigned char)scales[i] - 32);
}

// ── Q4_K single sub-block scale/min ──────────────────────────────────────
// Same 12-byte layout as unpack_q4k_q5k_scales, decoded for one sub-block
// `j` (0..7) instead of all eight — used by the monolithic MMQ/tiled-GEMM
// kernels that dequantize one sub-block at a time rather than a whole
// super-block up front.

static __device__ __forceinline__ void q4k_scale_min(
    const unsigned char* sc, int j, int* scale, int* minimum
) {
    if (j < 4) {
        *scale = sc[j] & 63;
        *minimum = sc[j + 4] & 63;
    } else {
        *scale = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *minimum = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

// ── Split-half nibble order (4-bit formats over 32-element blocks) ───────
// llama.cpp `dequantize_row_q4_0` packs element `j` and element `j + 16` of a
// block into ONE byte:
//
//     y[i*qk + j + 0    ] = (qs[j] & 0x0F) * d;   // first half
//     y[i*qk + j + qk/2 ] = (qs[j] >>   4) * d;   // second half
//
// The two nibbles of a byte are 16 elements apart in the output, NOT adjacent.
// Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and IQ4_XS all use it. Emitting them to
// consecutive positions permutes every weight inside the block and nothing
// errors: shape, block count and tensor RMS all stay correct. Q5_0/Q5_1 index
// their fifth-bit word `qh` with the SAME element index — bit `j` for the
// first half, bit `j + 16` for the second.
//
// `elem` is the element index within the 32-element block.
static __device__ __forceinline__ int gguf_split_half_nibble(
    const unsigned char* qs, int elem
) {
    const unsigned char byte = qs[elem & 15];
    return (elem & 16) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
}
