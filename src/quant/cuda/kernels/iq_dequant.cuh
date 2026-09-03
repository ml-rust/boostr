// GGUF IQ block decoders — the ONE place these layouts are written for CUDA.
//
// Every CUDA kernel that decodes an IQ block includes this instead of restating
// the layout. `src/quant/cpu/kernels/dequant_iq{1,2,3}.rs` is the CPU mirror;
// keep the two in step. `tests/gguf_conformance_llama_cpp.rs` gates both
// against llama.cpp.
//
// All IQ formats are CODEBOOK quantizations: `qs` holds INDICES into a grid of
// precomputed points, never the magnitudes themselves. A decoder that reads
// `qs` as packed 2- or 3-bit magnitudes still yields finite, plausibly scaled
// numbers, so a mistake here shows up as wrong output, never as an error.

#pragma once

#include <cuda_fp16.h>

#include "iq_grid.cuh"

// The offset added to every IQ1 grid component before scaling.
#define IQ1_DELTA 0.125f

static __device__ __forceinline__ float iq_load_d(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

static __device__ __forceinline__ unsigned int iq_load_u32(const unsigned char* p) {
    unsigned int v;
    memcpy(&v, p, sizeof(unsigned int));
    return v;
}

static __device__ __forceinline__ unsigned short iq_load_u16(const unsigned char* p) {
    unsigned short v;
    memcpy(&v, p, sizeof(unsigned short));
    return v;
}

// Component `pos` (0..8) of an 8-component grid point, as a magnitude.
static __device__ __forceinline__ float iq_grid8(unsigned long long point, int pos) {
    return (float)((unsigned char)(point >> (8 * pos)));
}

// Component `pos` (0..8) of an 8-component grid point, as a SIGNED value.
static __device__ __forceinline__ float iq_grid8_signed(unsigned long long point, int pos) {
    return (float)((signed char)(unsigned char)(point >> (8 * pos)));
}

// Component `pos` (0..4) of a 4-component grid point, as a magnitude.
static __device__ __forceinline__ float iq_grid4(unsigned int point, int pos) {
    return (float)((unsigned char)(point >> (8 * pos)));
}

// +1.0f or -1.0f for bit `pos` of a sign byte: a set bit negates.
static __device__ __forceinline__ float iq_sign(unsigned char sign_byte, int pos) {
    return ((sign_byte >> pos) & 1) ? -1.0f : 1.0f;
}

// The 4-bit scale for grid entry `entry`, packed two per byte.
static __device__ __forceinline__ int iq_packed_scale(const unsigned char* scales, int entry) {
    const int k = entry / 2;
    return (scales[k / 2] >> (4 * (k % 2))) & 0x0F;
}

// ── IQ2_XXS: d:f16(2) + qs[64] ─────────────────────────────────────────
// Eight pairs of u32: the first holds four 8-bit grid indices, the second a
// 4-bit scale in its top nibble over four 7-bit sign-table indices.
static __device__ void iq2_xxs_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs = block + 2;
    for (int group = 0; group < 8; group++) {
        const unsigned int indices = iq_load_u32(qs + group * 8);
        const unsigned int aux     = iq_load_u32(qs + group * 8 + 4);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        for (int sub = 0; sub < 4; sub++) {
            const unsigned long long point = IQ2XXS_GRID[(indices >> (8 * sub)) & 0xFF];
            const unsigned char signs = KSIGNS[(aux >> (7 * sub)) & 0x7F];
            for (int j = 0; j < 8; j++)
                out[group * 32 + sub * 8 + j] = db * iq_grid8(point, j) * iq_sign(signs, j);
        }
    }
}

// ── IQ2_XS: d:f16(2) + qs[64] + scales[8] ──────────────────────────────
// 32 u16, each a 9-bit grid index under a 7-bit sign-table index.
static __device__ void iq2_xs_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs = block + 2;
    const unsigned char* scales = block + 66;
    for (int entry = 0; entry < 32; entry++) {
        const unsigned short q = iq_load_u16(qs + entry * 2);
        const unsigned long long point = IQ2XS_GRID[q & 511];
        const unsigned char signs = KSIGNS[q >> 9];
        const float db = d * (0.5f + (float)iq_packed_scale(scales, entry)) * 0.25f;
        for (int j = 0; j < 8; j++)
            out[entry * 8 + j] = db * iq_grid8(point, j) * iq_sign(signs, j);
    }
}

// ── IQ2_S: d:f16(2) + qs[32] + signs[32] + qh[8] + scales[8] ───────────
// Two extra index bits per entry from qh; explicit sign bits, no sign table.
static __device__ void iq2_s_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs     = block + 2;
    const unsigned char* signs  = block + 34;
    const unsigned char* qh     = block + 66;
    const unsigned char* scales = block + 74;
    for (int entry = 0; entry < 32; entry++) {
        const int high = (qh[entry / 4] >> (2 * (entry % 4))) & 0x03;
        const unsigned long long point = IQ2S_GRID[qs[entry] | (high << 8)];
        const float db = d * (0.5f + (float)iq_packed_scale(scales, entry)) * 0.25f;
        for (int j = 0; j < 8; j++)
            out[entry * 8 + j] = db * iq_grid8(point, j) * iq_sign(signs[entry], j);
    }
}

// ── IQ3_XXS: d:f16(2) + qs[64] + scales[32] ────────────────────────────
// Eight u32 scale words; each qs byte is a 4-component point, so two bytes
// cover one 8-element sign sub-group.
static __device__ void iq3_xxs_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs = block + 2;
    const unsigned char* scales = block + 66;
    for (int group = 0; group < 8; group++) {
        const unsigned int aux = iq_load_u32(scales + group * 4);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        for (int sub = 0; sub < 4; sub++) {
            const unsigned char signs = KSIGNS[(aux >> (7 * sub)) & 0x7F];
            const unsigned int lo = IQ3XXS_GRID[qs[group * 8 + sub * 2]];
            const unsigned int hi = IQ3XXS_GRID[qs[group * 8 + sub * 2 + 1]];
            for (int j = 0; j < 8; j++) {
                const float mag = (j < 4) ? iq_grid4(lo, j) : iq_grid4(hi, j - 4);
                out[group * 32 + sub * 8 + j] = db * mag * iq_sign(signs, j);
            }
        }
    }
}

// ── IQ3_S: d:f16(2) + qs[64] + qh[8] + signs[32] + scales[4] ───────────
// One extra index bit per qs byte from qh; explicit sign bits.
static __device__ void iq3_s_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs     = block + 2;
    const unsigned char* qh     = block + 66;
    const unsigned char* signs  = block + 74;
    const unsigned char* scales = block + 106;
    for (int e = 0; e < 256; e++) {
        const int entry = e / 4;
        const int ninth = (qh[entry / 8] >> (entry % 8)) & 1;
        const unsigned int point = IQ3S_GRID[qs[entry] | (ninth << 8)];
        const int scale = (scales[(e / 32) / 2] >> (4 * ((e / 32) % 2))) & 0x0F;
        const float db = d * (1.0f + 2.0f * (float)scale);
        out[e] = db * iq_grid4(point, e % 4) * iq_sign(signs[e / 8], e % 8);
    }
}

// ── IQ1_S: d:f16(2) + qs[32] + qh[16] ──────────────────────────────────
// One u16 per 32-element group: four 3-bit index-high fields, a 3-bit scale at
// bits 12..14, and the group's delta sign in bit 15.
static __device__ void iq1_s_dequant_block(const unsigned char* block, float* out) {
    const float d = iq_load_d(block);
    const unsigned char* qs = block + 2;
    const unsigned char* qh = block + 34;
    for (int group = 0; group < 8; group++) {
        const unsigned short h = iq_load_u16(qh + group * 2);
        const float dl = d * (2.0f * (float)((h >> 12) & 7) + 1.0f);
        const float delta = (h & 0x8000) == 0 ? IQ1_DELTA : -IQ1_DELTA;
        for (int sub = 0; sub < 4; sub++) {
            const unsigned long long point =
                IQ1_GRID[qs[group * 4 + sub] | (((h >> (3 * sub)) & 7) << 8)];
            for (int j = 0; j < 8; j++)
                out[group * 32 + sub * 8 + j] = dl * (iq_grid8_signed(point, j) + delta);
        }
    }
}

// ── IQ1_M: qs[32] + qh[16] + scales[8] ─────────────────────────────────
// No leading d field. The f16 scale is split across the top nibbles of the four
// u16 in scales, low nibble first; those u16 also carry sixteen 3-bit
// sub-scales. Each qh nibble gives one entry's index-high bits and delta sign.
static __device__ void iq1_m_dequant_block(const unsigned char* block, float* out) {
    const unsigned char* qs     = block;
    const unsigned char* qh     = block + 32;
    const unsigned char* scales = block + 48;

    unsigned short sc[4];
    for (int i = 0; i < 4; i++) sc[i] = iq_load_u16(scales + i * 2);
    const unsigned short d_bits = (unsigned short)(((sc[0] & 0xF000) >> 12)
                                                 | ((sc[1] & 0xF000) >> 8)
                                                 | ((sc[2] & 0xF000) >> 4)
                                                 |  (sc[3] & 0xF000));
    __half dh;
    memcpy(&dh, &d_bits, sizeof(__half));
    const float d = __half2float(dh);

    for (int entry = 0; entry < 32; entry++) {
        const int nibble = (qh[entry / 2] >> (4 * (entry % 2))) & 0x0F;
        const unsigned long long point = IQ1_GRID[qs[entry] | ((nibble & 7) << 8)];
        const float delta = (nibble & 8) == 0 ? IQ1_DELTA : -IQ1_DELTA;
        for (int j = 0; j < 8; j++) {
            const int e = entry * 8 + j;
            // One 3-bit sub-scale covers 16 elements, i.e. two grid entries.
            const int k = e / 16;
            const int scale = (sc[k / 4] >> (3 * (k % 4))) & 0x07;
            const float dl = d * (2.0f * (float)scale + 1.0f);
            out[e] = dl * (iq_grid8_signed(point, j) + delta);
        }
    }
}
