// Shared helpers for quantized GEMM kernels (tiled matmul, M > 64)
//
// All GEMM kernels follow the same structure:
//   Grid: (N/16, M/16, 1), Block: (16, 16, 1)
//   Each thread computes one output element [row, col]
//   Iterates over K blocks, dequantizes on-the-fly, accumulates float sum

#pragma once

#include <cuda_fp16.h>

// ── Safe unaligned load helpers ─────────────────────────────────────────

static __device__ __forceinline__ float load_f16_as_f32_gemm(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

// ── Q4_K / Q5_K scale unpacking (shared) ────────────────────────────────

static __device__ __forceinline__ void unpack_q4k_q5k_scales_gemm(
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

// ── Q3_K scale unpacking ────────────────────────────────────────────────

static __device__ __forceinline__ void unpack_q3k_scales_gemm(
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
