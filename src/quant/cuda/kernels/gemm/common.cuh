// Shared helpers for quantized GEMM kernels (tiled matmul, M > 64)
//
// All GEMM kernels follow the same structure:
//   Grid: (N/16, M/16, 1), Block: (16, 16, 1)
//   Each thread computes one output element [row, col]
//   Iterates over K blocks, dequantizes on-the-fly, accumulates float sum

#pragma once

#include <cuda_fp16.h>

#include "../decode.cuh"

// ── Safe unaligned load helpers ─────────────────────────────────────────

static __device__ __forceinline__ float load_f16_as_f32_gemm(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

// ── Q4_K / Q5_K scale unpacking (shared) ────────────────────────────────
// Thin forwarding wrapper: keeps the GEMM-side name stable for callers
// while the packing itself lives once in decode.cuh.

static __device__ __forceinline__ void unpack_q4k_q5k_scales_gemm(
    const unsigned char* sc,
    unsigned char* scales,
    unsigned char* mins
) {
    unpack_q4k_q5k_scales(sc, scales, mins);
}

// ── Q3_K scale unpacking ────────────────────────────────────────────────
// Thin forwarding wrapper, see unpack_q4k_q5k_scales_gemm above.

static __device__ __forceinline__ void unpack_q3k_scales_gemm(
    const unsigned char* sc_raw,
    signed char* scales
) {
    unpack_q3k_scales(sc_raw, scales);
}
