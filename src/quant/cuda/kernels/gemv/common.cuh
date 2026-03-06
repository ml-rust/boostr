// Shared helpers for quantized GEMV kernels
//
// Included by all per-format GEMV .cu files. Contains:
//   - dp4a intrinsic (with fallback for pre-Pascal)
//   - Unaligned loads
//   - MWR constants
//   - Scale unpacking helpers
//   - SiLU activation
//   - In-register F32→Q8_1 quantization helpers

#pragma once

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE_256 (WARP_SIZE * WARPS_PER_BLOCK)
#define NWARPS_K 4

// ── dp4a intrinsic ──────────────────────────────────────────────────────
// 4-element int8 dot product in a single instruction (compute >= 6.1)

static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// ── SiLU activation ─────────────────────────────────────────────────────

static __device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

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

// ── MWR reduction template ──────────────────────────────────────────────
// Used by all MWR kernels: shared memory reduction across NWARPS_K warps

static __device__ __forceinline__ float mwr_reduce(
    float acc, int warp_id, int lane_id,
    float smem[NWARPS_K][WARP_SIZE]
) {
    smem[warp_id][lane_id] = acc;
    __syncthreads();

    if (warp_id != 0) return 0.0f;

    float sum = smem[0][lane_id];
    #pragma unroll
    for (int w = 1; w < NWARPS_K; w++)
        sum += smem[w][lane_id];

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    return sum;
}

// Dual-accumulator version for fused SwiGLU

static __device__ __forceinline__ void mwr_reduce_dual(
    float gate_acc, float up_acc, int warp_id, int lane_id,
    float smem[2][NWARPS_K][WARP_SIZE],
    float* gate_out, float* up_out
) {
    smem[0][warp_id][lane_id] = gate_acc;
    smem[1][warp_id][lane_id] = up_acc;
    __syncthreads();

    *gate_out = 0.0f;
    *up_out = 0.0f;
    if (warp_id != 0) return;

    float gate_sum = smem[0][0][lane_id];
    float up_sum = smem[1][0][lane_id];
    #pragma unroll
    for (int w = 1; w < NWARPS_K; w++) {
        gate_sum += smem[0][w][lane_id];
        up_sum += smem[1][w][lane_id];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
        up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
    }

    *gate_out = gate_sum;
    *up_out = up_sum;
}

// ── Warp reduction helper ───────────────────────────────────────────────

static __device__ __forceinline__ float warp_reduce_sum(float acc) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    return acc;
}
