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

#include "../decode.cuh"

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

// Q4_K/Q5_K/Q3_K scale unpacking (unpack_q4k_q5k_scales, unpack_scales_mwr,
// unpack_q3k_scales) now lives in decode.cuh, included above.

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
