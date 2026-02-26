// Calibration kernels for quantization (AWQ, Fisher Information, GPTQ)
//
// AWQ:    act_scale (per-channel max-abs) + score_reduce (weighted mean)
// Fisher: squared gradient accumulation + normalize
// GPTQ:   column-wise quantization step (F32 only)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// AWQ: Per-channel max-abs activation scale — F32
// activations: [N, K], output: [K] (max_n |act[n, j]|)
// Each thread handles one element, atomicMax on output.
// ============================================================================

extern "C" __global__ void awq_act_scale_f32(
    const float* __restrict__ act,
    float* __restrict__ out,
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int j = idx % K;
    const float val = fabsf(act[idx]);

    // atomicMax for float via int reinterpretation (works for non-negative values)
    int* out_int = (int*)out;
    int val_int = __float_as_int(val);
    atomicMax(out_int + j, val_int);
}

// ============================================================================
// AWQ: Score reduction — F32
// weights: [M, K], act_scale: [K], output: [K]
// score[j] = mean_i(act_scale[j] * |W[i, j]|)
// Uses atomicAdd for accumulation, then normalize by M.
// ============================================================================

extern "C" __global__ void awq_score_reduce_f32(
    const float* __restrict__ weights,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int M,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    const int j = idx % K;
    const float val = act_scale[j] * fabsf(weights[idx]);
    atomicAdd(out + j, val / (float)M);
}

// ============================================================================
// Fisher: Squared gradient accumulation — F32
// gradients: [N, P], output: [P]
// fisher[i] = sum_n(grad[n, i]^2) / N
// ============================================================================

extern "C" __global__ void fisher_accumulate_f32(
    const float* __restrict__ grad,
    float* __restrict__ out,
    const int N,
    const int P
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;

    const int p = idx % P;
    const float g = grad[idx];
    atomicAdd(out + p, (g * g) / (float)N);
}

// ============================================================================
// 16-bit atomic helpers (CAS loop — works for both F16 and BF16)
//
// CUDA has no native atomicMax/atomicAdd for __half/__nv_bfloat16 on individual
// elements. The standard approach is a 32-bit CAS loop: read the aligned 32-bit
// word containing our 16-bit slot, compute the new value, and swap atomically.
// The loop retries only on contention (another thread updated the same 32-bit
// word between our read and our CAS).
//
// Address alignment: CUDA guarantees all device allocations are at least 256-byte
// aligned, so individual __half / __nv_bfloat16 elements within a contiguous
// buffer are 2-byte aligned. The `(size_t)addr & 2` test selects the upper or
// lower 16-bit slot within the surrounding aligned 32-bit word.
// ============================================================================

// atomicAdd for a single __half element.
static __device__ __forceinline__ void atomic_add_f16(__half* addr, float addend) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~(size_t)2);
    unsigned int old_word = *base;
    unsigned int assumed;
    const bool hi = (size_t)addr & 2;
    do {
        assumed = old_word;
        unsigned short slot = hi ? (unsigned short)(assumed >> 16)
                                 : (unsigned short)(assumed & 0xffffu);
        float updated = __half2float(__ushort_as_half(slot)) + addend;
        unsigned short new_slot = __half_as_ushort(__float2half(updated));
        unsigned int new_word = hi ? ((assumed & 0x0000ffffu) | ((unsigned int)new_slot << 16))
                                   : ((assumed & 0xffff0000u) | new_slot);
        old_word = atomicCAS(base, assumed, new_word);
    } while (old_word != assumed);
}

// atomicMax for a single __half element (non-negative values only — safe for
// abs-max accumulation used in AWQ act-scale).
static __device__ __forceinline__ void atomic_max_f16(__half* addr, float candidate) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~(size_t)2);
    unsigned int old_word = *base;
    unsigned int assumed;
    const bool hi = (size_t)addr & 2;
    do {
        assumed = old_word;
        unsigned short slot = hi ? (unsigned short)(assumed >> 16)
                                 : (unsigned short)(assumed & 0xffffu);
        float current = __half2float(__ushort_as_half(slot));
        if (candidate <= current) return;  // No update needed — early exit
        unsigned short new_slot = __half_as_ushort(__float2half(candidate));
        unsigned int new_word = hi ? ((assumed & 0x0000ffffu) | ((unsigned int)new_slot << 16))
                                   : ((assumed & 0xffff0000u) | new_slot);
        old_word = atomicCAS(base, assumed, new_word);
    } while (old_word != assumed);
}

// atomicAdd for a single __nv_bfloat16 element.
static __device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* addr, float addend) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~(size_t)2);
    unsigned int old_word = *base;
    unsigned int assumed;
    const bool hi = (size_t)addr & 2;
    do {
        assumed = old_word;
        unsigned short slot = hi ? (unsigned short)(assumed >> 16)
                                 : (unsigned short)(assumed & 0xffffu);
        float updated = __bfloat162float(__ushort_as_bfloat16(slot)) + addend;
        unsigned short new_slot = __bfloat16_as_ushort(__float2bfloat16(updated));
        unsigned int new_word = hi ? ((assumed & 0x0000ffffu) | ((unsigned int)new_slot << 16))
                                   : ((assumed & 0xffff0000u) | new_slot);
        old_word = atomicCAS(base, assumed, new_word);
    } while (old_word != assumed);
}

// atomicMax for a single __nv_bfloat16 element (non-negative values only).
static __device__ __forceinline__ void atomic_max_bf16(__nv_bfloat16* addr, float candidate) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~(size_t)2);
    unsigned int old_word = *base;
    unsigned int assumed;
    const bool hi = (size_t)addr & 2;
    do {
        assumed = old_word;
        unsigned short slot = hi ? (unsigned short)(assumed >> 16)
                                 : (unsigned short)(assumed & 0xffffu);
        float current = __bfloat162float(__ushort_as_bfloat16(slot));
        if (candidate <= current) return;
        unsigned short new_slot = __bfloat16_as_ushort(__float2bfloat16(candidate));
        unsigned int new_word = hi ? ((assumed & 0x0000ffffu) | ((unsigned int)new_slot << 16))
                                   : ((assumed & 0xffff0000u) | new_slot);
        old_word = atomicCAS(base, assumed, new_word);
    } while (old_word != assumed);
}

// ============================================================================
// FP16 variants
// ============================================================================

extern "C" __global__ void awq_act_scale_f16(
    const __half* __restrict__ act,
    __half* __restrict__ out,
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int j = idx % K;
    const float val = fabsf(__half2float(act[idx]));
    atomic_max_f16(out + j, val);
}

extern "C" __global__ void awq_score_reduce_f16(
    const __half* __restrict__ weights,
    const __half* __restrict__ act_scale,
    __half* __restrict__ out,
    const int M,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    const int j = idx % K;
    const float scale = __half2float(act_scale[j]);
    const float val = scale * fabsf(__half2float(weights[idx])) / (float)M;
    atomic_add_f16(out + j, val);
}

extern "C" __global__ void fisher_accumulate_f16(
    const __half* __restrict__ grad,
    __half* __restrict__ out,
    const int N,
    const int P
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;

    const int p = idx % P;
    const float g = __half2float(grad[idx]);
    atomic_add_f16(out + p, (g * g) / (float)N);
}

// ============================================================================
// BF16 variants
// ============================================================================

extern "C" __global__ void awq_act_scale_bf16(
    const __nv_bfloat16* __restrict__ act,
    __nv_bfloat16* __restrict__ out,
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int j = idx % K;
    const float val = fabsf(__bfloat162float(act[idx]));
    atomic_max_bf16(out + j, val);
}

extern "C" __global__ void awq_score_reduce_bf16(
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ act_scale,
    __nv_bfloat16* __restrict__ out,
    const int M,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    const int j = idx % K;
    const float scale = __bfloat162float(act_scale[j]);
    const float val = scale * fabsf(__bfloat162float(weights[idx])) / (float)M;
    atomic_add_bf16(out + j, val);
}

extern "C" __global__ void fisher_accumulate_bf16(
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ out,
    const int N,
    const int P
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;

    const int p = idx % P;
    const float g = __bfloat162float(grad[idx]);
    atomic_add_bf16(out + p, (g * g) / (float)N);
}
