// Calibration kernels for quantization (AWQ, Fisher Information, GPTQ)
//
// AWQ:    act_scale (per-channel max-abs) + score_reduce (weighted mean)
// Fisher: squared gradient accumulation + normalize
// GPTQ:   column-wise quantization step (F32 only)
//
// Every summing reduction here (score_reduce, fisher_accumulate) writes an F32
// accumulator whatever the input storage dtype is, and reduces inside the block
// before touching it. Both properties are load-bearing:
//
//  * An atomic into 16-bit storage is a read-modify-write that re-rounds the
//    running sum on EVERY add. The terms are squares or absolute values, so
//    nothing cancels and the rounding error accumulates in one direction — a
//    systematic undercount that grows with the accumulation count. The launcher
//    allocates an F32 scratch buffer and narrows it to the caller's dtype once,
//    after the kernel, so each output element is rounded exactly once.
//  * One atomic per matrix element serializes the whole reduction on the output
//    slot. The block folds its contributions first and issues one atomic per
//    output slot per block.

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
// Column-segmented block reduction
//
// Both summing reductions consume a [rows, cols] matrix and produce one output
// slot per COLUMN. The block is 2D: threadIdx.x selects the column (so a warp
// reads 32 adjacent elements of a row — fully coalesced), threadIdx.y walks the
// rows. Each thread folds its own strided slice of rows into a register, then
// the block folds along y so that each column ends in a single value. That is a
// segmented reduction: CALIB_BLOCK_X independent trees run side by side in one
// shared-memory tile, rather than one tree collapsing the block to a scalar.
//
// A flat `idx % cols` layout cannot do this — it interleaves the output slots
// across the block, so no contiguous group of threads shares a slot.
//
// gridDim.y splits the rows so a small column count still fills the device; the
// launcher caps it, which caps the atomics at cols * gridDim.y.
//
// CALIB_BLOCK_X and CALIB_BLOCK_Y are duplicated as `CALIB_BLOCK_X`/`CALIB_BLOCK_Y`
// in `calibration.rs` (the block_dim of `calib_reduce_config`). There is no
// compile-time link between the two copies — a mismatch changes the shared-memory
// tile layout each side assumes and corrupts the reduction silently. Both sides
// must change together.
// ============================================================================

#define CALIB_BLOCK_X 32
#define CALIB_BLOCK_Y 8

// Fold `val` down the y-dimension of the block, per column. The result is the
// column total for this block; only threadIdx.y == 0 is expected to use it.
// Every thread of the block must reach this — it synchronizes.
static __device__ __forceinline__ float calib_reduce_column(float val) {
    __shared__ float smem[CALIB_BLOCK_Y * CALIB_BLOCK_X];

    smem[threadIdx.y * CALIB_BLOCK_X + threadIdx.x] = val;
    __syncthreads();

    for (int s = CALIB_BLOCK_Y / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.y < s) {
            smem[threadIdx.y * CALIB_BLOCK_X + threadIdx.x] +=
                smem[(threadIdx.y + s) * CALIB_BLOCK_X + threadIdx.x];
        }
        __syncthreads();
    }

    return smem[threadIdx.x];
}

// ============================================================================
// AWQ: Score reduction — F32
// weights: [M, K], act_scale: [K], output: [K] (F32 accumulator)
// score[j] = mean_i(act_scale[j] * |W[i, j]|)
//
// act_scale[j] is constant over i, so it multiplies the column total once
// instead of every term. The /M normalization is applied once per block, to the
// block's partial sum — the partials still add up to total/M.
// ============================================================================

extern "C" __global__ void awq_score_reduce_f32(
    const float* __restrict__ weights,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int M,
    const int K
) {
    const int j = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (j < K) {
        for (int i = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             i < M;
             i += CALIB_BLOCK_Y * (int)gridDim.y) {
            acc += fabsf(weights[i * K + j]);
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && j < K) {
        atomicAdd(out + j, act_scale[j] * total / (float)M);
    }
}

// ============================================================================
// Fisher: Squared gradient accumulation — F32
// gradients: [N, P], output: [P] (F32 accumulator)
// fisher[i] = sum_n(grad[n, i]^2) / N
// ============================================================================

extern "C" __global__ void fisher_accumulate_f32(
    const float* __restrict__ grad,
    float* __restrict__ out,
    const int N,
    const int P
) {
    const int p = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (p < P) {
        for (int n = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             n < N;
             n += CALIB_BLOCK_Y * (int)gridDim.y) {
            const float g = grad[n * P + p];
            acc += g * g;
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && p < P) {
        atomicAdd(out + p, total / (float)N);
    }
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

// atomicAdd for a single __half element. Unused: the summing reductions above
// accumulate into an F32 buffer with native atomicAdd(float*).
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

// atomicAdd for a single __nv_bfloat16 element. Unused, for the same reason as
// atomic_add_f16 above.
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
    float* __restrict__ out,
    const int M,
    const int K
) {
    const int j = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (j < K) {
        for (int i = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             i < M;
             i += CALIB_BLOCK_Y * (int)gridDim.y) {
            acc += fabsf(__half2float(weights[i * K + j]));
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && j < K) {
        atomicAdd(out + j, __half2float(act_scale[j]) * total / (float)M);
    }
}

extern "C" __global__ void fisher_accumulate_f16(
    const __half* __restrict__ grad,
    float* __restrict__ out,
    const int N,
    const int P
) {
    const int p = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (p < P) {
        for (int n = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             n < N;
             n += CALIB_BLOCK_Y * (int)gridDim.y) {
            const float g = __half2float(grad[n * P + p]);
            acc += g * g;
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && p < P) {
        atomicAdd(out + p, total / (float)N);
    }
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
    float* __restrict__ out,
    const int M,
    const int K
) {
    const int j = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (j < K) {
        for (int i = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             i < M;
             i += CALIB_BLOCK_Y * (int)gridDim.y) {
            acc += fabsf(__bfloat162float(weights[i * K + j]));
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && j < K) {
        atomicAdd(out + j, __bfloat162float(act_scale[j]) * total / (float)M);
    }
}

extern "C" __global__ void fisher_accumulate_bf16(
    const __nv_bfloat16* __restrict__ grad,
    float* __restrict__ out,
    const int N,
    const int P
) {
    const int p = blockIdx.x * CALIB_BLOCK_X + threadIdx.x;

    float acc = 0.0f;
    if (p < P) {
        for (int n = blockIdx.y * CALIB_BLOCK_Y + threadIdx.y;
             n < N;
             n += CALIB_BLOCK_Y * (int)gridDim.y) {
            const float g = __bfloat162float(grad[n * P + p]);
            acc += g * g;
        }
    }

    const float total = calib_reduce_column(acc);
    if (threadIdx.y == 0 && p < P) {
        atomicAdd(out + p, total / (float)N);
    }
}
