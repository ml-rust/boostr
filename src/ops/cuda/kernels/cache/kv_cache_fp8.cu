// FP8 KV cache quantization.
//
// E4M3 carries 4 exponent bits and 3 mantissa bits, so it holds a wider
// dynamic range than INT8 at the same one byte per element. E5M2 trades
// mantissa for range again.
//
// Scale convention: `f32_to_fp8_e4m3_raw` does `val * scale` and
// `fp8_e4m3_to_f32` does `fp8_val / scale`, so a stored scale is 448/max_abs,
// where 448 is the E4M3 maximum.
//
// The conversion helpers use hardware FP8 from sm_89 and a bit-exact software
// encoder below it, so this unit compiles and runs correctly at sm_75.
//
// This file is canonical for per-token FP8 quantize/dequantize dispatch.
// kv_cache_quant.cu duplicates the fp16 and bf16 quantize entry points and
// additionally has an fp32 quantize kernel with no matching fp32 dequant
// anywhere; see the Rust dispatch in cuda/cache/kv_cache_quant.rs for why
// that fp32 kernel stays unwired.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Per-Tensor Quantization (Global scale for entire KV cache)
// ============================================================================

// Per-tensor quantization needs a max-abs reduced over the WHOLE tensor
// before any element can be quantized, but CUDA has no grid-wide barrier
// inside a single kernel. A prior version tried to do the reduction and the
// quantize pass in one launch, gated by `blockIdx.x == 0`: that silently
// discarded every block's local max except block 0's, and even for block 0
// the `__syncthreads()` between the two passes only fences within a block,
// giving other blocks no ordering guarantee against block 0's write. This
// is a three-kernel pipeline instead:
//   1. `quantize_kv_fp8_per_tensor_fp16_find_max` — every block folds its
//      slice to a local max, then contributes via `atomicMax` into `scale`
//      (reinterpreted as `int`; ordering over non-negative floats matches
//      ordering over their bit patterns, the same trick as
//      `awq_act_scale_f32` in quantization/calibration.cu). The caller MUST
//      zero-initialize `scale` first — an atomic max over garbage is a
//      silent wrong-answer bug.
//   2. `quantize_kv_fp8_per_tensor_fp16_finalize_scale` — single thread,
//      converts the reduced max-abs into the stored scale `448/max_abs`.
//      Stream ordering between sequential launches guarantees stage 1 has
//      fully finished before this runs.
//   3. `quantize_kv_fp8_per_tensor_fp16` — quantizes every element with the
//      now-finalized `*scale`.

// Stage 1: block-local max-abs, folded into a global max via atomicMax.
extern "C" __global__ void quantize_kv_fp8_per_tensor_fp16_find_max(
    const __half* __restrict__ kv_fp16,
    float* __restrict__ scale,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem_max[256];

    float local_max = 0.0f;
    if (idx < total_elements) {
        local_max = fabsf(__half2float(kv_fp16[idx]));
    }

    smem_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_max[threadIdx.x] = fmaxf(smem_max[threadIdx.x], smem_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int* scale_int = (int*)scale;
        atomicMax(scale_int, __float_as_int(smem_max[0]));
    }
}

// Stage 2: convert the reduced max-abs (still raw bits in `scale` from stage
// 1) into the stored scale convention `448/max_abs`.
extern "C" __global__ void quantize_kv_fp8_per_tensor_fp16_finalize_scale(
    float* __restrict__ scale
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float max_val = *scale;
    *scale = (max_val > 0.0f) ? (448.0f / max_val) : 1.0f;
}

// Stage 3: quantize every element with the finalized `*scale`.
//
// Args:
//   kv_fp8: Output FP8 tensor [batch, num_kv_heads, seq_len, head_dim]
//   kv_fp16: Input FP16 tensor [batch, num_kv_heads, seq_len, head_dim]
//   scale: Finalized scale factor (single value for entire tensor)
//   total_elements: batch * num_kv_heads * seq_len * head_dim
extern "C" __global__ void quantize_kv_fp8_per_tensor_fp16(
    boostr_fp8_e4m3* __restrict__ kv_fp8,
    const __half* __restrict__ kv_fp16,
    const float* __restrict__ scale,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = __half2float(kv_fp16[idx]);
    kv_fp8[idx] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(val, *scale));
}

// Dequantize FP8 KV cache to FP16 with per-tensor scaling
extern "C" __global__ void dequantize_kv_fp8_per_tensor_fp16(
    __half* __restrict__ kv_fp16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float scale,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    float val = fp8_e4m3_to_f32((uint8_t)kv_fp8[idx], scale);
    kv_fp16[idx] = __float2half(val);
}

// ============================================================================
// Per-Token Quantization (Separate scale for each token)
// ============================================================================

// Per-token quantization provides better accuracy for non-uniform distributions
// Each token (across head_dim) gets its own scale factor
//
// Args:
//   kv_fp8: Output FP8 tensor [batch, num_kv_heads, seq_len, head_dim]
//   kv_fp16: Input FP16 tensor [batch, num_kv_heads, seq_len, head_dim]
//   scales: Output scale factors [batch, num_kv_heads, seq_len]
//   batch, num_kv_heads, seq_len, head_dim: Tensor dimensions

__device__ void quantize_kv_fp8_per_token_impl(
    boostr_fp8_e4m3* __restrict__ kv_fp8,
    const __half* __restrict__ kv_fp16,
    float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int token_idx = blockIdx.x;  // Each block handles one token
    const int tid = threadIdx.x;

    if (token_idx >= batch * num_kv_heads * seq_len) return;

    const int token_offset = token_idx * head_dim;

    // Shared memory for reduction
    __shared__ float smem_max[256];

    // Each thread processes multiple elements if head_dim > blockDim.x
    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(kv_fp16[token_offset + d])));
    }

    // Block-level reduction
    smem_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + stride]);
        }
        __syncthreads();
    }

    // Compute scale: 448 / max_abs maps values into FP8 representable range [-448, 448]
    // f32_to_fp8_e4m3_raw does val * scale, fp8_e4m3_to_f32 does fp8_val / scale
    __shared__ float token_scale;
    if (tid == 0) {
        float max_val = smem_max[0];
        token_scale = (max_val > 0.0f) ? (448.0f / max_val) : 1.0f;
        scales[token_idx] = token_scale;
    }
    __syncthreads();

    // Quantize this token's values
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = __half2float(kv_fp16[token_offset + d]);
        kv_fp8[token_offset + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(val, token_scale));
    }
}

extern "C" __global__ void quantize_kv_fp8_per_token_fp16(
    boostr_fp8_e4m3* kv_fp8,
    const __half* kv_fp16,
    float* scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    quantize_kv_fp8_per_token_impl(kv_fp8, kv_fp16, scales, batch, num_kv_heads, seq_len, head_dim);
}

// Dequantize with per-token scales
__device__ void dequantize_kv_fp8_per_token_impl(
    __half* __restrict__ kv_fp16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= batch * num_kv_heads * seq_len) return;

    const int token_offset = token_idx * head_dim;
    const float token_scale = scales[token_idx];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = fp8_e4m3_to_f32((uint8_t)kv_fp8[token_offset + d], token_scale);
        kv_fp16[token_offset + d] = __float2half(val);
    }
}

extern "C" __global__ void dequantize_kv_fp8_per_token_fp16(
    __half* kv_fp16,
    const boostr_fp8_e4m3* kv_fp8,
    const float* scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    dequantize_kv_fp8_per_token_impl(kv_fp16, kv_fp8, scales, batch, num_kv_heads, seq_len, head_dim);
}

// ============================================================================
// BF16 Variants
// ============================================================================

__device__ void quantize_kv_fp8_per_token_bf16_impl(
    boostr_fp8_e4m3* __restrict__ kv_fp8,
    const __nv_bfloat16* __restrict__ kv_bf16,
    float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= batch * num_kv_heads * seq_len) return;

    const int token_offset = token_idx * head_dim;

    __shared__ float smem_max[256];

    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(kv_bf16[token_offset + d])));
    }

    smem_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + stride]);
        }
        __syncthreads();
    }

    __shared__ float token_scale;
    if (tid == 0) {
        float max_val = smem_max[0];
        token_scale = (max_val > 0.0f) ? (448.0f / max_val) : 1.0f;
        scales[token_idx] = token_scale;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = __bfloat162float(kv_bf16[token_offset + d]);
        kv_fp8[token_offset + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(val, token_scale));
    }
}

extern "C" __global__ void quantize_kv_fp8_per_token_bf16(
    boostr_fp8_e4m3* kv_fp8,
    const __nv_bfloat16* kv_bf16,
    float* scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    quantize_kv_fp8_per_token_bf16_impl(kv_fp8, kv_bf16, scales, batch, num_kv_heads, seq_len, head_dim);
}

__device__ void dequantize_kv_fp8_per_token_bf16_impl(
    __nv_bfloat16* __restrict__ kv_bf16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= batch * num_kv_heads * seq_len) return;

    const int token_offset = token_idx * head_dim;
    const float token_scale = scales[token_idx];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = fp8_e4m3_to_f32((uint8_t)kv_fp8[token_offset + d], token_scale);
        kv_bf16[token_offset + d] = __float2bfloat16(val);
    }
}

extern "C" __global__ void dequantize_kv_fp8_per_token_bf16(
    __nv_bfloat16* kv_bf16,
    const boostr_fp8_e4m3* kv_fp8,
    const float* scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    dequantize_kv_fp8_per_token_bf16_impl(kv_bf16, kv_fp8, scales, batch, num_kv_heads, seq_len, head_dim);
}
