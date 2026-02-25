// FP8 KV Cache Backward Pass for Training
// Reference: "FP8 Formats for Deep Learning" (NVIDIA H100 Architecture Whitepaper)
//
// Backward pass for quantized KV cache operations:
// - grad_kv = grad_kv_fp8 * (1 / scale)  (straight-through estimator)
// - grad_scale = sum(grad_kv_fp8 * kv_fp8 / scale^2)
//
// For training with FP8 KV cache:
// Forward:  kv_fp8 = quantize(kv, scale)
// Backward: grad_kv = dequantize_grad(grad_kv_fp8, scale)
//           grad_scale = compute_scale_grad(grad_kv_fp8, kv_fp8, scale)
//
// The straight-through estimator (STE) passes gradients through quantization
// as if it were the identity function, with scaling adjustments.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Warp-level Primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_kv(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Straight-Through Estimator (STE) Backward for Per-Tensor Quantization
// ============================================================================

// Backward through dequantization: grad_kv = grad_output
// The STE approximation treats quantization as identity during backward
//
// Args:
//   grad_kv_fp16: Output gradient for KV cache (FP16) [total_elements]
//   grad_output_fp16: Input gradient from attention (FP16) [total_elements]
//   kv_fp8: Quantized KV cache (for scale gradient) [total_elements]
//   scale: Quantization scale factor
//   total_elements: Total number of elements
//   grad_scale: Output gradient for scale factor (single value)

extern "C" __global__ void kv_cache_fp8_bwd_per_tensor_fp16(
    __half* __restrict__ grad_kv_fp16,
    const __half* __restrict__ grad_output_fp16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float scale,
    const int total_elements,
    float* __restrict__ grad_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    if (idx < total_elements) {
        // STE: gradient passes through unchanged
        float grad_out = __half2float(grad_output_fp16[idx]);
        grad_kv_fp16[idx] = __float2half(grad_out);

        // Compute scale gradient contribution
        // d(kv_fp8 / scale) / d(scale) = -kv_fp8 / scale^2
        // grad_scale += grad_output * (-kv_fp8 / scale^2)
        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[idx], 1.0f);  // Get raw FP8 value
        local_grad_scale = grad_out * (-kv_val / (scale * scale));
    }

    // Block-level reduction for scale gradient
    smem_sum[threadIdx.x] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // First thread atomically adds to global scale gradient
    if (threadIdx.x == 0) {
        atomicAdd(grad_scale, smem_sum[0]);
    }
}

// ============================================================================
// Backward for Per-Token Quantization (FP16)
// ============================================================================

// Per-token backward: each token has its own scale
// grad_kv = grad_output (STE)
// grad_scales[token] = sum_d(grad_output[d] * -kv_fp8[d] / scale^2)

extern "C" __global__ void kv_cache_fp8_bwd_per_token_fp16(
    __half* __restrict__ grad_kv_fp16,
    const __half* __restrict__ grad_output_fp16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    float* __restrict__ grad_scales
) {
    const int token_idx = blockIdx.x;  // Each block handles one token
    const int tid = threadIdx.x;

    const int total_tokens = batch * num_kv_heads * seq_len;
    if (token_idx >= total_tokens) return;

    const int token_offset = token_idx * head_dim;
    const float token_scale = scales[token_idx];
    const float inv_scale_sq = 1.0f / (token_scale * token_scale);

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    // Process head_dim elements
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int global_idx = token_offset + d;

        // STE: pass gradient through
        float grad_out = __half2float(grad_output_fp16[global_idx]);
        grad_kv_fp16[global_idx] = __float2half(grad_out);

        // Accumulate scale gradient
        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[global_idx], 1.0f);
        local_grad_scale += grad_out * (-kv_val * inv_scale_sq);
    }

    // Block-level reduction
    smem_sum[tid] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_sum[tid] += smem_sum[tid + stride];
        }
        __syncthreads();
    }

    // Write scale gradient for this token
    if (tid == 0) {
        grad_scales[token_idx] = smem_sum[0];
    }
}

// ============================================================================
// Backward for Per-Tensor Quantization (BF16)
// ============================================================================

extern "C" __global__ void kv_cache_fp8_bwd_per_tensor_bf16(
    __nv_bfloat16* __restrict__ grad_kv_bf16,
    const __nv_bfloat16* __restrict__ grad_output_bf16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float scale,
    const int total_elements,
    float* __restrict__ grad_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    if (idx < total_elements) {
        float grad_out = __bfloat162float(grad_output_bf16[idx]);
        grad_kv_bf16[idx] = __float2bfloat16(grad_out);

        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[idx], 1.0f);
        local_grad_scale = grad_out * (-kv_val / (scale * scale));
    }

    smem_sum[threadIdx.x] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(grad_scale, smem_sum[0]);
    }
}

// ============================================================================
// Backward for Per-Token Quantization (BF16)
// ============================================================================

extern "C" __global__ void kv_cache_fp8_bwd_per_token_bf16(
    __nv_bfloat16* __restrict__ grad_kv_bf16,
    const __nv_bfloat16* __restrict__ grad_output_bf16,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    float* __restrict__ grad_scales
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int total_tokens = batch * num_kv_heads * seq_len;
    if (token_idx >= total_tokens) return;

    const int token_offset = token_idx * head_dim;
    const float token_scale = scales[token_idx];
    const float inv_scale_sq = 1.0f / (token_scale * token_scale);

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int global_idx = token_offset + d;

        float grad_out = __bfloat162float(grad_output_bf16[global_idx]);
        grad_kv_bf16[global_idx] = __float2bfloat16(grad_out);

        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[global_idx], 1.0f);
        local_grad_scale += grad_out * (-kv_val * inv_scale_sq);
    }

    smem_sum[tid] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_sum[tid] += smem_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_scales[token_idx] = smem_sum[0];
    }
}

// ============================================================================
// Backward for Per-Tensor Quantization (FP32)
// ============================================================================

extern "C" __global__ void kv_cache_fp8_bwd_per_tensor_fp32(
    float* __restrict__ grad_kv_fp32,
    const float* __restrict__ grad_output_fp32,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float scale,
    const int total_elements,
    float* __restrict__ grad_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    if (idx < total_elements) {
        float grad_out = grad_output_fp32[idx];
        grad_kv_fp32[idx] = grad_out;

        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[idx], 1.0f);
        local_grad_scale = grad_out * (-kv_val / (scale * scale));
    }

    smem_sum[threadIdx.x] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(grad_scale, smem_sum[0]);
    }
}

// ============================================================================
// Backward for Per-Token Quantization (FP32)
// ============================================================================

extern "C" __global__ void kv_cache_fp8_bwd_per_token_fp32(
    float* __restrict__ grad_kv_fp32,
    const float* __restrict__ grad_output_fp32,
    const boostr_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    const int batch,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    float* __restrict__ grad_scales
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int total_tokens = batch * num_kv_heads * seq_len;
    if (token_idx >= total_tokens) return;

    const int token_offset = token_idx * head_dim;
    const float token_scale = scales[token_idx];
    const float inv_scale_sq = 1.0f / (token_scale * token_scale);

    __shared__ float smem_sum[256];

    float local_grad_scale = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int global_idx = token_offset + d;

        float grad_out = grad_output_fp32[global_idx];
        grad_kv_fp32[global_idx] = grad_out;

        float kv_val = fp8_e4m3_to_f32((uint8_t)kv_fp8[global_idx], 1.0f);
        local_grad_scale += grad_out * (-kv_val * inv_scale_sq);
    }

    smem_sum[tid] = local_grad_scale;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_sum[tid] += smem_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_scales[token_idx] = smem_sum[0];
    }
}
