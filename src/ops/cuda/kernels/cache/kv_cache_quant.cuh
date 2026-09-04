#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// Quantize FP16/BF16/FP32 to FP8 with per-token scales
// Input: [num_tokens, head_dim]
// Output: [num_tokens, head_dim] (FP8) + [num_tokens] (scales)
template<typename T>
__device__ __forceinline__ void quantize_kv_fp8_per_token_impl(
    const T* __restrict__ input,           // [num_tokens, head_dim]
    boostr_fp8_e4m3* __restrict__ output,  // [num_tokens, head_dim]
    float* __restrict__ scales,             // [num_tokens]
    int num_tokens,
    int head_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    const T* token_in = input + token_idx * head_dim;
    boostr_fp8_e4m3* token_out = output + token_idx * head_dim;

    // Step 1: Find max absolute value for this token (parallel reduction)
    __shared__ float sdata[256];
    float local_max = 0.0f;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(token_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = token_in[i];
        } else { // BF16
            val = __bfloat162float(token_in[i]);
        }
        local_max = fmaxf(local_max, fabsf(val));
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction to find global max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Compute scale: max_val / FP8_max (448 for E4M3)
    __shared__ float scale;
    if (tid == 0) {
        float max_val = sdata[0];
        scale = (max_val > 0.0f) ? (max_val / 448.0f) : 1.0f;
        scales[token_idx] = scale;
    }
    __syncthreads();

    // Step 2: Quantize elements using dtype_traits conversions
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(token_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = token_in[i];
        } else {
            val = __bfloat162float(token_in[i]);
        }
        token_out[i] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(val, scale));
    }
}

template<typename T>
__device__ __forceinline__ void quantize_kv_fp8_per_head_impl(
    const T* __restrict__ input,           // [num_heads, seq_len, head_dim]
    boostr_fp8_e4m3* __restrict__ output,  // [num_heads, seq_len, head_dim]
    float* __restrict__ scales,             // [num_heads]
    int num_heads,
    int seq_len,
    int head_dim
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (head_idx >= num_heads) return;

    const T* head_in = input + head_idx * seq_len * head_dim;
    boostr_fp8_e4m3* head_out = output + head_idx * seq_len * head_dim;

    // Find max absolute value across entire head
    __shared__ float sdata[256];
    float local_max = 0.0f;

    int total_elements = seq_len * head_dim;
    for (int i = tid; i < total_elements; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(head_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = head_in[i];
        } else {
            val = __bfloat162float(head_in[i]);
        }
        local_max = fmaxf(local_max, fabsf(val));
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    __shared__ float scale;
    if (tid == 0) {
        float max_val = sdata[0];
        scale = (max_val > 0.0f) ? (max_val / 448.0f) : 1.0f;
        scales[head_idx] = scale;
    }
    __syncthreads();

    // Quantize all elements using dtype_traits conversions
    for (int i = tid; i < total_elements; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(head_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = head_in[i];
        } else {
            val = __bfloat162float(head_in[i]);
        }
        head_out[i] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(val, scale));
    }
}

// INT8 per-token quantization: each token finds its max absolute value
// and scales so max_abs → 127
template<typename T>
__device__ __forceinline__ void quantize_kv_int8_per_token_impl(
    const T* __restrict__ input,           // [num_tokens, head_dim]
    int8_t* __restrict__ output,           // [num_tokens, head_dim]
    float* __restrict__ scales,            // [num_tokens]
    int num_tokens,
    int head_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    const T* token_in = input + token_idx * head_dim;
    int8_t* token_out = output + token_idx * head_dim;

    // Step 1: Find max absolute value for this token (parallel reduction)
    __shared__ float sdata[256];
    float local_max = 0.0f;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(token_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = token_in[i];
        } else { // BF16
            val = __bfloat162float(token_in[i]);
        }
        local_max = fmaxf(local_max, fabsf(val));
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction to find global max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Compute scale: max_val / 127 for INT8
    __shared__ float scale;
    if (tid == 0) {
        float max_val = sdata[0];
        scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
        scales[token_idx] = scale;
    }
    __syncthreads();

    // Step 2: Quantize elements
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(token_in[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            val = token_in[i];
        } else {
            val = __bfloat162float(token_in[i]);
        }
        float normalized = val / scale;
        int8_t quantized = (int8_t)__float2int_rn(fminf(127.0f, fmaxf(-127.0f, normalized)));
        token_out[i] = quantized;
    }
}
