// Rotary Position Embedding (RoPE) - Fused Kernel
// Reference: https://arxiv.org/abs/2104.09864 (RoFormer)
//
// Applies rotary embeddings to query/key tensors in-place using split-half representation:
// x is split into [x_first_half, x_second_half] where each position i is a pair.
// For each pair i:
//   out_first[i]  = x_first[i] * cos[s, i] - x_second[i] * sin[s, i]
//   out_second[i] = x_first[i] * sin[s, i] + x_second[i] * cos[s, i]
//
// Layout: x is [B, H, S, D] where D is even
//         cos_cache, sin_cache are [S, D/2]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// RoPE Application - FP32
// ============================================================================

// Each thread handles one element from either the first half or second half
// of the hidden dimension, applying the rotation formula.
// Total threads = batch_size * num_heads * seq_len * head_dim
extern "C" __global__ void rope_apply_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    // Decompose linear index into (b, h, s, d)
    const int d = idx % head_dim;
    int remainder = idx / head_dim;

    const int s = remainder % seq_len;
    remainder /= seq_len;

    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    // Load cos and sin for this position and pair index
    const float cos_val = cos_cache[s * half_d + pair_idx];
    const float sin_val = sin_cache[s * half_d + pair_idx];

    // Positions in the x tensor
    const int head_offset = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim;
    const int pos_first = head_offset + pair_idx;           // x[..., pair_idx]
    const int pos_second = head_offset + half_d + pair_idx; // x[..., half_d + pair_idx]

    const float x_first = x[pos_first];
    const float x_second = x[pos_second];

    // Apply rotation formula:
    // out_first = x_first * cos - x_second * sin
    // out_second = x_first * sin + x_second * cos
    if (d < half_d) {
        // First half: write out_first
        out[pos_first] = x_first * cos_val - x_second * sin_val;
    } else {
        // Second half: write out_second
        out[pos_second] = x_first * sin_val + x_second * cos_val;
    }
}

// ============================================================================
// RoPE Application - FP16
// ============================================================================

extern "C" __global__ void rope_apply_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    __half* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    // Decompose linear index into (b, h, s, d)
    const int d = idx % head_dim;
    int remainder = idx / head_dim;

    const int s = remainder % seq_len;
    remainder /= seq_len;

    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    // Load cos and sin
    const __half cos_val = cos_cache[s * half_d + pair_idx];
    const __half sin_val = sin_cache[s * half_d + pair_idx];

    // Linear positions
    const int head_offset = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim;
    const int pos_first = head_offset + pair_idx;
    const int pos_second = head_offset + half_d + pair_idx;

    // Load and convert to float for computation
    float x_first = __half2float(x[pos_first]);
    float x_second = __half2float(x[pos_second]);
    float cos_f = __half2float(cos_val);
    float sin_f = __half2float(sin_val);

    // Apply rotation in FP32
    if (d < half_d) {
        float out_first = x_first * cos_f - x_second * sin_f;
        out[pos_first] = __float2half(out_first);
    } else {
        float out_second = x_first * sin_f + x_second * cos_f;
        out[pos_second] = __float2half(out_second);
    }
}

// ============================================================================
// RoPE Application - BF16
// ============================================================================

extern "C" __global__ void rope_apply_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    // Decompose linear index into (b, h, s, d)
    const int d = idx % head_dim;
    int remainder = idx / head_dim;

    const int s = remainder % seq_len;
    remainder /= seq_len;

    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    // Load cos and sin
    const __nv_bfloat16 cos_val = cos_cache[s * half_d + pair_idx];
    const __nv_bfloat16 sin_val = sin_cache[s * half_d + pair_idx];

    // Linear positions
    const int head_offset = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim;
    const int pos_first = head_offset + pair_idx;
    const int pos_second = head_offset + half_d + pair_idx;

    // Load and convert to float
    float x_first = __bfloat162float(x[pos_first]);
    float x_second = __bfloat162float(x[pos_second]);
    float cos_f = __bfloat162float(cos_val);
    float sin_f = __bfloat162float(sin_val);

    // Apply rotation in FP32
    if (d < half_d) {
        float out_first = x_first * cos_f - x_second * sin_f;
        out[pos_first] = __float2bfloat16(out_first);
    } else {
        float out_second = x_first * sin_f + x_second * cos_f;
        out[pos_second] = __float2bfloat16(out_second);
    }
}
