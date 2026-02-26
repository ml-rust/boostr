// YaRN Rotary Position Embedding - Fused Kernel
// Reference: https://arxiv.org/abs/2309.00071
//
// YaRN = standard split-half RoPE with:
//   1. Frequency scaling baked into cos/sin caches
//   2. Attention scaling factor applied to the output
//
// Same split-half pairing as standard RoPE: (x[d], x[d + D/2])
//
// For each pair d:
//   rotated_first  = x_first[d] * cos[s,d] - x_second[d] * sin[s,d]
//   rotated_second = x_first[d] * sin[s,d] + x_second[d] * cos[s,d]
//   out = rotated * attn_scale
//
// Layout: x is [B, H, S, D] where D is even
//         cos_cache, sin_cache are [S, D/2]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// YaRN RoPE - FP32
// ============================================================================

// Each thread handles one element (first half or second half).
// Total threads = batch_size * num_heads * seq_len * head_dim
extern "C" __global__ void rope_yarn_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float attn_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    const int d = idx % head_dim;
    int remainder = idx / head_dim;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    const float cos_val = cos_cache[s * half_d + pair_idx];
    const float sin_val = sin_cache[s * half_d + pair_idx];

    const int head_offset = b * (num_heads * seq_len * head_dim)
                          + h * (seq_len * head_dim)
                          + s * head_dim;
    const int pos_first = head_offset + pair_idx;
    const int pos_second = head_offset + half_d + pair_idx;

    const float x_first = x[pos_first];
    const float x_second = x[pos_second];

    if (d < half_d) {
        out[pos_first] = (x_first * cos_val - x_second * sin_val) * attn_scale;
    } else {
        out[pos_second] = (x_first * sin_val + x_second * cos_val) * attn_scale;
    }
}

// ============================================================================
// YaRN RoPE - FP16
// ============================================================================

extern "C" __global__ void rope_yarn_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    __half* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float attn_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    const int d = idx % head_dim;
    int remainder = idx / head_dim;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    const float cos_val = __half2float(cos_cache[s * half_d + pair_idx]);
    const float sin_val = __half2float(sin_cache[s * half_d + pair_idx]);

    const int head_offset = b * (num_heads * seq_len * head_dim)
                          + h * (seq_len * head_dim)
                          + s * head_dim;
    const int pos_first = head_offset + pair_idx;
    const int pos_second = head_offset + half_d + pair_idx;

    const float x_first = __half2float(x[pos_first]);
    const float x_second = __half2float(x[pos_second]);

    if (d < half_d) {
        out[pos_first] = __float2half((x_first * cos_val - x_second * sin_val) * attn_scale);
    } else {
        out[pos_second] = __float2half((x_first * sin_val + x_second * cos_val) * attn_scale);
    }
}

// ============================================================================
// YaRN RoPE - BF16
// ============================================================================

extern "C" __global__ void rope_yarn_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float attn_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    const int d = idx % head_dim;
    int remainder = idx / head_dim;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const int half_d = head_dim / 2;
    const int pair_idx = d < half_d ? d : (d - half_d);

    const float cos_val = __bfloat162float(cos_cache[s * half_d + pair_idx]);
    const float sin_val = __bfloat162float(sin_cache[s * half_d + pair_idx]);

    const int head_offset = b * (num_heads * seq_len * head_dim)
                          + h * (seq_len * head_dim)
                          + s * head_dim;
    const int pos_first = head_offset + pair_idx;
    const int pos_second = head_offset + half_d + pair_idx;

    const float x_first = __bfloat162float(x[pos_first]);
    const float x_second = __bfloat162float(x[pos_second]);

    if (d < half_d) {
        out[pos_first] = __float2bfloat16((x_first * cos_val - x_second * sin_val) * attn_scale);
    } else {
        out[pos_second] = __float2bfloat16((x_first * sin_val + x_second * cos_val) * attn_scale);
    }
}
