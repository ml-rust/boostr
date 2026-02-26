// Interleaved Rotary Position Embedding (RoPE) - Fused Kernel
// Reference: https://arxiv.org/abs/2104.09864 (RoFormer)
//
// Interleaved pairing: pairs are (x[2d], x[2d+1]) for d in [0, head_dim/2)
// This is the "mathematically pure" complex number form used by GPT-NeoX, Qwen, RoFormer.
//
// For each pair d:
//   out[2d]   = x[2d] * cos[s,d] - x[2d+1] * sin[s,d]
//   out[2d+1] = x[2d] * sin[s,d] + x[2d+1] * cos[s,d]
//
// Layout: x is [B, H, S, D] where D is even
//         cos_cache, sin_cache are [S, D/2]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Interleaved RoPE - FP32
// ============================================================================

// Each thread handles one dimension pair (2 elements).
// Total threads = batch_size * num_heads * seq_len * (head_dim / 2)
extern "C" __global__ void rope_interleaved_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_d = head_dim / 2;
    const int total_pairs = batch_size * num_heads * seq_len * half_d;

    if (pair_idx >= total_pairs) return;

    // Decompose linear index into (b, h, s, d_pair)
    const int d_pair = pair_idx % half_d;
    int remainder = pair_idx / half_d;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    // Load cos/sin for this position and pair
    const float cos_val = cos_cache[s * half_d + d_pair];
    const float sin_val = sin_cache[s * half_d + d_pair];

    // Compute base offset in x: [b, h, s, ...]
    const int base = b * (num_heads * seq_len * head_dim)
                   + h * (seq_len * head_dim)
                   + s * head_dim;

    const int idx_even = base + d_pair * 2;
    const int idx_odd = base + d_pair * 2 + 1;

    const float x_even = x[idx_even];
    const float x_odd = x[idx_odd];

    out[idx_even] = x_even * cos_val - x_odd * sin_val;
    out[idx_odd]  = x_even * sin_val + x_odd * cos_val;
}

// ============================================================================
// Interleaved RoPE - FP16
// ============================================================================

extern "C" __global__ void rope_interleaved_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    __half* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_d = head_dim / 2;
    const int total_pairs = batch_size * num_heads * seq_len * half_d;

    if (pair_idx >= total_pairs) return;

    const int d_pair = pair_idx % half_d;
    int remainder = pair_idx / half_d;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const float cos_val = __half2float(cos_cache[s * half_d + d_pair]);
    const float sin_val = __half2float(sin_cache[s * half_d + d_pair]);

    const int base = b * (num_heads * seq_len * head_dim)
                   + h * (seq_len * head_dim)
                   + s * head_dim;

    const int idx_even = base + d_pair * 2;
    const int idx_odd = base + d_pair * 2 + 1;

    const float x_even = __half2float(x[idx_even]);
    const float x_odd = __half2float(x[idx_odd]);

    out[idx_even] = __float2half(x_even * cos_val - x_odd * sin_val);
    out[idx_odd]  = __float2half(x_even * sin_val + x_odd * cos_val);
}

// ============================================================================
// Interleaved RoPE - BF16
// ============================================================================

extern "C" __global__ void rope_interleaved_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_d = head_dim / 2;
    const int total_pairs = batch_size * num_heads * seq_len * half_d;

    if (pair_idx >= total_pairs) return;

    const int d_pair = pair_idx % half_d;
    int remainder = pair_idx / half_d;
    const int s = remainder % seq_len;
    remainder /= seq_len;
    const int h = remainder % num_heads;
    const int b = remainder / num_heads;

    const float cos_val = __bfloat162float(cos_cache[s * half_d + d_pair]);
    const float sin_val = __bfloat162float(sin_cache[s * half_d + d_pair]);

    const int base = b * (num_heads * seq_len * head_dim)
                   + h * (seq_len * head_dim)
                   + s * head_dim;

    const int idx_even = base + d_pair * 2;
    const int idx_odd = base + d_pair * 2 + 1;

    const float x_even = __bfloat162float(x[idx_even]);
    const float x_odd = __bfloat162float(x[idx_odd]);

    out[idx_even] = __float2bfloat16(x_even * cos_val - x_odd * sin_val);
    out[idx_odd]  = __float2bfloat16(x_even * sin_val + x_odd * cos_val);
}
