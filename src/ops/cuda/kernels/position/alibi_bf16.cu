// ALiBi bf16 kernels, split out of alibi.cu.
// `__nv_bfloat16` conversion needs sm_80. Splitting keeps the F32/F16/FP8
// kernels in alibi.cu on sm_75 so they still load on Turing.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include "alibi.cuh"

// ============================================================================
// ALiBi Bias Injection - BF16
// ============================================================================

__device__ void alibi_add_bias_bf16_impl(
    __nv_bfloat16* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len_q * seq_len_k;

    if (idx >= total) return;

    const int k_pos = idx % seq_len_k;
    const int q_pos = (idx / seq_len_k) % seq_len_q;
    const int head_idx = (idx / (seq_len_k * seq_len_q)) % num_heads;
    const int batch_idx = idx / (seq_len_k * seq_len_q * num_heads);

    const float slope = get_alibi_slope(head_idx, num_heads);
    const int distance = q_pos - k_pos;
    const float bias = -slope * abs(distance);

    float score = __bfloat162float(scores[idx]);
    score += bias;
    scores[idx] = __float2bfloat16(score);
}

extern "C" __global__ void alibi_add_bias_bf16(
    __nv_bfloat16* scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k
) {
    alibi_add_bias_bf16_impl(scores, batch_size, num_heads, seq_len_q, seq_len_k);
}

// ============================================================================
// ALiBi Bias + Causal Mask (combined, single pass) - BF16
// ============================================================================

__device__ void alibi_add_bias_causal_bf16_impl(
    __nv_bfloat16* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int position
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len_q * seq_len_k;

    if (idx >= total) return;

    const int k_pos = idx % seq_len_k;
    const int q_pos = (idx / seq_len_k) % seq_len_q;
    const int head_idx = (idx / (seq_len_k * seq_len_q)) % num_heads;

    const int abs_q_pos = q_pos + position;

    if (k_pos > abs_q_pos) {
        scores[idx] = __float2bfloat16(-INFINITY);
        return;
    }

    const float slope = get_alibi_slope(head_idx, num_heads);
    const int distance = abs_q_pos - k_pos;
    const float bias = -slope * (float)distance;

    float score = __bfloat162float(scores[idx]);
    score += bias;
    scores[idx] = __float2bfloat16(score);
}

extern "C" __global__ void alibi_add_bias_causal_bf16(
    __nv_bfloat16* scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int position
) {
    alibi_add_bias_causal_bf16_impl(scores, batch_size, num_heads, seq_len_q, seq_len_k, position);
}
