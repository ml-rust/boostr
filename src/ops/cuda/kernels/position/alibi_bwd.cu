// ALiBi (Attention with Linear Biases) - Backward Pass
// Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
//
// Forward:
//   scores = Q @ K^T / sqrt(d) + alibi_bias
//   alibi_bias[i,j] = -slope * |i - j|
//   attention_out = softmax(scores) @ V
//
// Backward:
//   Since ALiBi bias is a constant (only depends on positions, not inputs),
//   the gradient flows through unchanged:
//   d(scores + bias)/d(scores) = 1
//   d(scores + bias)/d(bias) = 1 (but we don't backprop to positions)
//
//   The key insight is that ALiBi does NOT require gradient computation for
//   the bias itself - it's a fixed positional encoding computed on-the-fly.
//
//   However, we DO need to propagate gradients through the attention itself:
//   grad_Q, grad_K, grad_V = attention_backward(grad_output, probs, Q, K, V, scale)
//
//   No kernel here takes a slope: an additive bias contributes nothing to the
//   Q/K/V gradients, so these kernels serve ANY additive bias, not only ALiBi.
//   They back the MATERIALIZED biased-attention path, which keeps `probs`.
//
// Multi-dtype support: F32, F16, BF16

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// ALiBi Slope Computation (same as forward)
// ============================================================================

__device__ __forceinline__ float get_alibi_slope(int head_idx, int num_heads) {
    return powf(2.0f, -8.0f * head_idx / (float)num_heads);
}

// ============================================================================
// Per-type F32 conversion
// ============================================================================
//
// F16 is 1-5-10, BF16 is 1-8-7. The exponent widths differ, so decoding BF16
// bits with __half2float produces garbage. Every load and store below goes
// through these specializations, which are selected by the actual type T —
// never by sizeof(T).
//
// No __CUDA_ARCH__ guard: __bfloat162float has none, and __float2bfloat16
// ships a round-to-nearest-even software fallback, so both compile and run
// correctly on every arch this file targets. A guard here would silently emit
// nothing on arches it excluded.

template <typename T> __device__ __forceinline__ float alibi_to_f32(const T& x);

template <> __device__ __forceinline__ float alibi_to_f32<float>(const float& x) {
    return x;
}

template <> __device__ __forceinline__ float alibi_to_f32<__half>(const __half& x) {
    return __half2float(x);
}

template <> __device__ __forceinline__ float alibi_to_f32<__nv_bfloat16>(const __nv_bfloat16& x) {
    return __bfloat162float(x);
}

template <typename T> __device__ __forceinline__ T alibi_from_f32(float x);

template <> __device__ __forceinline__ float alibi_from_f32<float>(float x) {
    return x;
}

template <> __device__ __forceinline__ __half alibi_from_f32<__half>(float x) {
    return __float2half(x);
}

template <> __device__ __forceinline__ __nv_bfloat16 alibi_from_f32<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// Fused ALiBi Attention Backward - Softmax Jacobian with ALiBi
// ============================================================================

// Compute softmax backward with ALiBi bias consideration
//
// For standard attention:
//   scores = Q @ K^T / sqrt(d)
//   probs = softmax(scores)
//   grad_scores = probs * (grad_probs - sum(grad_probs * probs))
//
// With ALiBi:
//   scores_alibi = scores + alibi_bias
//   probs = softmax(scores_alibi)
//   grad_scores = probs * (grad_probs - sum(grad_probs * probs))
//   (ALiBi bias doesn't change the softmax jacobian formula)
//
// This kernel computes grad_scores given grad_probs and probs
//
// Launch contract:
//   grid  = (1, seq_len_q, batch_size * num_heads)
//   block = (P, 1, 1) where P is a POWER OF TWO — the tree reduction below
//   halves `block_size` each step and would drop lanes otherwise.
//   dynamic shared memory = blockDim.x * sizeof(float)

template<typename T>
__device__ void alibi_softmax_backward_impl(
    const T* __restrict__ grad_probs,  // [batch, heads, seq_q, seq_k]
    const T* __restrict__ probs,       // [batch, heads, seq_q, seq_k]
    T* __restrict__ grad_scores,       // [batch, heads, seq_q, seq_k]
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k
) {
    extern __shared__ float sdata[];

    int batch_head = blockIdx.z;
    int q_pos = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = batch_head / num_heads;

    if (batch_idx >= batch_size || q_pos >= seq_len_q) return;

    int row_offset = batch_head * seq_len_q * seq_len_k + q_pos * seq_len_k;

    // Step 1: Compute sum(grad_probs * probs) for this row
    float dot_sum = 0.0f;
    for (int k = tid; k < seq_len_k; k += block_size) {
        float gp = alibi_to_f32<T>(grad_probs[row_offset + k]);
        float p = alibi_to_f32<T>(probs[row_offset + k]);
        dot_sum += gp * p;
    }

    // Block reduction
    sdata[tid] = dot_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_dot = sdata[0];
    __syncthreads();

    // Step 2: Compute grad_scores = probs * (grad_probs - row_dot)
    for (int k = tid; k < seq_len_k; k += block_size) {
        float gp = alibi_to_f32<T>(grad_probs[row_offset + k]);
        float p = alibi_to_f32<T>(probs[row_offset + k]);

        float gs = p * (gp - row_dot);

        grad_scores[row_offset + k] = alibi_from_f32<T>(gs);
    }
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. Q
// ============================================================================

// grad_Q = scale * grad_scores @ K
// grad_scores: [batch, heads, seq_q, seq_k]
// K: [batch, heads, seq_k, head_dim]
// grad_Q: [batch, heads, seq_q, head_dim]
//
// Launch contract:
//   grid  = (ceil(head_dim / B), seq_len_q, batch_size * num_heads)
//   block = (B, 1, 1), no shared memory

template<typename T>
__device__ void alibi_backward_grad_q_impl(
    const T* __restrict__ grad_scores,
    const T* __restrict__ K,
    T* __restrict__ grad_Q,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_head = blockIdx.z;
    int q_pos = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = batch_head / num_heads;

    if (batch_idx >= batch_size || q_pos >= seq_len_q || d >= head_dim) return;

    int score_row_offset = batch_head * seq_len_q * seq_len_k + q_pos * seq_len_k;
    int k_batch_offset = batch_head * seq_len_k * head_dim;

    // grad_Q[q_pos, d] = scale * sum_k grad_scores[q_pos, k] * K[k, d]
    float sum = 0.0f;
    for (int k = 0; k < seq_len_k; k++) {
        float gs = alibi_to_f32<T>(grad_scores[score_row_offset + k]);
        float kv = alibi_to_f32<T>(K[k_batch_offset + k * head_dim + d]);
        sum += gs * kv;
    }

    sum *= scale;

    int out_idx = batch_head * seq_len_q * head_dim + q_pos * head_dim + d;
    grad_Q[out_idx] = alibi_from_f32<T>(sum);
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. K
// ============================================================================

// grad_K = scale * grad_scores^T @ Q
// grad_scores^T: [batch, heads, seq_k, seq_q]
// Q: [batch, heads, seq_q, head_dim]
// grad_K: [batch, heads, seq_k, head_dim]
//
// Launch contract:
//   grid  = (ceil(head_dim / B), seq_len_k, batch_size * num_heads)
//   block = (B, 1, 1), no shared memory

template<typename T>
__device__ void alibi_backward_grad_k_impl(
    const T* __restrict__ grad_scores,
    const T* __restrict__ Q,
    T* __restrict__ grad_K,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_head = blockIdx.z;
    int k_pos = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = batch_head / num_heads;

    if (batch_idx >= batch_size || k_pos >= seq_len_k || d >= head_dim) return;

    int score_batch_offset = batch_head * seq_len_q * seq_len_k;
    int q_batch_offset = batch_head * seq_len_q * head_dim;

    // grad_K[k_pos, d] = scale * sum_q grad_scores[q, k_pos] * Q[q, d]
    float sum = 0.0f;
    for (int q = 0; q < seq_len_q; q++) {
        float gs = alibi_to_f32<T>(grad_scores[score_batch_offset + q * seq_len_k + k_pos]);
        float qv = alibi_to_f32<T>(Q[q_batch_offset + q * head_dim + d]);
        sum += gs * qv;
    }

    sum *= scale;

    int out_idx = batch_head * seq_len_k * head_dim + k_pos * head_dim + d;
    grad_K[out_idx] = alibi_from_f32<T>(sum);
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. V
// ============================================================================

// grad_V = probs^T @ grad_output
// probs^T: [batch, heads, seq_k, seq_q]
// grad_output: [batch, heads, seq_q, head_dim]
// grad_V: [batch, heads, seq_k, head_dim]
//
// Launch contract:
//   grid  = (ceil(head_dim / B), seq_len_k, batch_size * num_heads)
//   block = (B, 1, 1), no shared memory

template<typename T>
__device__ void alibi_backward_grad_v_impl(
    const T* __restrict__ probs,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_V,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    int batch_head = blockIdx.z;
    int k_pos = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = batch_head / num_heads;

    if (batch_idx >= batch_size || k_pos >= seq_len_k || d >= head_dim) return;

    int prob_batch_offset = batch_head * seq_len_q * seq_len_k;
    int grad_out_batch_offset = batch_head * seq_len_q * head_dim;

    // grad_V[k_pos, d] = sum_q probs[q, k_pos] * grad_output[q, d]
    float sum = 0.0f;
    for (int q = 0; q < seq_len_q; q++) {
        float p = alibi_to_f32<T>(probs[prob_batch_offset + q * seq_len_k + k_pos]);
        float go = alibi_to_f32<T>(grad_output[grad_out_batch_offset + q * head_dim + d]);
        sum += p * go;
    }

    int out_idx = batch_head * seq_len_k * head_dim + k_pos * head_dim + d;
    grad_V[out_idx] = alibi_from_f32<T>(sum);
}

// ============================================================================
// F32 Kernel Instantiations
// ============================================================================

extern "C" __global__ void alibi_softmax_backward_f32(
    const float* grad_probs, const float* probs, float* grad_scores,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k
) {
    alibi_softmax_backward_impl<float>(
        grad_probs, probs, grad_scores,
        batch_size, num_heads, seq_len_q, seq_len_k
    );
}

extern "C" __global__ void alibi_backward_grad_q_f32(
    const float* grad_scores, const float* K, float* grad_Q,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_q_impl<float>(
        grad_scores, K, grad_Q, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_k_f32(
    const float* grad_scores, const float* Q, float* grad_K,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_k_impl<float>(
        grad_scores, Q, grad_K, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_v_f32(
    const float* probs, const float* grad_output, float* grad_V,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_v_impl<float>(
        probs, grad_output, grad_V,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

// ============================================================================
// F16 Kernel Instantiations
// ============================================================================

extern "C" __global__ void alibi_softmax_backward_f16(
    const __half* grad_probs, const __half* probs, __half* grad_scores,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k
) {
    alibi_softmax_backward_impl<__half>(
        grad_probs, probs, grad_scores,
        batch_size, num_heads, seq_len_q, seq_len_k
    );
}

extern "C" __global__ void alibi_backward_grad_q_f16(
    const __half* grad_scores, const __half* K, __half* grad_Q,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_q_impl<__half>(
        grad_scores, K, grad_Q, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_k_f16(
    const __half* grad_scores, const __half* Q, __half* grad_K,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_k_impl<__half>(
        grad_scores, Q, grad_K, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_v_f16(
    const __half* probs, const __half* grad_output, __half* grad_V,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_v_impl<__half>(
        probs, grad_output, grad_V,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

// ============================================================================
// BF16 Kernel Instantiations
// ============================================================================

extern "C" __global__ void alibi_softmax_backward_bf16(
    const __nv_bfloat16* grad_probs, const __nv_bfloat16* probs, __nv_bfloat16* grad_scores,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k
) {
    alibi_softmax_backward_impl<__nv_bfloat16>(
        grad_probs, probs, grad_scores,
        batch_size, num_heads, seq_len_q, seq_len_k
    );
}

extern "C" __global__ void alibi_backward_grad_q_bf16(
    const __nv_bfloat16* grad_scores, const __nv_bfloat16* K, __nv_bfloat16* grad_Q,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_q_impl<__nv_bfloat16>(
        grad_scores, K, grad_Q, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_k_bf16(
    const __nv_bfloat16* grad_scores, const __nv_bfloat16* Q, __nv_bfloat16* grad_K,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_k_impl<__nv_bfloat16>(
        grad_scores, Q, grad_K, scale,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_v_bf16(
    const __nv_bfloat16* probs, const __nv_bfloat16* grad_output, __nv_bfloat16* grad_V,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_v_impl<__nv_bfloat16>(
        probs, grad_output, grad_V,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

