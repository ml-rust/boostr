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
//   However, we DO need to propagate gradients through the fused ALiBi attention:
//   grad_Q, grad_K, grad_V = attention_backward(grad_output, Q, K, V, scores, alibi_slope)
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
    int head_idx = batch_head % num_heads;

    if (batch_idx >= batch_size || q_pos >= seq_len_q) return;

    int row_offset = batch_head * seq_len_q * seq_len_k + q_pos * seq_len_k;

    // Step 1: Compute sum(grad_probs * probs) for this row
    float dot_sum = 0.0f;
    for (int k = tid; k < seq_len_k; k += block_size) {
        float gp, p;
        if constexpr (sizeof(T) == 4) {
            gp = grad_probs[row_offset + k];
            p = probs[row_offset + k];
        } else if constexpr (sizeof(T) == 2) {
            // Check if it's __half or __nv_bfloat16
            gp = __half2float(*reinterpret_cast<const __half*>(&grad_probs[row_offset + k]));
            p = __half2float(*reinterpret_cast<const __half*>(&probs[row_offset + k]));
        }
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
        float gp, p;
        if constexpr (sizeof(T) == 4) {
            gp = grad_probs[row_offset + k];
            p = probs[row_offset + k];
        } else if constexpr (sizeof(T) == 2) {
            gp = __half2float(*reinterpret_cast<const __half*>(&grad_probs[row_offset + k]));
            p = __half2float(*reinterpret_cast<const __half*>(&probs[row_offset + k]));
        }

        float gs = p * (gp - row_dot);

        if constexpr (sizeof(T) == 4) {
            grad_scores[row_offset + k] = gs;
        } else if constexpr (sizeof(T) == 2) {
            *reinterpret_cast<__half*>(&grad_scores[row_offset + k]) = __float2half(gs);
        }
    }
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. Q
// ============================================================================

// grad_Q = scale * grad_scores @ K
// grad_scores: [batch, heads, seq_q, seq_k]
// K: [batch, heads, seq_k, head_dim]
// grad_Q: [batch, heads, seq_q, head_dim]

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
    int head_idx = batch_head % num_heads;

    if (batch_idx >= batch_size || q_pos >= seq_len_q || d >= head_dim) return;

    int score_row_offset = batch_head * seq_len_q * seq_len_k + q_pos * seq_len_k;
    int k_batch_offset = batch_head * seq_len_k * head_dim;

    // grad_Q[q_pos, d] = scale * sum_k grad_scores[q_pos, k] * K[k, d]
    float sum = 0.0f;
    for (int k = 0; k < seq_len_k; k++) {
        float gs, kv;
        if constexpr (sizeof(T) == 4) {
            gs = grad_scores[score_row_offset + k];
            kv = K[k_batch_offset + k * head_dim + d];
        } else if constexpr (sizeof(T) == 2) {
            gs = __half2float(*reinterpret_cast<const __half*>(&grad_scores[score_row_offset + k]));
            kv = __half2float(*reinterpret_cast<const __half*>(&K[k_batch_offset + k * head_dim + d]));
        }
        sum += gs * kv;
    }

    sum *= scale;

    int out_idx = batch_head * seq_len_q * head_dim + q_pos * head_dim + d;
    if constexpr (sizeof(T) == 4) {
        grad_Q[out_idx] = sum;
    } else if constexpr (sizeof(T) == 2) {
        *reinterpret_cast<__half*>(&grad_Q[out_idx]) = __float2half(sum);
    }
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. K
// ============================================================================

// grad_K = scale * grad_scores^T @ Q
// grad_scores^T: [batch, heads, seq_k, seq_q]
// Q: [batch, heads, seq_q, head_dim]
// grad_K: [batch, heads, seq_k, head_dim]

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
    int head_idx = batch_head % num_heads;

    if (batch_idx >= batch_size || k_pos >= seq_len_k || d >= head_dim) return;

    int score_batch_offset = batch_head * seq_len_q * seq_len_k;
    int q_batch_offset = batch_head * seq_len_q * head_dim;

    // grad_K[k_pos, d] = scale * sum_q grad_scores[q, k_pos] * Q[q, d]
    float sum = 0.0f;
    for (int q = 0; q < seq_len_q; q++) {
        float gs, qv;
        if constexpr (sizeof(T) == 4) {
            gs = grad_scores[score_batch_offset + q * seq_len_k + k_pos];
            qv = Q[q_batch_offset + q * head_dim + d];
        } else if constexpr (sizeof(T) == 2) {
            gs = __half2float(*reinterpret_cast<const __half*>(&grad_scores[score_batch_offset + q * seq_len_k + k_pos]));
            qv = __half2float(*reinterpret_cast<const __half*>(&Q[q_batch_offset + q * head_dim + d]));
        }
        sum += gs * qv;
    }

    sum *= scale;

    int out_idx = batch_head * seq_len_k * head_dim + k_pos * head_dim + d;
    if constexpr (sizeof(T) == 4) {
        grad_K[out_idx] = sum;
    } else if constexpr (sizeof(T) == 2) {
        *reinterpret_cast<__half*>(&grad_K[out_idx]) = __float2half(sum);
    }
}

// ============================================================================
// Fused ALiBi Attention Backward - Gradient w.r.t. V
// ============================================================================

// grad_V = probs^T @ grad_output
// probs^T: [batch, heads, seq_k, seq_q]
// grad_output: [batch, heads, seq_q, head_dim]
// grad_V: [batch, heads, seq_k, head_dim]

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
    int head_idx = batch_head % num_heads;

    if (batch_idx >= batch_size || k_pos >= seq_len_k || d >= head_dim) return;

    int prob_batch_offset = batch_head * seq_len_q * seq_len_k;
    int grad_out_batch_offset = batch_head * seq_len_q * head_dim;

    // grad_V[k_pos, d] = sum_q probs[q, k_pos] * grad_output[q, d]
    float sum = 0.0f;
    for (int q = 0; q < seq_len_q; q++) {
        float p, go;
        if constexpr (sizeof(T) == 4) {
            p = probs[prob_batch_offset + q * seq_len_k + k_pos];
            go = grad_output[grad_out_batch_offset + q * head_dim + d];
        } else if constexpr (sizeof(T) == 2) {
            p = __half2float(*reinterpret_cast<const __half*>(&probs[prob_batch_offset + q * seq_len_k + k_pos]));
            go = __half2float(*reinterpret_cast<const __half*>(&grad_output[grad_out_batch_offset + q * head_dim + d]));
        }
        sum += p * go;
    }

    int out_idx = batch_head * seq_len_k * head_dim + k_pos * head_dim + d;
    if constexpr (sizeof(T) == 4) {
        grad_V[out_idx] = sum;
    } else if constexpr (sizeof(T) == 2) {
        *reinterpret_cast<__half*>(&grad_V[out_idx]) = __float2half(sum);
    }
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
    // BF16 uses same logic as F16, reinterpret as __half for template
    alibi_softmax_backward_impl<__half>(
        reinterpret_cast<const __half*>(grad_probs),
        reinterpret_cast<const __half*>(probs),
        reinterpret_cast<__half*>(grad_scores),
        batch_size, num_heads, seq_len_q, seq_len_k
    );
}

extern "C" __global__ void alibi_backward_grad_q_bf16(
    const __nv_bfloat16* grad_scores, const __nv_bfloat16* K, __nv_bfloat16* grad_Q,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_q_impl<__half>(
        reinterpret_cast<const __half*>(grad_scores),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<__half*>(grad_Q),
        scale, batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_k_bf16(
    const __nv_bfloat16* grad_scores, const __nv_bfloat16* Q, __nv_bfloat16* grad_K,
    float scale, int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_k_impl<__half>(
        reinterpret_cast<const __half*>(grad_scores),
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<__half*>(grad_K),
        scale, batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

extern "C" __global__ void alibi_backward_grad_v_bf16(
    const __nv_bfloat16* probs, const __nv_bfloat16* grad_output, __nv_bfloat16* grad_V,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim
) {
    alibi_backward_grad_v_impl<__half>(
        reinterpret_cast<const __half*>(probs),
        reinterpret_cast<const __half*>(grad_output),
        reinterpret_cast<__half*>(grad_V),
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim
    );
}

