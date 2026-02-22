// ALiBi (Attention with Linear Biases) - Position Encoding Alternative
// Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
// https://arxiv.org/abs/2108.12409
//
// Key innovation: Replace positional encodings with position-dependent bias added to attention scores
// Used in: BLOOM, MPT, Falcon, many models with length extrapolation
//
// Standard attention:
//   scores = Q @ K^T / sqrt(d)
//
// ALiBi attention:
//   scores = Q @ K^T / sqrt(d) + bias
//   bias[i,j] = -m * |i - j|
//   m = head-specific slope (geometric sequence: m_h = 2^(-8h/H))
//
// Benefits:
//   - No positional encodings needed (simpler architecture)
//   - Better length extrapolation (train on 2k, test on 100k+)
//   - Minimal computational overhead (just add bias during attention)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include "dtype_traits.cuh"

// ============================================================================
// ALiBi Bias Computation
// ============================================================================

// Compute head-specific ALiBi slope
// Formula: m_h = 2^(-8h/H) where h = head index, H = total heads
//
// Example (H=8):
//   Head 0: m = 2^0 = 1.0
//   Head 1: m = 2^(-1) = 0.5
//   Head 2: m = 2^(-2) = 0.25
//   ...
//   Head 7: m = 2^(-7) = 0.0078125

__device__ __forceinline__ float get_alibi_slope(int head_idx, int num_heads) {
    return powf(2.0f, -8.0f * head_idx / (float)num_heads);
}

// ============================================================================
// ALiBi Bias Injection - FP32
// ============================================================================

// Add ALiBi bias to attention scores (applied BEFORE softmax)
//
// Args:
//   scores: Attention scores [batch, num_heads, seq_len_q, seq_len_k] (in/out)
//   batch_size, num_heads, seq_len_q, seq_len_k: Tensor dimensions
//
// Notes:
//   - Bias is computed on-the-fly: bias[i,j] = -slope * |i - j|
//   - Each head has a different slope (geometric sequence)
//   - Negative bias means nearby tokens get higher scores
//   - This kernel is called AFTER Q@K^T but BEFORE softmax

__device__ void alibi_add_bias_fp32_impl(
    float* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len_q * seq_len_k;

    if (idx >= total) return;

    // Decompose linear index
    const int k_pos = idx % seq_len_k;
    const int q_pos = (idx / seq_len_k) % seq_len_q;
    const int head_idx = (idx / (seq_len_k * seq_len_q)) % num_heads;
    const int batch_idx = idx / (seq_len_k * seq_len_q * num_heads);

    // Compute ALiBi slope for this head
    const float slope = get_alibi_slope(head_idx, num_heads);

    // Compute position distance (can be negative for causal masking)
    const int distance = q_pos - k_pos;

    // ALiBi bias: -slope * |distance|
    const float bias = -slope * abs(distance);

    // Add bias to attention score
    scores[idx] += bias;
}

extern "C" __global__ void alibi_add_bias_fp32(
    float* scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k
) {
    alibi_add_bias_fp32_impl(scores, batch_size, num_heads, seq_len_q, seq_len_k);
}

// ============================================================================
// ALiBi Bias Injection - FP16
// ============================================================================

__device__ void alibi_add_bias_fp16_impl(
    __half* __restrict__ scores,
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

    // Read current score, add bias, write back
    float score = __half2float(scores[idx]);
    score += bias;
    scores[idx] = __float2half(score);
}

extern "C" __global__ void alibi_add_bias_fp16(
    __half* scores,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k
) {
    alibi_add_bias_fp16_impl(scores, batch_size, num_heads, seq_len_q, seq_len_k);
}

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
// Fused ALiBi Flash Attention - FP32
// ============================================================================

// Flash Attention with integrated ALiBi bias
// Computes attention on-the-fly with ALiBi bias injected during score computation
//
// This is more efficient than separate kernels:
//   1. Compute Q @ K^T
//   2. Add ALiBi bias  <-- This kernel fuses steps 2-4
//   3. Apply softmax
//   4. Multiply by V

#define SMEM_STRIDE(dim, pad) ((dim) + (pad))

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_alibi_fp32_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    extern __shared__ float smem[];

    float* Q_smem_flat = smem;
    float* K_smem_flat = smem + BLOCK_M * HEAD_STRIDE;
    float* V_smem_flat = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // Compute ALiBi slope for this head
    const float alibi_slope = get_alibi_slope(head_idx, num_heads);

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_heads * seq_len_k * HEAD_DIM
                              + head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const float* Q_base = Q + head_offset;
    const float* K_base = K + kv_head_offset;
    const float* V_base = V + kv_head_offset;
    float* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    float O_local[HEAD_DIM];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        O_local[d] = 0.0f;
    }

    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max with ALiBi bias
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                if (causal && q_pos < k_pos) continue;

                // Compute Q @ K^T score
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;

                // Add ALiBi bias: -slope * |q_pos - k_pos|
                const int distance = q_pos - k_pos;
                const float alibi_bias = -alibi_slope * abs(distance);
                score += alibi_bias;

                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate with ALiBi bias
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                if (causal && q_pos < k_pos) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;

                const int distance = q_pos - k_pos;
                const float alibi_bias = -alibi_slope * abs(distance);
                score += alibi_bias;

                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * V_smem(j, d);
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    // Final normalization
    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[(q_start + q_row) * HEAD_DIM + d] = O_local[d] * inv_l;
        }

        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

extern "C" __global__ void flash_attention_alibi_64_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_alibi_fp32_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_alibi_128_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_alibi_fp32_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}


// ============================================================================
// FP8 ALiBi Bias Addition
// ============================================================================

extern "C" __global__ void alibi_add_bias_fp8_e4m3(
    boostr_fp8_e4m3* scores,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float score_scale, const float out_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len_q * seq_len_k;

    if (idx >= total) return;

    const int k_pos = idx % seq_len_k;
    const int q_pos = (idx / seq_len_k) % seq_len_q;
    const int head_idx = (idx / (seq_len_k * seq_len_q)) % num_heads;

    const float slope = powf(2.0f, -8.0f * head_idx / (float)num_heads);
    const int distance = q_pos - k_pos;
    const float alibi_bias = -slope * abs(distance);

    float score = fp8_e4m3_to_f32((uint8_t)scores[idx], score_scale);
    score += alibi_bias;
    scores[idx] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(score, out_scale));
}

extern "C" __global__ void alibi_add_bias_fp8_e5m2(
    boostr_fp8_e5m2* scores,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float score_scale, const float out_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len_q * seq_len_k;

    if (idx >= total) return;

    const int k_pos = idx % seq_len_k;
    const int q_pos = (idx / seq_len_k) % seq_len_q;
    const int head_idx = (idx / (seq_len_k * seq_len_q)) % num_heads;

    const float slope = powf(2.0f, -8.0f * head_idx / (float)num_heads);
    const int distance = q_pos - k_pos;
    const float alibi_bias = -slope * abs(distance);

    float score = fp8_e5m2_to_f32((uint8_t)scores[idx], score_scale);
    score += alibi_bias;
    scores[idx] = boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(score, out_scale));
}


