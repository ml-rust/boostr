// Scaled Dot-Product Attention (SDPA) - Fused Kernel
// Standard O(N²) attention but fused into a single kernel
// Used for MLA where K and V have different last dimensions
//
// Forward pass:
//   scores = Q @ K^T / sqrt(d_k)
//   attn = softmax(scores, dim=-1)
//   output = attn @ V
//
// Layout:
//   Q: [B, H, S_q, D_k]
//   K: [B, H, S_kv, D_k]  (note: D_k can differ from standard attention)
//   V: [B, H, S_kv, D_v]  (D_v can differ from D_k)
//   Output: [B, H, S_q, D_v]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// Configuration
#define BLOCK_M 128  // Number of Q rows per thread block
#define BLOCK_N 128  // Number of K/V columns (KV pairs) per iteration

// ============================================================================
// SDPA Forward - FP32
// ============================================================================

// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M), 1)
// Block: (BLOCK_M, 1, 1)
// Each thread block processes one (batch, head) pair and iterates over K/V blocks
extern "C" __global__ void sdpa_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim_k,
    const int head_dim_v,
    const float scale,
    const int causal
) {
    // Shared memory for loading Q, K, V tiles
    extern __shared__ char smem[];
    float* Q_smem = (float*)smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim_k;
    float* V_smem = K_smem + BLOCK_N * head_dim_k;

    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Base pointers for this (batch, head) pair
    const int head_offset_q = batch_idx * (num_heads * seq_len_q * head_dim_k) +
                              head_idx * (seq_len_q * head_dim_k);
    const int head_offset_kv = batch_idx * (num_heads * seq_len_k * head_dim_k) +
                               head_idx * (seq_len_k * head_dim_k);
    const int head_offset_v = batch_idx * (num_heads * seq_len_k * head_dim_v) +
                              head_idx * (seq_len_k * head_dim_v);
    const int head_offset_o = batch_idx * (num_heads * seq_len_q * head_dim_v) +
                              head_idx * (seq_len_q * head_dim_v);

    const float* Q_base = Q + head_offset_q;
    const float* K_base = K + head_offset_kv;
    const float* V_base = V + head_offset_v;
    float* O_base = O + head_offset_o;

    const int tid = threadIdx.x;

    // Load Q tile into shared memory
    for (int i = tid; i < q_tile_size * head_dim_k; i += blockDim.x) {
        const int row = i / head_dim_k;
        const int col = i % head_dim_k;
        Q_smem[row * head_dim_k + col] = Q_base[(q_start + row) * head_dim_k + col];
    }
    __syncthreads();

    // Initialize output for this thread
    const bool is_valid_thread = (tid < q_tile_size);
    float O_local[256];  // Max head_dim_v = 256
    float m_local = -INFINITY;
    float l_local = 0.0f;

    #pragma unroll
    for (int d = 0; d < head_dim_v; ++d) {
        if (d < head_dim_v) {
            O_local[d] = 0.0f;
        }
    }

    // Process K/V blocks
    const int num_kv_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles
        for (int i = tid; i < k_tile_size * head_dim_k; i += blockDim.x) {
            const int row = i / head_dim_k;
            const int col = i % head_dim_k;
            K_smem[row * head_dim_k + col] = K_base[(k_start + row) * head_dim_k + col];
        }
        for (int i = tid; i < k_tile_size * head_dim_v; i += blockDim.x) {
            const int row = i / head_dim_v;
            const int col = i % head_dim_v;
            V_smem[row * head_dim_v + col] = V_base[(k_start + row) * head_dim_v + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: find max with online softmax
            float m_new = m_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                // Causal mask
                if (causal && q_pos < k_pos) continue;

                // Compute Q @ K^T
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                m_new = fmaxf(m_new, score);
            }

            // Update O with exp(m_old - m_new)
            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < head_dim_v; ++d) {
                if (d < head_dim_v) {
                    O_local[d] *= alpha;
                }
            }

            // Second pass: accumulate attention weights and V
            float l_new = alpha * l_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                // Causal mask
                if (causal && q_pos < k_pos) continue;

                // Compute score
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                // Softmax weight
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                // Accumulate V
                #pragma unroll
                for (int d = 0; d < head_dim_v; ++d) {
                    if (d < head_dim_v) {
                        O_local[d] += exp_score * V_smem[k_idx * head_dim_v + d];
                    }
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    // Finalize: divide by l_local (normalization)
    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        #pragma unroll
        for (int d = 0; d < head_dim_v; ++d) {
            if (d < head_dim_v) {
                O_base[(q_start + tid) * head_dim_v + d] = O_local[d] * inv_l;
            }
        }
    }
}

// ============================================================================
// SDPA Forward - FP16
// ============================================================================

extern "C" __global__ void sdpa_f16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim_k,
    const int head_dim_v,
    const float scale,
    const int causal
) {
    // Shared memory for loading tiles (store as float for computation)
    extern __shared__ char smem[];
    float* Q_smem = (float*)smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim_k;
    float* V_smem = K_smem + BLOCK_N * head_dim_k;

    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Base pointers
    const int head_offset_q = batch_idx * (num_heads * seq_len_q * head_dim_k) +
                              head_idx * (seq_len_q * head_dim_k);
    const int head_offset_kv = batch_idx * (num_heads * seq_len_k * head_dim_k) +
                               head_idx * (seq_len_k * head_dim_k);
    const int head_offset_v = batch_idx * (num_heads * seq_len_k * head_dim_v) +
                              head_idx * (seq_len_k * head_dim_v);
    const int head_offset_o = batch_idx * (num_heads * seq_len_q * head_dim_v) +
                              head_idx * (seq_len_q * head_dim_v);

    const __half* Q_base = Q + head_offset_q;
    const __half* K_base = K + head_offset_kv;
    const __half* V_base = V + head_offset_v;
    __half* O_base = O + head_offset_o;

    const int tid = threadIdx.x;

    // Load Q tile and convert to float
    for (int i = tid; i < q_tile_size * head_dim_k; i += blockDim.x) {
        const int row = i / head_dim_k;
        const int col = i % head_dim_k;
        Q_smem[row * head_dim_k + col] = __half2float(Q_base[(q_start + row) * head_dim_k + col]);
    }
    __syncthreads();

    // Initialize output
    const bool is_valid_thread = (tid < q_tile_size);
    float O_local[256];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    for (int d = 0; d < head_dim_v; ++d) {
        if (d < head_dim_v) {
            O_local[d] = 0.0f;
        }
    }

    // Process K/V blocks
    const int num_kv_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V, convert to float
        for (int i = tid; i < k_tile_size * head_dim_k; i += blockDim.x) {
            const int row = i / head_dim_k;
            const int col = i % head_dim_k;
            K_smem[row * head_dim_k + col] = __half2float(K_base[(k_start + row) * head_dim_k + col]);
        }
        for (int i = tid; i < k_tile_size * head_dim_v; i += blockDim.x) {
            const int row = i / head_dim_v;
            const int col = i % head_dim_v;
            V_smem[row * head_dim_v + col] = __half2float(V_base[(k_start + row) * head_dim_v + col]);
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: find max
            float m_new = m_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                if (causal && q_pos < k_pos) continue;

                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);
            for (int d = 0; d < head_dim_v; ++d) {
                if (d < head_dim_v) {
                    O_local[d] *= alpha;
                }
            }

            // Second pass: accumulate
            float l_new = alpha * l_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                if (causal && q_pos < k_pos) continue;

                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                for (int d = 0; d < head_dim_v; ++d) {
                    if (d < head_dim_v) {
                        O_local[d] += exp_score * V_smem[k_idx * head_dim_v + d];
                    }
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    // Finalize and convert back to FP16
    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        for (int d = 0; d < head_dim_v; ++d) {
            if (d < head_dim_v) {
                O_base[(q_start + tid) * head_dim_v + d] = __float2half(O_local[d] * inv_l);
            }
        }
    }
}

// ============================================================================
// SDPA Forward - BF16
// ============================================================================

extern "C" __global__ void sdpa_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim_k,
    const int head_dim_v,
    const float scale,
    const int causal
) {
    extern __shared__ char smem[];
    float* Q_smem = (float*)smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim_k;
    float* V_smem = K_smem + BLOCK_N * head_dim_k;

    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    const int head_offset_q = batch_idx * (num_heads * seq_len_q * head_dim_k) +
                              head_idx * (seq_len_q * head_dim_k);
    const int head_offset_kv = batch_idx * (num_heads * seq_len_k * head_dim_k) +
                               head_idx * (seq_len_k * head_dim_k);
    const int head_offset_v = batch_idx * (num_heads * seq_len_k * head_dim_v) +
                              head_idx * (seq_len_k * head_dim_v);
    const int head_offset_o = batch_idx * (num_heads * seq_len_q * head_dim_v) +
                              head_idx * (seq_len_q * head_dim_v);

    const __nv_bfloat16* Q_base = Q + head_offset_q;
    const __nv_bfloat16* K_base = K + head_offset_kv;
    const __nv_bfloat16* V_base = V + head_offset_v;
    __nv_bfloat16* O_base = O + head_offset_o;

    const int tid = threadIdx.x;

    // Load Q and convert to float
    for (int i = tid; i < q_tile_size * head_dim_k; i += blockDim.x) {
        const int row = i / head_dim_k;
        const int col = i % head_dim_k;
        Q_smem[row * head_dim_k + col] = __bfloat162float(Q_base[(q_start + row) * head_dim_k + col]);
    }
    __syncthreads();

    const bool is_valid_thread = (tid < q_tile_size);
    float O_local[256];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    for (int d = 0; d < head_dim_v; ++d) {
        if (d < head_dim_v) {
            O_local[d] = 0.0f;
        }
    }

    const int num_kv_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        for (int i = tid; i < k_tile_size * head_dim_k; i += blockDim.x) {
            const int row = i / head_dim_k;
            const int col = i % head_dim_k;
            K_smem[row * head_dim_k + col] = __bfloat162float(K_base[(k_start + row) * head_dim_k + col]);
        }
        for (int i = tid; i < k_tile_size * head_dim_v; i += blockDim.x) {
            const int row = i / head_dim_v;
            const int col = i % head_dim_v;
            V_smem[row * head_dim_v + col] = __bfloat162float(V_base[(k_start + row) * head_dim_v + col]);
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                if (causal && q_pos < k_pos) continue;

                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);
            for (int d = 0; d < head_dim_v; ++d) {
                if (d < head_dim_v) {
                    O_local[d] *= alpha;
                }
            }

            float l_new = alpha * l_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                if (causal && q_pos < k_pos) continue;

                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                for (int d = 0; d < head_dim_v; ++d) {
                    if (d < head_dim_v) {
                        O_local[d] += exp_score * V_smem[k_idx * head_dim_v + d];
                    }
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        for (int d = 0; d < head_dim_v; ++d) {
            if (d < head_dim_v) {
                O_base[(q_start + tid) * head_dim_v + d] = __float2bfloat16(O_local[d] * inv_l);
            }
        }
    }
}
