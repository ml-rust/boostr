// Paged Attention - vLLM-style non-contiguous KV cache
// Based on "Efficient Memory Management for Large Language Model Serving with PagedAttention"
// Kwon et al., 2023 (https://arxiv.org/abs/2309.06180)
//
// Key features:
// 1. Block table indirection for non-contiguous KV cache storage
// 2. Eliminates memory fragmentation and copying
// 3. 2-3x memory efficiency vs contiguous cache
// 4. Supports variable sequence lengths without padding

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Warp-level Primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Paged KV Cache Indexing Helper
// ============================================================================

// Compute physical address for a KV token using block table
// block_table: [batch_size, max_num_blocks] - logical to physical block mapping
// token_idx: Logical token index within sequence
// block_size: Number of tokens per block (typically 16)
// head_dim: Dimension of each head
// Returns: Physical offset into K_blocks or V_blocks array
__device__ __forceinline__ int get_paged_kv_offset(
    const int* __restrict__ block_table,
    int batch_idx,
    int max_num_blocks,
    int token_idx,
    int block_size,
    int head_dim
) {
    int logical_block = token_idx / block_size;
    int block_offset = token_idx % block_size;
    int physical_block = block_table[batch_idx * max_num_blocks + logical_block];
    return physical_block * block_size * head_dim + block_offset * head_dim;
}

// ============================================================================
// FP32 Paged Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_fp32_impl(
    const float* __restrict__ Q,           // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ K_blocks,    // [num_blocks, block_size, head_dim] - non-contiguous
    const float* __restrict__ V_blocks,    // [num_blocks, block_size, head_dim] - non-contiguous
    const int* __restrict__ block_table,   // [batch, max_num_blocks]
    float* __restrict__ O,                 // [batch, num_heads, seq_len_q, head_dim]
    float* __restrict__ L,                 // [batch, num_heads, seq_len_q]
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float scale,
    const int causal
) {
    extern __shared__ float smem[];

    // Partition shared memory
    float* Q_smem_flat = smem;
    float* K_smem_flat = smem + BLOCK_M * HEAD_DIM;
    float* V_smem_flat = smem + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // Base pointers for this (batch, head)
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const float* Q_base = Q + head_offset;
    float* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    // Q tile indices
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile into shared memory
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread processes one Q row
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Per-thread accumulation
    float O_local[HEAD_DIM];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        O_local[d] = 0.0f;
    }

    // Iterate over K/V tiles (using paged indexing)
    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles from paged blocks into shared memory
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            // Use block table to find physical location
            const int k_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );
            const int v_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[k_offset + col];
            V_smem(row, col) = V_blocks[v_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                m_new = fmaxf(m_new, score);
            }

            // Rescale previous output
            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate weighted values
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
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

    // Final normalization and write output
    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[out_row * HEAD_DIM + d] = O_local[d] * inv_l;
        }

        // Write logsumexp for backward pass
        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// Kernel Entry Points - HEAD_DIM=64, BLOCK_M=128, BLOCK_N=64
// ============================================================================

extern "C" __global__ void paged_flash_attention_fwd_64_fp32(
    const float* Q,
    const float* K_blocks,
    const float* V_blocks,
    const int* block_table,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int max_num_blocks,
    int block_size,
    float scale,
    int causal
) {
    paged_flash_attention_fwd_fp32_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp32(
    const float* Q,
    const float* K_blocks,
    const float* V_blocks,
    const int* block_table,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int max_num_blocks,
    int block_size,
    float scale,
    int causal
) {
    paged_flash_attention_fwd_fp32_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// FP16 Paged Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_blocks,
    const __half* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    __half* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float scale,
    const int causal
) {
    extern __shared__ __half smem_fp16[];

    __half* Q_smem_flat = smem_fp16;
    __half* K_smem_flat = smem_fp16 + BLOCK_M * HEAD_DIM;
    __half* V_smem_flat = smem_fp16 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __half* Q_base = Q + head_offset;
    __half* O_base = O + head_offset;
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

        // Load K and V from paged blocks
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int k_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );
            const int v_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[k_offset + col];
            V_smem(row, col) = V_blocks[v_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(Q_smem(q_row, d)) * __half2float(K_smem(j, d));
                }
                score *= scale;
                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(Q_smem(q_row, d)) * __half2float(K_smem(j, d));
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * __half2float(V_smem(j, d));
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[out_row * HEAD_DIM + d] = __float2half(O_local[d] * inv_l);
        }

        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

extern "C" __global__ void paged_flash_attention_fwd_64_fp16(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp16(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// BF16 Paged Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_blocks,
    const __nv_bfloat16* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float scale,
    const int causal
) {
    extern __shared__ __nv_bfloat16 smem_bf16[];

    __nv_bfloat16* Q_smem_flat = smem_bf16;
    __nv_bfloat16* K_smem_flat = smem_bf16 + BLOCK_M * HEAD_DIM;
    __nv_bfloat16* V_smem_flat = smem_bf16 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __nv_bfloat16* Q_base = Q + head_offset;
    __nv_bfloat16* O_base = O + head_offset;
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

        // Load K and V from paged blocks
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int k_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );
            const int v_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[k_offset + col];
            V_smem(row, col) = V_blocks[v_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __bfloat162float(Q_smem(q_row, d)) * __bfloat162float(K_smem(j, d));
                }
                score *= scale;
                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __bfloat162float(Q_smem(q_row, d)) * __bfloat162float(K_smem(j, d));
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * __bfloat162float(V_smem(j, d));
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[out_row * HEAD_DIM + d] = __float2bfloat16(O_local[d] * inv_l);
        }

        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

extern "C" __global__ void paged_flash_attention_fwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// Small Block Size Variants (fit in 48KB shared memory)
// BLOCK_M=64, BLOCK_N=32 for head_dim=64; BLOCK_M=32, BLOCK_N=32 for head_dim=128
// ============================================================================

// FP32 small variants
extern "C" __global__ void paged_flash_attention_fwd_64_fp32_small(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const int* block_table, float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp32_small(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const int* block_table, float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// FP16 small variants
extern "C" __global__ void paged_flash_attention_fwd_64_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// BF16 small variants
extern "C" __global__ void paged_flash_attention_fwd_64_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// FP8 E4M3 Paged Flash Attention Forward
// ============================================================================
// FP8 E4M3 is the preferred format for inference (higher precision for activations)
// Uses per-tensor scale factors for dequantization
// Requires Ampere (sm_80) or newer GPUs

#if __CUDA_ARCH__ >= 800

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_fp8_e4m3_impl(
    const boostr_fp8_e4m3* __restrict__ Q,           // FP8 E4M3 [batch, num_heads, seq_len_q, head_dim]
    const boostr_fp8_e4m3* __restrict__ K_blocks,    // FP8 E4M3 [num_blocks, block_size, head_dim]
    const boostr_fp8_e4m3* __restrict__ V_blocks,    // FP8 E4M3 [num_blocks, block_size, head_dim]
    const int* __restrict__ block_table,
    boostr_fp8_e4m3* __restrict__ O,                 // FP8 E4M3 output
    float* __restrict__ L,                   // FP32 logsumexp
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float attn_scale,                  // 1/sqrt(head_dim)
    const float q_scale,                     // Q dequantization scale
    const float k_scale,                     // K dequantization scale
    const float v_scale,                     // V dequantization scale
    const float o_scale,                     // O quantization scale
    const int causal
) {
    // Use FP32 shared memory for accumulation (FP8 is for storage only)
    extern __shared__ float smem_fp8[];

    float* Q_smem_flat = smem_fp8;
    float* K_smem_flat = smem_fp8 + BLOCK_M * HEAD_DIM;
    float* V_smem_flat = smem_fp8 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const boostr_fp8_e4m3* Q_base = Q + head_offset;
    boostr_fp8_e4m3* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile and dequantize to FP32
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        boostr_fp8_e4m3 q_fp8 = Q_base[(q_start + row) * HEAD_DIM + col];
        Q_smem(row, col) = fp8_e4m3_to_f32((uint8_t)q_fp8, q_scale);
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

        // Load K and V from paged blocks, dequantize
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            boostr_fp8_e4m3 k_fp8 = K_blocks[kv_offset + col];
            boostr_fp8_e4m3 v_fp8 = V_blocks[kv_offset + col];
            K_smem(row, col) = fp8_e4m3_to_f32((uint8_t)k_fp8, k_scale);
            V_smem(row, col) = fp8_e4m3_to_f32((uint8_t)v_fp8, v_scale);
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= attn_scale;
                m_new = fmaxf(m_new, score);
            }

            // Rescale previous output
            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate weighted values
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= attn_scale;
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

    // Final normalization and quantize output
    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float o_val = O_local[d] * inv_l;
            uint8_t fp8_val = f32_to_fp8_e4m3_raw(o_val, o_scale);
            O_base[out_row * HEAD_DIM + d] = boostr_fp8_e4m3(fp8_val);
        }

        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// FP8 E4M3 kernel entry points - use larger blocks since FP8 uses FP32 for compute
// Shared memory: (BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 4 bytes (FP32)
// head_dim=64, BLOCK_M=64, BLOCK_N=32: (64+64)*64*4 = 32KB
// head_dim=128, BLOCK_M=32, BLOCK_N=32: (32+64)*128*4 = 49KB

extern "C" __global__ void paged_flash_attention_fwd_64_fp8_e4m3_small(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K_blocks, const boostr_fp8_e4m3* V_blocks,
    const int* block_table, boostr_fp8_e4m3* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float attn_scale,
    float q_scale, float k_scale, float v_scale, float o_scale, int causal
) {
    paged_flash_attention_fwd_fp8_e4m3_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, attn_scale,
        q_scale, k_scale, v_scale, o_scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp8_e4m3_small(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K_blocks, const boostr_fp8_e4m3* V_blocks,
    const int* block_table, boostr_fp8_e4m3* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float attn_scale,
    float q_scale, float k_scale, float v_scale, float o_scale, int causal
) {
    paged_flash_attention_fwd_fp8_e4m3_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, attn_scale,
        q_scale, k_scale, v_scale, o_scale, causal
    );
}

// ============================================================================
// FP8 E5M2 Paged Flash Attention Forward
// ============================================================================
// FP8 E5M2 has larger dynamic range (5 exp bits) but less precision (2 mant bits)
// Better suited for gradients and values with larger variance

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_fp8_e5m2_impl(
    const boostr_fp8_e5m2* __restrict__ Q,
    const boostr_fp8_e5m2* __restrict__ K_blocks,
    const boostr_fp8_e5m2* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    boostr_fp8_e5m2* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float attn_scale,
    const float q_scale,
    const float k_scale,
    const float v_scale,
    const float o_scale,
    const int causal
) {
    extern __shared__ float smem_fp8_e5m2[];

    float* Q_smem_flat = smem_fp8_e5m2;
    float* K_smem_flat = smem_fp8_e5m2 + BLOCK_M * HEAD_DIM;
    float* V_smem_flat = smem_fp8_e5m2 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem_e5(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem_e5(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem_e5(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const boostr_fp8_e5m2* Q_base = Q + head_offset;
    boostr_fp8_e5m2* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile and dequantize
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        boostr_fp8_e5m2 q_fp8 = Q_base[(q_start + row) * HEAD_DIM + col];
        Q_smem_e5(row, col) = fp8_e5m2_to_f32((uint8_t)q_fp8, q_scale);
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

        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            boostr_fp8_e5m2 k_fp8 = K_blocks[kv_offset + col];
            boostr_fp8_e5m2 v_fp8 = V_blocks[kv_offset + col];
            K_smem_e5(row, col) = fp8_e5m2_to_f32((uint8_t)k_fp8, k_scale);
            V_smem_e5(row, col) = fp8_e5m2_to_f32((uint8_t)v_fp8, v_scale);
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem_e5(q_row, d) * K_smem_e5(j, d);
                }
                score *= attn_scale;
                m_new = fmaxf(m_new, score);
            }

            const float alpha = __expf(m_local - m_new);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem_e5(q_row, d) * K_smem_e5(j, d);
                }
                score *= attn_scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * V_smem_e5(j, d);
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float o_val = O_local[d] * inv_l;
            uint8_t fp8_val = f32_to_fp8_e5m2_raw(o_val, o_scale);
            O_base[out_row * HEAD_DIM + d] = boostr_fp8_e5m2(fp8_val);
        }

        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem_e5
    #undef K_smem_e5
    #undef V_smem_e5
}

extern "C" __global__ void paged_flash_attention_fwd_64_fp8_e5m2_small(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K_blocks, const boostr_fp8_e5m2* V_blocks,
    const int* block_table, boostr_fp8_e5m2* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float attn_scale,
    float q_scale, float k_scale, float v_scale, float o_scale, int causal
) {
    paged_flash_attention_fwd_fp8_e5m2_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, attn_scale,
        q_scale, k_scale, v_scale, o_scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp8_e5m2_small(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K_blocks, const boostr_fp8_e5m2* V_blocks,
    const int* block_table, boostr_fp8_e5m2* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float attn_scale,
    float q_scale, float k_scale, float v_scale, float o_scale, int causal
) {
    paged_flash_attention_fwd_fp8_e5m2_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, attn_scale,
        q_scale, k_scale, v_scale, o_scale, causal
    );
}

#endif // __CUDA_ARCH__ >= 800
