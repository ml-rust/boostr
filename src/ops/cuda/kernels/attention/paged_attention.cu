// Paged Attention - vLLM-style non-contiguous KV cache
// Based on "Efficient Memory Management for Large Language Model Serving with PagedAttention"
// Kwon et al., 2023 (https://arxiv.org/abs/2309.06180)
//
// Key features:
// 1. Block table indirection for non-contiguous KV cache storage
// 2. Eliminates memory fragmentation and copying
// 3. 2-3x memory efficiency vs contiguous cache
// 4. Supports variable sequence lengths without padding
//
// Causal convention: ABSOLUTE (bottom-right) alignment. The block table indexes
// keys by their absolute position in the sequence, and the seq_len_q query rows
// are the LAST positions of that seq_len_k context, so query row r sits at
// absolute position key_offset + r with key_offset = seq_len_k - seq_len_q.
// Key j is masked when j > key_offset + r. A full prefill
// (seq_len_q == seq_len_k) gives key_offset == 0, leaving the rule identical to
// the previous top-left form. Same convention as
// `ops/impl_generic/attention/flash_standard.rs::build_attention_mask` and
// `kernels/attention/flash_v2.cu`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "paged_attention.cuh"

// ============================================================================
// FP32 Paged Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_fwd_fp32_impl(
    const float* __restrict__ Q,           // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ K_blocks,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ V_blocks,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table,   // [batch, max_num_blocks]
    float* __restrict__ O,                 // [batch, num_heads, seq_len_q, head_dim]
    float* __restrict__ L,                 // [batch, num_heads, seq_len_q]
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
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
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

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
    // Absolute (bottom-right) causal alignment — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

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

            // Use block table to find physical location (with GQA head mapping)
            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size,
                num_kv_heads, kv_head_idx, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[kv_offset + col];
            V_smem(row, col) = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
    const float* Q, const float* K_blocks, const float* V_blocks,
    const int* block_table, float* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp32(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const int* block_table, float* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
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
    const int num_kv_heads,
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
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __half* Q_base = Q + head_offset;
    __half* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;
    // Absolute (bottom-right) causal alignment — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

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

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size,
                num_kv_heads, kv_head_idx, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[kv_offset + col];
            V_smem(row, col) = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp16(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
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
    const int num_kv_heads,
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
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __nv_bfloat16* Q_base = Q + head_offset;
    __nv_bfloat16* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;
    // Absolute (bottom-right) causal alignment — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

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

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size,
                num_kv_heads, kv_head_idx, HEAD_DIM
            );

            K_smem(row, col) = K_blocks[kv_offset + col];
            V_smem(row, col) = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
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
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp32_small(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const int* block_table, float* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp32_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// FP16 small variants
extern "C" __global__ void paged_flash_attention_fwd_64_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const int* block_table, __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_fp16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// BF16 small variants
extern "C" __global__ void paged_flash_attention_fwd_64_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_fwd_128_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const int* block_table, __nv_bfloat16* O, float* L,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len_q, int seq_len_k, int max_num_blocks,
    int block_size, float scale, int causal
) {
    paged_flash_attention_fwd_bf16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, block_table, O, L,
        batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

