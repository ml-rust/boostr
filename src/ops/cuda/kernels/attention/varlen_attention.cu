// VarLen (Ragged) Flash Attention - Eliminates padding waste
// Based on Flash Attention v2 with cumulative sequence length indexing
//
// Key features:
// 1. Sequences of different lengths packed into 1D buffer
// 2. 30-50% memory savings by eliminating padding
// 3. Cumulative sequence length indexing (cu_seqlens)
// 4. Efficient for training and inference with variable lengths

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// FP32 VarLen Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void varlen_flash_attention_fwd_fp32_impl(
    const float* __restrict__ Q,           // [total_tokens_q, num_heads, head_dim]
    const float* __restrict__ K,           // [total_tokens_k, num_kv_heads, head_dim]
    const float* __restrict__ V,           // [total_tokens_k, num_kv_heads, head_dim]
    const int* __restrict__ cu_seqlens_q,  // [batch_size + 1]
    const int* __restrict__ cu_seqlens_k,  // [batch_size + 1]
    float* __restrict__ O,                 // [total_tokens_q, num_heads, head_dim]
    float* __restrict__ L,                 // [total_tokens_q, num_heads]
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,                // GQA: num_kv_heads <= num_heads; MHA: == num_heads
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int causal
) {
    // +1 padding on the column stride eliminates 32-way bank conflicts for
    // power-of-2 head dimensions (mirrors flash_v2.cu HEAD_STRIDE = HEAD_DIM+1).
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    extern __shared__ float smem[];

    float* Q_smem_flat = smem;
    float* K_smem_flat = smem + BLOCK_M * HEAD_STRIDE;
    float* V_smem_flat = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x % num_heads;
    const int remaining = blockIdx.x / num_heads;
    // Decode batch and per-batch Q block from grid index
    const int num_q_blocks_per_batch = (max_seqlen_q + BLOCK_M - 1) / BLOCK_M;
    const int batch_idx = remaining / num_q_blocks_per_batch;
    const int q_block_in_batch = remaining % num_q_blocks_per_batch;

    // Bounds check
    if (batch_idx >= batch_size) return;

    // Get sequence boundaries for this batch
    const int seq_start_q = cu_seqlens_q[batch_idx];
    const int seq_end_q = cu_seqlens_q[batch_idx + 1];
    const int seq_len_q = seq_end_q - seq_start_q;

    const int seq_start_k = cu_seqlens_k[batch_idx];
    const int seq_end_k = cu_seqlens_k[batch_idx + 1];
    const int seq_len_k = seq_end_k - seq_start_k;

    // Local Q block position within this sequence
    const int q_start = q_block_in_batch * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Check if this block is within valid range
    if (q_start >= seq_len_q) return;

    // GQA head mapping: each kv head serves (num_heads / num_kv_heads) query heads.
    // kv_head_idx = head_idx / (num_heads / num_kv_heads)
    //             = head_idx * num_kv_heads / num_heads  (integer division)
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;

    // Q and O use full num_heads stride; K and V use num_kv_heads stride.
    const float* Q_head = Q + head_idx * HEAD_DIM;
    const float* K_head = K + kv_head_idx * HEAD_DIM;
    const float* V_head = V + kv_head_idx * HEAD_DIM;
    float* O_head = O + head_idx * HEAD_DIM;

    // Load Q tile into shared memory
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int global_q_pos = seq_start_q + q_start + row;
        Q_smem(row, col) = Q_head[global_q_pos * num_heads * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread processes one Q row
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    float O_local[HEAD_DIM];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        O_local[d] = 0.0f;
    }

    // Iterate over K/V tiles
    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles — K/V row stride uses num_kv_heads (GQA layout)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int global_k_pos = seq_start_k + k_start + row;
            K_smem(row, col) = K_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
            V_smem(row, col) = V_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
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

            // Second pass: accumulate
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

    // Write output
    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int global_out_pos = seq_start_q + q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_head[global_out_pos * num_heads * HEAD_DIM + d] = O_local[d] * inv_l;
        }

        L[global_out_pos * num_heads + head_idx] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// Kernel Entry Points - HEAD_DIM=64, BLOCK_M=128, BLOCK_N=64
// ============================================================================

extern "C" __global__ void varlen_flash_attention_fwd_64_fp32(
    const float* Q,
    const float* K,
    const float* V,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    float scale,
    int causal
) {
    varlen_flash_attention_fwd_fp32_impl<64, 128, 64>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_fwd_128_fp32(
    const float* Q,
    const float* K,
    const float* V,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    float scale,
    int causal
) {
    varlen_flash_attention_fwd_fp32_impl<128, 128, 64>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}

// ============================================================================
// FP16 VarLen Flash Attention Forward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void varlen_flash_attention_fwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    __half* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,                // GQA: num_kv_heads <= num_heads; MHA: == num_heads
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int causal
) {
    // +1 padding on the column stride eliminates 32-way bank conflicts for
    // power-of-2 head dimensions (mirrors flash_v2.cu HEAD_STRIDE = HEAD_DIM+1).
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    extern __shared__ __half smem_fp16[];

    __half* Q_smem_flat = smem_fp16;
    __half* K_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE;
    __half* V_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x % num_heads;
    const int remaining = blockIdx.x / num_heads;
    const int num_q_blocks_per_batch = (max_seqlen_q + BLOCK_M - 1) / BLOCK_M;
    const int batch_idx = remaining / num_q_blocks_per_batch;
    const int q_block_in_batch = remaining % num_q_blocks_per_batch;

    if (batch_idx >= batch_size) return;

    const int seq_start_q = cu_seqlens_q[batch_idx];
    const int seq_end_q = cu_seqlens_q[batch_idx + 1];
    const int seq_len_q = seq_end_q - seq_start_q;

    const int seq_start_k = cu_seqlens_k[batch_idx];
    const int seq_end_k = cu_seqlens_k[batch_idx + 1];
    const int seq_len_k = seq_end_k - seq_start_k;

    const int q_start = q_block_in_batch * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    if (q_start >= seq_len_q) return;

    // GQA head mapping: kv_head_idx = head_idx * num_kv_heads / num_heads
    const int kv_head_idx_fp16 = head_idx * num_kv_heads / num_heads;

    const __half* Q_head = Q + head_idx * HEAD_DIM;
    const __half* K_head = K + kv_head_idx_fp16 * HEAD_DIM;
    const __half* V_head = V + kv_head_idx_fp16 * HEAD_DIM;
    __half* O_head = O + head_idx * HEAD_DIM;

    // Load Q tile
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int global_q_pos = seq_start_q + q_start + row;
        Q_smem(row, col) = Q_head[global_q_pos * num_heads * HEAD_DIM + col];
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

        // K/V row stride uses num_kv_heads (GQA layout)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int global_k_pos = seq_start_k + k_start + row;
            K_smem(row, col) = K_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
            V_smem(row, col) = V_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
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
        const int global_out_pos = seq_start_q + q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_head[global_out_pos * num_heads * HEAD_DIM + d] = __float2half(O_local[d] * inv_l);
        }

        L[global_out_pos * num_heads + head_idx] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

extern "C" __global__ void varlen_flash_attention_fwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads, int max_seqlen_q, int max_seqlen_k,
    float scale, int causal
) {
    varlen_flash_attention_fwd_fp16_impl<64, 128, 64>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_fwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads, int max_seqlen_q, int max_seqlen_k,
    float scale, int causal
) {
    varlen_flash_attention_fwd_fp16_impl<128, 128, 64>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}

// ============================================================================
// Kernel Entry Points - HEAD_DIM=256
//
// head_dim=256 with BLOCK_M=128,BLOCK_N=64 would require ~256 KB smem which
// exceeds all GPU limits.  Use smaller tiles instead:
//   fp32: BLOCK_M=16, BLOCK_N=16 → smem=(16+2*16)*256*4 = 49152 B = 48 KB
//   fp16: BLOCK_M=32, BLOCK_N=32 → smem=(32+2*32)*256*2 = 49152 B = 48 KB
// Both fit within the 48 KB default smem limit (set_smem_attribute raises it
// to the required size with cudaFuncAttributeMaxDynamicSharedMemorySize).
//
// O_local[256] at head_dim=256 will spill to local memory.  This is accepted
// for this CORRECTNESS-FIRST implementation; optimisation is deferred.
//
// BWD for head_dim=256 is NOT instantiated here; the Rust launcher guards
// against it.  Forward-only usage supports inference (EmbeddingGemma).
// ============================================================================

extern "C" __global__ void varlen_flash_attention_fwd_256_fp32(
    const float* Q,
    const float* K,
    const float* V,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    float scale,
    int causal
) {
    varlen_flash_attention_fwd_fp32_impl<256, 16, 16>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_fwd_256_fp16(
    const __half* Q, const __half* K, const __half* V,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* O, float* L,
    int batch_size, int num_heads, int num_kv_heads, int max_seqlen_q, int max_seqlen_k,
    float scale, int causal
) {
    varlen_flash_attention_fwd_fp16_impl<256, 32, 32>(
        Q, K, V, cu_seqlens_q, cu_seqlens_k, O, L,
        batch_size, num_heads, num_kv_heads, max_seqlen_q, max_seqlen_k, scale, causal
    );
}
