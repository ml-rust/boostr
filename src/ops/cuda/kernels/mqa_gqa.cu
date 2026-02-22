// Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
// Based on Flash Attention v2 with KV head broadcasting
//
// MQA: 1 KV head shared across all Q heads (Llama 2, PaLM)
// GQA: Multiple KV heads, each shared across a group of Q heads (Llama 3, Mistral)
//
// Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
// https://arxiv.org/abs/2305.13245
//
// Key differences from standard MHA:
// 1. kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads)
// 2. Same Flash V2 algorithm, just different head indexing
// 3. Same performance characteristics as MHA
//
// Shared memory padding strategy:
// - Custom padding per dimension (default: +1 for head dims to avoid bank conflicts)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

#define SMEM_STRIDE(dim, pad) ((dim) + (pad))

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
// FP32 MQA/GQA Implementation
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_fwd_fp32_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
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

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;

    // KEY: GQA/MQA head mapping
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    // Q uses num_q_heads, K/V use num_kv_heads
    const int q_offset = batch_idx * num_q_heads * seq_len_q * HEAD_DIM
                       + q_head_idx * seq_len_q * HEAD_DIM;
    const int kv_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                        + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_q_heads * seq_len_q + q_head_idx * seq_len_q;

    const float* Q_base = Q + q_offset;
    const float* K_base = K + kv_offset;
    const float* V_base = V + kv_offset;
    float* O_base = O + q_offset;
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

    // Per-thread accumulation
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
// FP16 MQA/GQA Implementation
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_fwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    extern __shared__ __half smem_fp16[];

    __half* Q_smem_flat = smem_fp16;
    __half* K_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE;
    __half* V_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int q_offset = batch_idx * num_q_heads * seq_len_q * HEAD_DIM
                       + q_head_idx * seq_len_q * HEAD_DIM;
    const int kv_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                        + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_q_heads * seq_len_q + q_head_idx * seq_len_q;

    const __half* Q_base = Q + q_offset;
    const __half* K_base = K + kv_offset;
    const __half* V_base = V + kv_offset;
    __half* O_base = O + q_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // FP32 accumulation
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
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
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
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[(q_start + q_row) * HEAD_DIM + d] = __float2half(O_local[d] * inv_l);
        }

        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// BF16 MQA/GQA Implementation
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_fwd_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    extern __shared__ __nv_bfloat16 smem_bf16[];

    __nv_bfloat16* Q_smem_flat = smem_bf16;
    __nv_bfloat16* K_smem_flat = smem_bf16 + BLOCK_M * HEAD_STRIDE;
    __nv_bfloat16* V_smem_flat = smem_bf16 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int q_offset = batch_idx * num_q_heads * seq_len_q * HEAD_DIM
                       + q_head_idx * seq_len_q * HEAD_DIM;
    const int kv_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                        + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_q_heads * seq_len_q + q_head_idx * seq_len_q;

    const __nv_bfloat16* Q_base = Q + q_offset;
    const __nv_bfloat16* K_base = K + kv_offset;
    const __nv_bfloat16* V_base = V + kv_offset;
    __nv_bfloat16* O_base = O + q_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

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

        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
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
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[(q_start + q_row) * HEAD_DIM + d] = __float2bfloat16(O_local[d] * inv_l);
        }

        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// Kernel Instantiations - FP32
// ============================================================================

extern "C" __global__ void mqa_gqa_fwd_64_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp32_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_128_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp32_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_32_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp32_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// Kernel Instantiations - FP16
// ============================================================================

extern "C" __global__ void mqa_gqa_fwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp16_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp16_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_32_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_fp16_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// Kernel Instantiations - BF16
// ============================================================================

extern "C" __global__ void mqa_gqa_fwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_bf16_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_bf16_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_fwd_32_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_fwd_bf16_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// FP8 MQA/GQA Implementation
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_fwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ Q,
    const boostr_fp8_e4m3* __restrict__ K,
    const boostr_fp8_e4m3* __restrict__ V,
    boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float q_scale,
    const float k_scale,
    const float v_scale,
    const float o_scale
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

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int q_offset = batch_idx * num_q_heads * seq_len_q * HEAD_DIM
                       + q_head_idx * seq_len_q * HEAD_DIM;
    const int kv_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                        + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_q_heads * seq_len_q + q_head_idx * seq_len_q;

    const boostr_fp8_e4m3* Q_base = Q + q_offset;
    const boostr_fp8_e4m3* K_base = K + kv_offset;
    const boostr_fp8_e4m3* V_base = V + kv_offset;
    boostr_fp8_e4m3* O_base = O + q_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile (dequantize FP8 → FP32)
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

        // Load K and V tiles (dequantize FP8 → FP32)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            boostr_fp8_e4m3 k_fp8 = K_base[(k_start + row) * HEAD_DIM + col];
            boostr_fp8_e4m3 v_fp8 = V_base[(k_start + row) * HEAD_DIM + col];
            K_smem(row, col) = fp8_e4m3_to_f32((uint8_t)k_fp8, k_scale);
            V_smem(row, col) = fp8_e4m3_to_f32((uint8_t)v_fp8, v_scale);
        }
        __syncthreads();

        if (is_valid_thread) {
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

            const float m_old = m_local;
            m_local = m_new;
            const float exp_diff = __expf(m_old - m_new);
            l_local *= exp_diff;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= exp_diff;
            }

            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                const float p = __expf(score - m_new);
                l_local += p;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += p * V_smem(j, d);
                }
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float out_val = O_local[d] * inv_l;
            uint8_t fp8_val = f32_to_fp8_e4m3_raw(out_val, o_scale);
            O_base[(q_start + q_row) * HEAD_DIM + d] = boostr_fp8_e4m3(fp8_val);
        }
        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// FP8 E4M3 kernel instantiations
extern "C" __global__ void mqa_gqa_fwd_64_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void mqa_gqa_fwd_128_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void mqa_gqa_fwd_32_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

// FP8 E5M2 kernels (same impl, just different kernel names for routing)
extern "C" __global__ void mqa_gqa_fwd_64_fp8_e5m2(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void mqa_gqa_fwd_128_fp8_e5m2(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void mqa_gqa_fwd_32_fp8_e5m2(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    mqa_gqa_fwd_fp8_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}


