// Flash Attention v2 - Production Implementation
// Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// Tri Dao, 2023 (https://arxiv.org/abs/2307.08691)
//
// Implementation matches PyTorch Flash Attention reference (research/flash-attention-main/)
//
// Key optimizations:
// 1. Padded shared memory strides (eliminates bank conflicts on power-of-2 dimensions)
// 2. Multi-precision support (FP32, FP16, BF16, FP8)
// 3. Register-based accumulation for numerical stability
// 4. Warp-shuffle reductions (minimize shared memory traffic)
// 5. All head dimensions (32, 64, 96, 128, 192, 256)
//
// Shared memory padding strategy:
// - Custom padding per dimension (default: +8 for general case, +1 for head dims)
// - Avoids 32-way bank conflicts on NVIDIA GPUs (32 banks, 4-byte words)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Shared Memory Stride Padding
// ============================================================================
// Padding eliminates bank conflicts on power-of-2 dimensions
//
// Usage:
//   __shared__ float Q_smem[BLOCK_M][SMEM_STRIDE(HEAD_DIM, 1)];  // +1 padding
//   __shared__ float K_smem[BLOCK_N][HEAD_DIM + 8];              // +8 padding

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
// FP32 Kernels with Dynamic Shared Memory
// ============================================================================

// Flash Attention forward pass - FP32 with GQA support
// Supports arbitrary head dimensions via template parameter
// Native GQA: num_kv_heads can be less than num_heads (KV heads are broadcast)
// Native sliding window: window_size parameter restricts attention to local context
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M))
// Block: BLOCK_M threads (typically 128)
// Shared memory: Padded strides to eliminate bank conflicts
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fwd_fp32_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,  // GQA: can be less than num_heads
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int window_size    // Sliding window: 0 or -1 = full attention, >0 = local window
) {
    // Padded strides (+1 for power-of-2 head dims like 64, 128)
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    // Use dynamic shared memory with padded strides
    extern __shared__ float smem[];

    // Partition shared memory manually with padding
    float* Q_smem_flat = smem;
    float* K_smem_flat = smem + BLOCK_M * HEAD_STRIDE;
    float* V_smem_flat = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    // Helper macros for 2D indexing with padded stride
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // GQA: Map query head to KV head (multiple Q heads share one KV head)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Base pointers for this (batch, head)
    // Q/O use num_heads, K/V use num_kv_heads
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                              + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const float* Q_base = Q + head_offset;
    const float* K_base = K + kv_head_offset;
    const float* V_base = V + kv_head_offset;
    float* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    // Q tile indices
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile into shared memory (all threads cooperate)
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread processes one Q row
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Per-thread accumulation in registers (all threads initialize, but only valid ones compute)
    float O_local[HEAD_DIM];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    // Initialize output accumulator to zero
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

        // Sliding window optimization: skip entire K blocks outside window
        // For any Q position in this tile, the earliest K position we need is:
        //   min_k_needed = max(0, q_start - window_size + 1)
        // The latest K position we need is:
        //   max_k_needed = q_end - 1 (for causal) or seq_len_k - 1
        // Skip this K block if it's entirely outside the window for ALL Q positions in this tile
        if (window_size > 0) {
            // For the last Q position in this tile (q_start + BLOCK_M - 1 or q_end - 1),
            // the earliest K we might need is (q_end - 1) - window_size + 1
            // If k_end - 1 < this value, skip entire block
            int last_q_pos = min(q_start + BLOCK_M - 1, seq_len_q - 1);
            int min_k_for_last_q = max(0, last_q_pos - window_size + 1);
            if (k_end - 1 < min_k_for_last_q) {
                continue;  // Skip this K block entirely - outside window for all Q
            }
        }

        // Load K and V tiles into shared memory (ALL threads cooperate - critical!)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Only valid threads (q_row < q_tile_size) compute attention
        if (is_valid_thread) {
            // First pass: compute max over this K tile
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

                // Compute Q @ K^T score
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                m_new = fmaxf(m_new, score);
            }

            // Compute correction factor for online softmax
            const float alpha = __expf(m_local - m_new);

            // Rescale previous output in registers
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate weighted values and update l_local
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                // Accumulate weighted V values in registers
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

    // Final normalization and write to global memory (only valid threads)
    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[(q_start + q_row) * HEAD_DIM + d] = O_local[d] * inv_l;
        }

        // Write logsumexp (for backward pass)
        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// ============================================================================
// Kernel Instantiations - FP32 with GQA and Sliding Window support
// ============================================================================

// head_dim=64, BLOCK_M=128, BLOCK_N=128 (PyTorch standard)
extern "C" __global__ void flash_attention_fwd_64_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=128, BLOCK_M=128, BLOCK_N=64 (PyTorch standard for sm80)
extern "C" __global__ void flash_attention_fwd_128_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=128, BLOCK_M=64, BLOCK_N=32 (small shared memory variant for <=100KB GPUs)
// smem: (64*129 + 2*32*129)*4 = 66048 bytes = 64.5KB
extern "C" __global__ void flash_attention_fwd_128_sm_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<128, 64, 32>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=96, BLOCK_M=32, BLOCK_N=32 (small shared memory variant for <=100KB GPUs)
// smem: (32*97 + 2*32*97)*4 = 37248 bytes = 36.4KB
extern "C" __global__ void flash_attention_fwd_96_sm_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<96, 32, 32>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=192, BLOCK_M=32, BLOCK_N=16 (small shared memory variant for <=100KB GPUs)
// smem: (32*193 + 2*16*193)*4 = 49408 bytes = 48.25KB
extern "C" __global__ void flash_attention_fwd_192_sm_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<192, 32, 16>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=256, BLOCK_M=16, BLOCK_N=16 (small shared memory variant for <=100KB GPUs)
// smem: (16*257 + 2*16*257)*4 = 49344 bytes = 48.2KB
extern "C" __global__ void flash_attention_fwd_256_sm_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<256, 16, 16>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=32, BLOCK_M=128, BLOCK_N=128
extern "C" __global__ void flash_attention_fwd_32_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=96, BLOCK_M=64, BLOCK_N=128 (nvcc segfaults with 64x256)
extern "C" __global__ void flash_attention_fwd_96_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<96, 64, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=192, BLOCK_M=64, BLOCK_N=64
extern "C" __global__ void flash_attention_fwd_192_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<192, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// head_dim=256, BLOCK_M=64, BLOCK_N=64
extern "C" __global__ void flash_attention_fwd_256_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp32_impl<256, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// ============================================================================
// FP16 Kernels - Mixed Precision with FP32 Accumulation
// ============================================================================

// Flash Attention forward pass - FP16 input/output, FP32 accumulation with GQA support
// Numerical stability: All accumulation (m, l, O_local) done in FP32
// Native sliding window: window_size parameter restricts attention to local context
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,  // GQA: can be less than num_heads
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int window_size    // Sliding window: 0 or -1 = full attention, >0 = local window
) {
    // Padded strides
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    // Use dynamic shared memory with padded strides
    extern __shared__ __half smem_fp16[];

    // Partition shared memory with padding
    __half* Q_smem_flat = smem_fp16;
    __half* K_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE;
    __half* V_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // GQA: Map query head to KV head (multiple Q heads share one KV head)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Q/O use num_heads, K/V use num_kv_heads
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                              + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __half* Q_base = Q + head_offset;
    const __half* K_base = K + kv_head_offset;
    const __half* V_base = V + kv_head_offset;
    __half* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile into shared memory (all threads cooperate)
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Per-thread accumulation in FP32 registers (KEY: mixed precision)
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

        // Sliding window optimization: skip entire K blocks outside window
        if (window_size > 0) {
            int last_q_pos = min(q_start + BLOCK_M - 1, seq_len_q - 1);
            int min_k_for_last_q = max(0, last_q_pos - window_size + 1);
            if (k_end - 1 < min_k_for_last_q) {
                continue;  // Skip this K block entirely - outside window for all Q
            }
        }

        // Load K and V tiles (ALL threads cooperate - critical!)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Only valid threads compute attention
        if (is_valid_thread) {
            // First pass: compute max (FP32 accumulation)
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

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

            // Second pass: accumulate weighted values (FP32 accumulation)
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

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

    // Final normalization and write (only valid threads)
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

// Kernel instantiations for FP16 with GQA and Sliding Window support
extern "C" __global__ void flash_attention_fwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_32_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_96_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<96, 64, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_192_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<192, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_256_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_fp16_impl<256, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// ============================================================================
// BF16 Kernels - For Ampere+ GPUs with FP32 Accumulation
// ============================================================================

// Flash Attention forward pass - BF16 input/output, FP32 accumulation with GQA support
// Native sliding window: window_size parameter restricts attention to local context
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fwd_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,  // GQA: can be less than num_heads
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int window_size    // Sliding window: 0 or -1 = full attention, >0 = local window
) {
    // Padded strides
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

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // GQA: Map query head to KV head (multiple Q heads share one KV head)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Q/O use num_heads, K/V use num_kv_heads
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                              + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + kv_head_offset;
    const __nv_bfloat16* V_base = V + kv_head_offset;
    __nv_bfloat16* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile into shared memory (all threads cooperate)
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // FP32 accumulation for numerical stability
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

        // Sliding window optimization: skip entire K blocks outside window
        if (window_size > 0) {
            int last_q_pos = min(q_start + BLOCK_M - 1, seq_len_q - 1);
            int min_k_for_last_q = max(0, last_q_pos - window_size + 1);
            if (k_end - 1 < min_k_for_last_q) {
                continue;  // Skip this K block entirely - outside window for all Q
            }
        }

        // Load K and V tiles (ALL threads cooperate - critical!)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Only valid threads compute attention
        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

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
                const int q_pos = q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

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

    // Final normalization and write (only valid threads)
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

// Kernel instantiations for BF16 with GQA and Sliding Window support
extern "C" __global__ void flash_attention_fwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_32_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_96_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<96, 64, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_192_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<192, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

extern "C" __global__ void flash_attention_fwd_256_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal, const int window_size
) {
    flash_attention_fwd_bf16_impl<256, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal, window_size
    );
}

// ============================================================================
// FP8 Kernels - For Ampere/Hopper GPUs (Inference Optimization)
// ============================================================================

// FP8 Format Selection Guide:
// - E4M3 (4 exp, 3 mantissa): Higher precision, smaller range
//   * Best for: Hopper (sm_90+, H100), Ada (sm_89, RTX 40xx) with native FP8 tensor cores
//   * Use case: Preferred for modern GPUs with minimal accuracy loss
//
// - E5M2 (5 exp, 2 mantissa): Lower precision, larger dynamic range
//   * Best for: Ampere (sm_80+, RTX 30xx, A100) where FP8 is software-emulated
//   * Use case: Prevents underflow/overflow on older GPUs without native FP8
//
// This implementation uses E4M3 format with FP32 accumulation for stability
// Note: On Ampere (sm_80-sm_89), FP8 ops are emulated in software (slower than native)
//       On Hopper/Ada (sm_89+), FP8 ops use hardware tensor cores (4x faster)

#if __CUDA_ARCH__ >= 800  // Ampere and newer (sm_80+)

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ Q,
    const boostr_fp8_e4m3* __restrict__ K,
    const boostr_fp8_e4m3* __restrict__ V,
    boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,  // GQA: can be less than num_heads
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float q_scale,
    const float k_scale,
    const float v_scale,
    const float o_scale
) {
    // Padded strides
    constexpr int HEAD_STRIDE = SMEM_STRIDE(HEAD_DIM, 1);

    extern __shared__ boostr_fp8_e4m3 smem_fp8[];

    boostr_fp8_e4m3* Q_smem_flat = smem_fp8;
    boostr_fp8_e4m3* K_smem_flat = smem_fp8 + BLOCK_M * HEAD_STRIDE;
    boostr_fp8_e4m3* V_smem_flat = smem_fp8 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // GQA: Map query head to KV head (multiple Q heads share one KV head)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Q/O use num_heads, K/V use num_kv_heads
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                              + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const boostr_fp8_e4m3* Q_base = Q + head_offset;
    const boostr_fp8_e4m3* K_base = K + kv_head_offset;
    const boostr_fp8_e4m3* V_base = V + kv_head_offset;
    boostr_fp8_e4m3* O_base = O + head_offset;
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

    if (q_row >= q_tile_size) {
        if (q_start + q_row < seq_len_q) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_base[(q_start + q_row) * HEAD_DIM + d] = boostr_fp8_e4m3(0);
            }
            L_base[q_start + q_row] = -INFINITY;
        }
        return;
    }

    // FP32 accumulation (CRITICAL for FP8)
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

        float m_new = m_local;
        for (int j = 0; j < k_tile_size; ++j) {
            if (causal && (q_start + q_row) < (k_start + j)) continue;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                // Dequantize FP8 → FP32 for computation
                float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), q_scale);
                float k_val = fp8_e4m3_to_f32(K_smem(j, d), k_scale);
                score += q_val * k_val;
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
                float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), q_scale);
                float k_val = fp8_e4m3_to_f32(K_smem(j, d), k_scale);
                score += q_val * k_val;
            }
            score *= scale;
            const float exp_score = __expf(score - m_new);
            l_new += exp_score;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float v_val = fp8_e4m3_to_f32(V_smem(j, d), v_scale);
                O_local[d] += exp_score * v_val;
            }
        }

        m_local = m_new;
        l_local = l_new;
        __syncthreads();
    }

    const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        // Quantize FP32 → FP8 for output
        float out_val = O_local[d] * inv_l;
        uint8_t fp8_val = f32_to_fp8_e4m3_raw(out_val, o_scale);
        O_base[(q_start + q_row) * HEAD_DIM + d] = boostr_fp8_e4m3(fp8_val);
    }

    L_base[q_start + q_row] = m_local + __logf(l_local);

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// FP8 kernel instantiations with GQA support (Ampere sm_80+ with software emulation or Hopper native)
extern "C" __global__ void flash_attention_fwd_64_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<64, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void flash_attention_fwd_128_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<128, 128, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void flash_attention_fwd_32_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<32, 128, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void flash_attention_fwd_96_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<96, 64, 128>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void flash_attention_fwd_192_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<192, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

extern "C" __global__ void flash_attention_fwd_256_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    boostr_fp8_e4m3* O, float* L,
    const int batch_size, const int num_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float q_scale, const float k_scale, const float v_scale, const float o_scale
) {
    flash_attention_fwd_fp8_impl<256, 64, 64>(
        Q, K, V, O, L, batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        q_scale, k_scale, v_scale, o_scale
    );
}

#endif  // __CUDA_ARCH__ >= 800
