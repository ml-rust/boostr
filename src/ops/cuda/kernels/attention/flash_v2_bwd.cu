// Flash Attention v2 Backward Pass - Production Implementation
// Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// Tri Dao, 2023 (https://arxiv.org/abs/2307.08691)
//
// Backward Pass Algorithm:
// 1. Preprocessing: Compute D_i = rowsum(dO_i ⊙ O_i) for each query position
// 2. Main backward: For each K/V block (parallelized across CUDA blocks):
//    - Load K_j, V_j into shared memory (stays for entire Q loop)
//    - Iterate over Q blocks:
//      - Recompute P_ij = exp(Q_i @ K_j^T * scale - LSE_i)
//      - Accumulate dV_j += P_ij^T @ dO_i (local, no atomics)
//      - Compute dP_ij = dO_i @ V_j^T
//      - Compute dS_ij = P_ij * (dP_ij - D_i) * scale (softmax backward)
//      - Accumulate dK_j += dS_ij^T @ Q_i (local, no atomics)
//      - Compute dQ_i = dS_ij @ K_j and write with atomics (multiple K blocks contribute)
//
// Key Design Decisions:
// 1. **Parallelization**: Each CUDA block processes ONE K/V block, iterates over ALL Q blocks
//    - Benefit: dK, dV computed locally without atomics (major performance win)
//    - Tradeoff: dQ requires atomics (unavoidable, but less frequent than dK/dV would be)
// 2. **Shared Memory**: Dynamic allocation, adaptive block sizes per GPU
// 3. **Numerical Stability**: FP32 accumulation even for FP16/BF16/FP8 inputs
// 4. **Causal Masking**: Skip Q blocks where q_pos < k_pos (entire block skip)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Warp-level Primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Preprocessing Kernel: Compute D = rowsum(dO ⊙ O)
// ============================================================================

// Compute D_i = sum_d(dO_i[d] * O_i[d]) for each query position
// Used in softmax backward: dS = P * (dP - D) * scale
template<int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_impl(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len) return;

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const float* dO_row = dO + offset + q_pos * HEAD_DIM;
    const float* O_row = O + offset + q_pos * HEAD_DIM;

    // Compute row-wise dot product
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        sum += dO_row[d] * O_row[d];
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// Kernel instantiations for preprocessing
extern "C" __global__ void flash_attention_preprocess_bwd_64_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_impl<64>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_preprocess_bwd_128_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_impl<128>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_preprocess_bwd_32_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_impl<32>(dO, O, D, batch_size, num_heads, seq_len);
}

// ============================================================================
// Main Backward Kernel - FP32
// ============================================================================

// Main backward pass computation
// Grid: (batch_size * num_heads, num_k_blocks)
// Block: blockDim.x threads (typically 128 or 256)
// Each CUDA block processes ONE K/V block, iterates over ALL Q blocks
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_fp32_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    // Dynamic shared memory layout:
    // [K: BLOCK_N x HEAD_DIM][V: BLOCK_N x HEAD_DIM]
    // [Q: BLOCK_M x HEAD_DIM][dO: BLOCK_M x HEAD_DIM]
    extern __shared__ float smem[];

    float* K_smem_flat = smem;
    float* V_smem_flat = smem + BLOCK_N * HEAD_DIM;
    float* Q_smem_flat = smem + 2 * BLOCK_N * HEAD_DIM;
    float* dO_smem_flat = smem + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int k_block = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    // Base pointers for this (batch, head)
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_heads + head_idx) * seq_len_q;

    const float* Q_base = Q + head_offset;
    const float* K_base = K + kv_head_offset;
    const float* V_base = V + kv_head_offset;
    const float* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    float* dQ_base = dQ + head_offset;
    float* dK_base = dK + kv_head_offset;
    float* dV_base = dV + kv_head_offset;

    // Load K and V tiles into shared memory (stays for entire kernel)
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Per-thread accumulators for dK and dV (each thread computes part of K tile)
    // No atomics needed - each K/V block computed by exactly one CUDA block
    const int k_row = tid;
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    // Determine Q block range (for causal masking)
    int q_block_start = 0;
    int q_block_end = (seq_len_q + BLOCK_M - 1) / BLOCK_M;

    if (causal) {
        // For causal: only process Q blocks where q_pos >= k_pos
        q_block_start = k_block;  // Skip Q blocks before this K block
    }

    // Iterate over Q blocks
    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        // Load Q and dO tiles into shared memory
        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Each thread can process Q rows for dQ computation
        // AND accumulate dK/dV for its assigned K row
        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            // Accumulator for dQ (only for thread's assigned Q row)
            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            // Iterate over K positions in this K tile
            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                // Apply causal mask (position-level)
                if (causal && q_pos < k_pos) continue;

                // Recompute attention score: Q @ K^T
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += Q_smem(q_row, d) * K_smem(k_col, d);
                }
                qk_score *= scale;

                // Recompute attention probability: P = exp(score - LSE)
                const float p_val = __expf(qk_score - lse_val);

                // Compute dP = dO @ V^T (gradient w.r.t. pre-softmax scores)
                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += dO_smem(q_row, d) * V_smem(k_col, d);
                }

                // Softmax backward: dS = P * (dP - D) * scale
                const float ds_val = p_val * (dp_val - d_val) * scale;

                // Accumulate dQ += dS * K (only for tid's Q row)
                if (tid == q_row && tid < q_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * K_smem(k_col, d);
                    }
                }

                // Accumulate dV += P * dO (thread tid accumulates for K row k_row)
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * dO_smem(q_row, d);
                    }
                }

                // Accumulate dK += dS * Q (thread tid accumulates for K row k_row)
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dK_local[d] += ds_val * Q_smem(q_row, d);
                    }
                }
            }

            // Write dQ with atomic adds (only for tid's assigned Q row)
            if (tid == q_row && tid < q_tile_size) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV (no atomics - each K block computed by exactly one CUDA block)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[(k_start + k_row) * HEAD_DIM + d] = dK_local[d];
            dV_base[(k_start + k_row) * HEAD_DIM + d] = dV_local[d];
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// ============================================================================
// Kernel Instantiations - FP32
// ============================================================================

// head_dim=64, BLOCK_M=128, BLOCK_N=128
extern "C" __global__ void flash_attention_bwd_64_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// head_dim=128, BLOCK_M=128, BLOCK_N=64
extern "C" __global__ void flash_attention_bwd_128_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// head_dim=32, BLOCK_M=128, BLOCK_N=128
extern "C" __global__ void flash_attention_bwd_32_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_96_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_impl<96>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_96_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<96, 64, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_192_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_impl<192>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_192_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<192, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_256_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_impl<256>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_256_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp32_impl<256, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// FP16 Backward Kernels - Mixed Precision (FP16 I/O, FP32 accumulation)
// ============================================================================

// Preprocessing for FP16 - converts to FP32 for computation
template<int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_fp16_impl(
    const __half* __restrict__ dO,
    const __half* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len) return;

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const __half* dO_row = dO + offset + q_pos * HEAD_DIM;
    const __half* O_row = O + offset + q_pos * HEAD_DIM;

    // Compute row-wise dot product in FP32
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        sum += __half2float(dO_row[d]) * __half2float(O_row[d]);
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// Main backward kernel for FP16 - FP32 accumulation
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __half* __restrict__ dQ,
    __half* __restrict__ dK,
    __half* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    extern __shared__ __half smem_fp16[];

    __half* K_smem_flat = smem_fp16;
    __half* V_smem_flat = smem_fp16 + BLOCK_N * HEAD_DIM;
    __half* Q_smem_flat = smem_fp16 + 2 * BLOCK_N * HEAD_DIM;
    __half* dO_smem_flat = smem_fp16 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int k_block = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_heads + head_idx) * seq_len_q;

    const __half* Q_base = Q + head_offset;
    const __half* K_base = K + kv_head_offset;
    const __half* V_base = V + kv_head_offset;
    const __half* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    __half* dQ_base = dQ + head_offset;
    __half* dK_base = dK + kv_head_offset;
    __half* dV_base = dV + kv_head_offset;

    // Load K and V tiles
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // FP32 accumulators for dK and dV
    const int k_row = tid;
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    int q_block_start = 0;
    int q_block_end = (seq_len_q + BLOCK_M - 1) / BLOCK_M;

    if (causal) {
        q_block_start = k_block;
    }

    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                // Compute in FP32
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += __half2float(Q_smem(q_row, d)) * __half2float(K_smem(k_col, d));
                }
                qk_score *= scale;

                const float p_val = __expf(qk_score - lse_val);

                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += __half2float(dO_smem(q_row, d)) * __half2float(V_smem(k_col, d));
                }

                const float ds_val = p_val * (dp_val - d_val) * scale;

                if (tid == q_row && tid < q_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * __half2float(K_smem(k_col, d));
                    }
                }

                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * __half2float(dO_smem(q_row, d));
                        dK_local[d] += ds_val * __half2float(Q_smem(q_row, d));
                    }
                }
            }

            // Write dQ with atomics (convert FP32 → FP16)
            if (tid == q_row && tid < q_tile_size) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd((float*)&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV (convert FP32 → FP16)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[(k_start + k_row) * HEAD_DIM + d] = __float2half(dK_local[d]);
            dV_base[(k_start + k_row) * HEAD_DIM + d] = __float2half(dV_local[d]);
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// FP16 kernel instantiations
extern "C" __global__ void flash_attention_preprocess_bwd_64_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_fp16_impl<64>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_bwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_128_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_fp16_impl<128>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_bwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_32_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_fp16_impl<32>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_32_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_96_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_fp16_impl<96>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_96_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<96, 64, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_192_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_fp16_impl<192>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_192_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<192, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_256_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_fp16_impl<256>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_256_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_fp16_impl<256, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// BF16 Backward Kernels - Mixed Precision (BF16 I/O, FP32 accumulation)
// ============================================================================

// Preprocessing for BF16
template<int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_bf16_impl(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len) return;

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const __nv_bfloat16* dO_row = dO + offset + q_pos * HEAD_DIM;
    const __nv_bfloat16* O_row = O + offset + q_pos * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        sum += __bfloat162float(dO_row[d]) * __bfloat162float(O_row[d]);
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// Main backward kernel for BF16
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
    extern __shared__ __nv_bfloat16 smem_bf16[];

    __nv_bfloat16* K_smem_flat = smem_bf16;
    __nv_bfloat16* V_smem_flat = smem_bf16 + BLOCK_N * HEAD_DIM;
    __nv_bfloat16* Q_smem_flat = smem_bf16 + 2 * BLOCK_N * HEAD_DIM;
    __nv_bfloat16* dO_smem_flat = smem_bf16 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int k_block = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_heads + head_idx) * seq_len_q;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + kv_head_offset;
    const __nv_bfloat16* V_base = V + kv_head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    __nv_bfloat16* dQ_base = dQ + head_offset;
    __nv_bfloat16* dK_base = dK + kv_head_offset;
    __nv_bfloat16* dV_base = dV + kv_head_offset;

    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int k_row = tid;
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    int q_block_start = 0;
    int q_block_end = (seq_len_q + BLOCK_M - 1) / BLOCK_M;

    if (causal) {
        q_block_start = k_block;
    }

    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += __bfloat162float(Q_smem(q_row, d)) * __bfloat162float(K_smem(k_col, d));
                }
                qk_score *= scale;

                const float p_val = __expf(qk_score - lse_val);

                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += __bfloat162float(dO_smem(q_row, d)) * __bfloat162float(V_smem(k_col, d));
                }

                const float ds_val = p_val * (dp_val - d_val) * scale;

                if (tid == q_row && tid < q_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * __bfloat162float(K_smem(k_col, d));
                    }
                }

                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * __bfloat162float(dO_smem(q_row, d));
                        dK_local[d] += ds_val * __bfloat162float(Q_smem(q_row, d));
                    }
                }
            }

            if (tid == q_row && tid < q_tile_size) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd((float*)&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[(k_start + k_row) * HEAD_DIM + d] = __float2bfloat16(dK_local[d]);
            dV_base[(k_start + k_row) * HEAD_DIM + d] = __float2bfloat16(dV_local[d]);
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// BF16 kernel instantiations
extern "C" __global__ void flash_attention_preprocess_bwd_64_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_bf16_impl<64>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_bwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_128_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) {
    flash_attention_preprocess_bwd_bf16_impl<128>(dO, O, D, batch_size, num_heads, seq_len);
}

extern "C" __global__ void flash_attention_bwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_32_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_bf16_impl<32>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_32_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_96_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_bf16_impl<96>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_96_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<96, 64, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_192_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_bf16_impl<192>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_192_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<192, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_256_bf16(
    const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D,
    const int batch_size, const int num_heads, const int seq_len
) { flash_attention_preprocess_bwd_bf16_impl<256>(dO, O, D, batch_size, num_heads, seq_len); }

extern "C" __global__ void flash_attention_bwd_256_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_bwd_bf16_impl<256, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// FP8 Backward Kernels - For Ampere/Hopper GPUs (FP8 I/O, FP32 accumulation)
// ============================================================================

#if __CUDA_ARCH__ >= 800  // Ampere and newer

// Preprocessing for FP8
template<int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ dO,
    const boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float dO_scale,
    const float O_scale
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len) return;

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const boostr_fp8_e4m3* dO_row = dO + offset + q_pos * HEAD_DIM;
    const boostr_fp8_e4m3* O_row = O + offset + q_pos * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float dO_val = fp8_e4m3_to_f32(dO_row[d], dO_scale);
        float O_val = fp8_e4m3_to_f32(O_row[d], O_scale);
        sum += dO_val * O_val;
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// Main backward kernel for FP8
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ Q,
    const boostr_fp8_e4m3* __restrict__ K,
    const boostr_fp8_e4m3* __restrict__ V,
    const boostr_fp8_e4m3* __restrict__ O,
    const boostr_fp8_e4m3* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    boostr_fp8_e4m3* __restrict__ dQ,
    boostr_fp8_e4m3* __restrict__ dK,
    boostr_fp8_e4m3* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float Q_scale,
    const float K_scale,
    const float V_scale,
    const float dO_scale,
    const float dQ_scale,
    const float dK_scale,
    const float dV_scale
) {
    extern __shared__ boostr_fp8_e4m3 smem_fp8[];

    boostr_fp8_e4m3* K_smem_flat = smem_fp8;
    boostr_fp8_e4m3* V_smem_flat = smem_fp8 + BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* Q_smem_flat = smem_fp8 + 2 * BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* dO_smem_flat = smem_fp8 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int k_block = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_heads + head_idx) * seq_len_q;

    const boostr_fp8_e4m3* Q_base = Q + head_offset;
    const boostr_fp8_e4m3* K_base = K + kv_head_offset;
    const boostr_fp8_e4m3* V_base = V + kv_head_offset;
    const boostr_fp8_e4m3* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    boostr_fp8_e4m3* dQ_base = dQ + head_offset;
    boostr_fp8_e4m3* dK_base = dK + kv_head_offset;
    boostr_fp8_e4m3* dV_base = dV + kv_head_offset;

    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int k_row = tid;
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    int q_block_start = 0;
    int q_block_end = (seq_len_q + BLOCK_M - 1) / BLOCK_M;

    if (causal) {
        q_block_start = k_block;
    }

    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                // Dequantize FP8 → FP32 for computation
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), Q_scale);
                    float k_val = fp8_e4m3_to_f32(K_smem(k_col, d), K_scale);
                    qk_score += q_val * k_val;
                }
                qk_score *= scale;

                const float p_val = __expf(qk_score - lse_val);

                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float dO_val = fp8_e4m3_to_f32(dO_smem(q_row, d), dO_scale);
                    float v_val = fp8_e4m3_to_f32(V_smem(k_col, d), V_scale);
                    dp_val += dO_val * v_val;
                }

                const float ds_val = p_val * (dp_val - d_val) * scale;

                if (tid == q_row && tid < q_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        float k_val = fp8_e4m3_to_f32(K_smem(k_col, d), K_scale);
                        dQ_local[d] += ds_val * k_val;
                    }
                }

                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        float dO_val = fp8_e4m3_to_f32(dO_smem(q_row, d), dO_scale);
                        float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), Q_scale);
                        dV_local[d] += p_val * dO_val;
                        dK_local[d] += ds_val * q_val;
                    }
                }
            }

            if (tid == q_row && tid < q_tile_size) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd((float*)&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Quantize FP32 → FP8 for output
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[(k_start + k_row) * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dK_local[d], dK_scale));
            dV_base[(k_start + k_row) * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dV_local[d], dV_scale));
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// FP8 kernel instantiations
extern "C" __global__ void flash_attention_preprocess_bwd_64_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) {
    flash_attention_preprocess_bwd_fp8_impl<64>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale);
}

extern "C" __global__ void flash_attention_bwd_64_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_128_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) {
    flash_attention_preprocess_bwd_fp8_impl<128>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale);
}

extern "C" __global__ void flash_attention_bwd_128_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_32_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<32>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_32_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_96_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<96>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_96_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<96, 64, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_192_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<192>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_192_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<192, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_256_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<256>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_256_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<256, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

#endif  // __CUDA_ARCH__ >= 800
