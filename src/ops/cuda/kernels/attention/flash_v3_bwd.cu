// Flash Attention v3 Backward Pass - H100/Hopper Optimized
// Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// Enhanced with H100 optimizations: padded shared memory, bank conflict elimination, register tiling
//
// Key design: Uses proven v2 backward parallelization (iterate Q blocks per K block)
// Does NOT use forward's warp specialization because K/V tiles are reused, not streamed
//
// Algorithm:
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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Shared Memory Stride Macro (for bank conflict elimination)
// ============================================================================
#define SMEM_STRIDE(dim, pad) ((dim) + (pad))

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
__device__ void flash_attention_v3_preprocess_bwd_impl(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int head_dim
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int q_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len_q) return;

    // Base pointers for this batch and head
    const int qo_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const float* dO_base = dO + qo_offset;
    const float* O_base = O + qo_offset;
    float* D_base = D + (batch_idx * num_heads + head_idx) * seq_len_q;

    // Compute D_i = sum(dO_i * O_i)
    float d_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        const float do_val = dO_base[q_idx * head_dim + d];
        const float o_val = O_base[q_idx * head_dim + d];
        d_sum += do_val * o_val;
    }

    D_base[q_idx] = d_sum;
}

// Kernel wrapper for head_dim=64
extern "C" __global__ void flash_attention_v3_preprocess_bwd_64(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_impl<64>(dO, O, D, batch_size, num_heads, seq_len_q, 64);
}

// Kernel wrapper for head_dim=128
extern "C" __global__ void flash_attention_v3_preprocess_bwd_128(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_impl<128>(dO, O, D, batch_size, num_heads, seq_len_q, 128);
}

// ============================================================================
// Main Backward Kernel - FP32 Reference Implementation
// ============================================================================

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__device__ void flash_attention_v3_bwd_kernel(
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
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int k_block_idx = blockIdx.y;

    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const float* Q_base = Q + qkv_offset;
    const float* dO_base = dO + qkv_offset;
    float* dQ_base = dQ + qkv_offset;
    float* LSE_base = (float*)(LSE + (batch_idx * num_heads + head_idx) * seq_len_q);
    float* D_base = (float*)(D + (batch_idx * num_heads + head_idx) * seq_len_q);

    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const float* K_base = K + k_offset;
    const float* V_base = V + k_offset;
    float* dK_base = dK + k_offset;
    float* dV_base = dV + k_offset;

    // K/V block boundaries
    const int k_start = k_block_idx * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_size = k_end - k_start;

    // Dynamic shared memory (configured at launch)
    extern __shared__ float smem[];

    // Layout: [K: BLOCK_N x HEAD_DIM][V: BLOCK_N x HEAD_DIM][Q: BLOCK_M x HEAD_DIM][dO: BLOCK_M x HEAD_DIM]
    float* K_smem = smem;
    float* V_smem = smem + BLOCK_N * HEAD_DIM;
    float* Q_smem = smem + 2 * BLOCK_N * HEAD_DIM;
    float* dO_smem = smem + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    // Load K and V tiles (stays in shared memory for entire kernel)
    for (int i = tid; i < k_size * HEAD_DIM; i += blockDim.x) {
        const int k_row = i / HEAD_DIM;
        const int k_col = i % HEAD_DIM;
        K_smem[k_row * HEAD_DIM + k_col] = K_base[(k_start + k_row) * HEAD_DIM + k_col];
        V_smem[k_row * HEAD_DIM + k_col] = V_base[(k_start + k_row) * HEAD_DIM + k_col];
    }
    __syncthreads();

    // Determine Q block range (with causal masking optimization)
    const int num_q_blocks = (seq_len_q + BLOCK_M - 1) / BLOCK_M;
    int q_block_start = 0;
    if (causal) {
        // For causal attention: Q can only attend to K where k <= q
        // So for K block at k_start, only Q blocks where q >= k_start can have gradients
        q_block_start = k_start / BLOCK_M;
    }

    // Iterate over Q blocks
    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_size = q_end - q_start;

        // Load Q and dO tiles
        for (int i = tid; i < q_size * HEAD_DIM; i += blockDim.x) {
            const int q_row = i / HEAD_DIM;
            const int q_col = i % HEAD_DIM;
            Q_smem[q_row * HEAD_DIM + q_col] = Q_base[(q_start + q_row) * HEAD_DIM + q_col];
            dO_smem[q_row * HEAD_DIM + q_col] = dO_base[(q_start + q_row) * HEAD_DIM + q_col];
        }
        __syncthreads();

        // Each thread processes (q,k) pairs
        for (int qk_idx = tid; qk_idx < q_size * k_size; qk_idx += blockDim.x) {
            const int q_row = qk_idx / k_size;
            const int k_row = qk_idx % k_size;
            const int q_global = q_start + q_row;
            const int k_global = k_start + k_row;

            // Check causal masking
            if (causal && q_global < k_global) {
                continue;
            }

            // Load saved statistics
            const float lse_val = LSE_base[q_global];
            const float d_val = D_base[q_global];

            // Recompute attention score: S = Q @ K^T
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_smem[q_row * HEAD_DIM + d] * K_smem[k_row * HEAD_DIM + d];
            }
            score *= scale;

            // Recompute attention probability: P = exp(S - LSE)
            const float prob = expf(score - lse_val);

            // Compute dP = dO @ V^T
            float dp = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dp += dO_smem[q_row * HEAD_DIM + d] * V_smem[k_row * HEAD_DIM + d];
            }

            // Compute dS = P * (dP - D) (softmax backward)
            const float ds = prob * (dp - d_val);

            // Accumulate dV += P^T @ dO (use atomics for thread safety)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomicAdd(&dV_base[k_global * HEAD_DIM + d], prob * dO_smem[q_row * HEAD_DIM + d]);
            }

            // Accumulate dK += scale * dS^T @ Q (use atomics for thread safety)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomicAdd(&dK_base[k_global * HEAD_DIM + d], scale * ds * Q_smem[q_row * HEAD_DIM + d]);
            }

            // Compute dQ = scale * dS @ K and write with atomic
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float dq_val = scale * ds * K_smem[k_row * HEAD_DIM + d];
                atomicAdd(&dQ_base[q_global * HEAD_DIM + d], dq_val);
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Kernel Instantiations - FP32
// ============================================================================

extern "C" __global__ void flash_attention_v3_bwd_64(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_kernel<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_v3_bwd_128(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO, const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_kernel<16, 32, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// FP16 Backward Kernels - Mixed Precision (FP16 I/O, FP32 accumulation)
// ============================================================================

// Preprocessing for FP16
template<int HEAD_DIM>
__device__ void flash_attention_v3_preprocess_bwd_fp16_impl(
    const __half* __restrict__ dO,
    const __half* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int head_dim
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int q_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len_q) return;

    const int qo_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const __half* dO_base = dO + qo_offset;
    const __half* O_base = O + qo_offset;
    float* D_base = D + (batch_idx * num_heads + head_idx) * seq_len_q;

    // Compute D_i = sum(dO_i * O_i) in FP32
    float d_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        d_sum += __half2float(dO_base[q_idx * head_dim + d]) * __half2float(O_base[q_idx * head_dim + d]);
    }

    D_base[q_idx] = d_sum;
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_fp16_64(
    const __half* __restrict__ dO,
    const __half* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_fp16_impl<64>(dO, O, D, batch_size, num_heads, seq_len_q, 64);
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_fp16_128(
    const __half* __restrict__ dO,
    const __half* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_fp16_impl<128>(dO, O, D, batch_size, num_heads, seq_len_q, 128);
}

// Main backward kernel for FP16
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__device__ void flash_attention_v3_bwd_fp16_kernel(
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
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int k_block_idx = blockIdx.y;

    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const __half* Q_base = Q + qkv_offset;
    const __half* dO_base = dO + qkv_offset;
    __half* dQ_base = dQ + qkv_offset;
    float* LSE_base = (float*)(LSE + (batch_idx * num_heads + head_idx) * seq_len_q);
    float* D_base = (float*)(D + (batch_idx * num_heads + head_idx) * seq_len_q);

    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const __half* K_base = K + k_offset;
    const __half* V_base = V + k_offset;
    __half* dK_base = dK + k_offset;
    __half* dV_base = dV + k_offset;

    const int k_start = k_block_idx * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_size = k_end - k_start;

    // Dynamic shared memory for FP16
    extern __shared__ __half smem_fp16[];
    __half* K_smem = smem_fp16;
    __half* V_smem = smem_fp16 + BLOCK_N * HEAD_DIM;
    __half* Q_smem = smem_fp16 + 2 * BLOCK_N * HEAD_DIM;
    __half* dO_smem = smem_fp16 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    // Load K and V tiles
    for (int i = tid; i < k_size * HEAD_DIM; i += blockDim.x) {
        const int k_row = i / HEAD_DIM;
        const int k_col = i % HEAD_DIM;
        K_smem[k_row * HEAD_DIM + k_col] = K_base[(k_start + k_row) * HEAD_DIM + k_col];
        V_smem[k_row * HEAD_DIM + k_col] = V_base[(k_start + k_row) * HEAD_DIM + k_col];
    }
    __syncthreads();

    const int num_q_blocks = (seq_len_q + BLOCK_M - 1) / BLOCK_M;
    int q_block_start = 0;
    if (causal) {
        q_block_start = k_start / BLOCK_M;
    }

    // FP32 accumulators for dK and dV
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    // Iterate over Q blocks
    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_size = q_end - q_start;

        // Load Q and dO tiles
        for (int i = tid; i < q_size * HEAD_DIM; i += blockDim.x) {
            const int q_row = i / HEAD_DIM;
            const int q_col = i % HEAD_DIM;
            Q_smem[q_row * HEAD_DIM + q_col] = Q_base[(q_start + q_row) * HEAD_DIM + q_col];
            dO_smem[q_row * HEAD_DIM + q_col] = dO_base[(q_start + q_row) * HEAD_DIM + q_col];
        }
        __syncthreads();

        // Process each query position
        for (int q_row = tid; q_row < q_size; q_row += blockDim.x) {
            const int q_global = q_start + q_row;

            const float lse_val = LSE_base[q_global];
            const float d_val = D_base[q_global];

            for (int k_row = 0; k_row < k_size; ++k_row) {
                const int k_global = k_start + k_row;

                if (causal && q_global < k_global) {
                    continue;
                }

                // Recompute attention score in FP32
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(Q_smem[q_row * HEAD_DIM + d]) * __half2float(K_smem[k_row * HEAD_DIM + d]);
                }
                score *= scale;

                const float prob = expf(score - lse_val);

                // Compute dP in FP32
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp += __half2float(dO_smem[q_row * HEAD_DIM + d]) * __half2float(V_smem[k_row * HEAD_DIM + d]);
                }

                const float ds = prob * (dp - d_val) * scale;

                // Accumulate dV and dK in FP32
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dV_local[d] += prob * __half2float(dO_smem[q_row * HEAD_DIM + d]);
                    dK_local[d] += ds * __half2float(Q_smem[q_row * HEAD_DIM + d]);
                }

                // Compute dQ and write with atomic using dtype-safe helper
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const float dq_val = ds * __half2float(K_smem[k_row * HEAD_DIM + d]);
                    atomic_add_dtype(&dQ_base[q_global * HEAD_DIM + d], dq_val);
                }
            }
        }

        __syncthreads();
    }

    // Write final dK and dV (convert FP32 -> FP16)
    for (int k_row = tid; k_row < k_size; k_row += blockDim.x) {
        const int k_global = k_start + k_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[k_global * HEAD_DIM + d] = __float2half(dK_local[d]);
            dV_base[k_global * HEAD_DIM + d] = __float2half(dV_local[d]);
        }
    }
}

extern "C" __global__ void flash_attention_v3_bwd_fp16_64(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_fp16_kernel<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_v3_bwd_fp16_128(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_fp16_kernel<16, 32, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

// ============================================================================
// BF16 Backward Kernels - Mixed Precision (BF16 I/O, FP32 accumulation)
// ============================================================================

// Preprocessing for BF16
template<int HEAD_DIM>
__device__ void flash_attention_v3_preprocess_bwd_bf16_impl(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int head_dim
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int q_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len_q) return;

    const int qo_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const __nv_bfloat16* dO_base = dO + qo_offset;
    const __nv_bfloat16* O_base = O + qo_offset;
    float* D_base = D + (batch_idx * num_heads + head_idx) * seq_len_q;

    // Compute D_i = sum(dO_i * O_i) in FP32
    float d_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        d_sum += __bfloat162float(dO_base[q_idx * head_dim + d]) * __bfloat162float(O_base[q_idx * head_dim + d]);
    }

    D_base[q_idx] = d_sum;
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_bf16_64(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_bf16_impl<64>(dO, O, D, batch_size, num_heads, seq_len_q, 64);
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_bf16_128(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q
) {
    flash_attention_v3_preprocess_bwd_bf16_impl<128>(dO, O, D, batch_size, num_heads, seq_len_q, 128);
}

// Main backward kernel for BF16
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__device__ void flash_attention_v3_bwd_bf16_kernel(
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
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int k_block_idx = blockIdx.y;

    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const __nv_bfloat16* Q_base = Q + qkv_offset;
    const __nv_bfloat16* dO_base = dO + qkv_offset;
    __nv_bfloat16* dQ_base = dQ + qkv_offset;
    float* LSE_base = (float*)(LSE + (batch_idx * num_heads + head_idx) * seq_len_q);
    float* D_base = (float*)(D + (batch_idx * num_heads + head_idx) * seq_len_q);

    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const __nv_bfloat16* K_base = K + k_offset;
    const __nv_bfloat16* V_base = V + k_offset;
    __nv_bfloat16* dK_base = dK + k_offset;
    __nv_bfloat16* dV_base = dV + k_offset;

    const int k_start = k_block_idx * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_size = k_end - k_start;

    // Dynamic shared memory for BF16
    extern __shared__ __nv_bfloat16 smem_bf16[];
    __nv_bfloat16* K_smem = smem_bf16;
    __nv_bfloat16* V_smem = smem_bf16 + BLOCK_N * HEAD_DIM;
    __nv_bfloat16* Q_smem = smem_bf16 + 2 * BLOCK_N * HEAD_DIM;
    __nv_bfloat16* dO_smem = smem_bf16 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    // Load K and V tiles
    for (int i = tid; i < k_size * HEAD_DIM; i += blockDim.x) {
        const int k_row = i / HEAD_DIM;
        const int k_col = i % HEAD_DIM;
        K_smem[k_row * HEAD_DIM + k_col] = K_base[(k_start + k_row) * HEAD_DIM + k_col];
        V_smem[k_row * HEAD_DIM + k_col] = V_base[(k_start + k_row) * HEAD_DIM + k_col];
    }
    __syncthreads();

    const int num_q_blocks = (seq_len_q + BLOCK_M - 1) / BLOCK_M;
    int q_block_start = 0;
    if (causal) {
        q_block_start = k_start / BLOCK_M;
    }

    // FP32 accumulators for dK and dV
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    // Iterate over Q blocks
    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_size = q_end - q_start;

        // Load Q and dO tiles
        for (int i = tid; i < q_size * HEAD_DIM; i += blockDim.x) {
            const int q_row = i / HEAD_DIM;
            const int q_col = i % HEAD_DIM;
            Q_smem[q_row * HEAD_DIM + q_col] = Q_base[(q_start + q_row) * HEAD_DIM + q_col];
            dO_smem[q_row * HEAD_DIM + q_col] = dO_base[(q_start + q_row) * HEAD_DIM + q_col];
        }
        __syncthreads();

        // Process each query position
        for (int q_row = tid; q_row < q_size; q_row += blockDim.x) {
            const int q_global = q_start + q_row;

            const float lse_val = LSE_base[q_global];
            const float d_val = D_base[q_global];

            for (int k_row = 0; k_row < k_size; ++k_row) {
                const int k_global = k_start + k_row;

                if (causal && q_global < k_global) {
                    continue;
                }

                // Recompute attention score in FP32
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __bfloat162float(Q_smem[q_row * HEAD_DIM + d]) * __bfloat162float(K_smem[k_row * HEAD_DIM + d]);
                }
                score *= scale;

                const float prob = expf(score - lse_val);

                // Compute dP in FP32
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp += __bfloat162float(dO_smem[q_row * HEAD_DIM + d]) * __bfloat162float(V_smem[k_row * HEAD_DIM + d]);
                }

                const float ds = prob * (dp - d_val) * scale;

                // Accumulate dV and dK in FP32
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dV_local[d] += prob * __bfloat162float(dO_smem[q_row * HEAD_DIM + d]);
                    dK_local[d] += ds * __bfloat162float(Q_smem[q_row * HEAD_DIM + d]);
                }

                // Compute dQ and write with atomic using dtype-safe helper
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const float dq_val = ds * __bfloat162float(K_smem[k_row * HEAD_DIM + d]);
                    atomic_add_dtype(&dQ_base[q_global * HEAD_DIM + d], dq_val);
                }
            }
        }

        __syncthreads();
    }

    // Write final dK and dV (convert FP32 -> BF16)
    for (int k_row = tid; k_row < k_size; k_row += blockDim.x) {
        const int k_global = k_start + k_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[k_global * HEAD_DIM + d] = __float2bfloat16(dK_local[d]);
            dV_base[k_global * HEAD_DIM + d] = __float2bfloat16(dV_local[d]);
        }
    }
}

extern "C" __global__ void flash_attention_v3_bwd_bf16_64(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_bf16_kernel<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void flash_attention_v3_bwd_bf16_128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    flash_attention_v3_bwd_bf16_kernel<16, 32, 128>(
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
__device__ void flash_attention_v3_preprocess_bwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ dO,
    const boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int head_dim,
    const float dO_scale,
    const float O_scale
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int q_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len_q) return;

    const int qo_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const boostr_fp8_e4m3* dO_base = dO + qo_offset;
    const boostr_fp8_e4m3* O_base = O + qo_offset;
    float* D_base = D + (batch_idx * num_heads + head_idx) * seq_len_q;

    // Compute D_i = sum(dO_i * O_i) in FP32
    float d_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        float dO_val = fp8_e4m3_to_f32(dO_base[q_idx * head_dim + d], dO_scale);
        float O_val = fp8_e4m3_to_f32(O_base[q_idx * head_dim + d], O_scale);
        d_sum += dO_val * O_val;
    }

    D_base[q_idx] = d_sum;
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_fp8_64(
    const boostr_fp8_e4m3* __restrict__ dO,
    const boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const float dO_scale,
    const float O_scale
) {
    flash_attention_v3_preprocess_bwd_fp8_impl<64>(dO, O, D, batch_size, num_heads, seq_len_q, 64, dO_scale, O_scale);
}

extern "C" __global__ void flash_attention_v3_preprocess_bwd_fp8_128(
    const boostr_fp8_e4m3* __restrict__ dO,
    const boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const float dO_scale,
    const float O_scale
) {
    flash_attention_v3_preprocess_bwd_fp8_impl<128>(dO, O, D, batch_size, num_heads, seq_len_q, 128, dO_scale, O_scale);
}

// Main backward kernel for FP8
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__device__ void flash_attention_v3_bwd_fp8_kernel(
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
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int k_block_idx = blockIdx.y;

    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const boostr_fp8_e4m3* Q_base = Q + qkv_offset;
    const boostr_fp8_e4m3* dO_base = dO + qkv_offset;
    boostr_fp8_e4m3* dQ_base = dQ + qkv_offset;
    float* LSE_base = (float*)(LSE + (batch_idx * num_heads + head_idx) * seq_len_q);
    float* D_base = (float*)(D + (batch_idx * num_heads + head_idx) * seq_len_q);

    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const boostr_fp8_e4m3* K_base = K + k_offset;
    const boostr_fp8_e4m3* V_base = V + k_offset;
    boostr_fp8_e4m3* dK_base = dK + k_offset;
    boostr_fp8_e4m3* dV_base = dV + k_offset;

    const int k_start = k_block_idx * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_size = k_end - k_start;

    // Dynamic shared memory for FP8
    extern __shared__ boostr_fp8_e4m3 smem_fp8[];
    boostr_fp8_e4m3* K_smem = smem_fp8;
    boostr_fp8_e4m3* V_smem = smem_fp8 + BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* Q_smem = smem_fp8 + 2 * BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* dO_smem = smem_fp8 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    // Load K and V tiles
    for (int i = tid; i < k_size * HEAD_DIM; i += blockDim.x) {
        const int k_row = i / HEAD_DIM;
        const int k_col = i % HEAD_DIM;
        K_smem[k_row * HEAD_DIM + k_col] = K_base[(k_start + k_row) * HEAD_DIM + k_col];
        V_smem[k_row * HEAD_DIM + k_col] = V_base[(k_start + k_row) * HEAD_DIM + k_col];
    }
    __syncthreads();

    const int num_q_blocks = (seq_len_q + BLOCK_M - 1) / BLOCK_M;
    int q_block_start = 0;
    if (causal) {
        q_block_start = k_start / BLOCK_M;
    }

    // FP32 accumulators for dK and dV
    float dK_local[HEAD_DIM];
    float dV_local[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dK_local[d] = 0.0f;
        dV_local[d] = 0.0f;
    }

    // Iterate over Q blocks
    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_size = q_end - q_start;

        // Load Q and dO tiles
        for (int i = tid; i < q_size * HEAD_DIM; i += blockDim.x) {
            const int q_row = i / HEAD_DIM;
            const int q_col = i % HEAD_DIM;
            Q_smem[q_row * HEAD_DIM + q_col] = Q_base[(q_start + q_row) * HEAD_DIM + q_col];
            dO_smem[q_row * HEAD_DIM + q_col] = dO_base[(q_start + q_row) * HEAD_DIM + q_col];
        }
        __syncthreads();

        // Process each query position
        for (int q_row = tid; q_row < q_size; q_row += blockDim.x) {
            const int q_global = q_start + q_row;

            const float lse_val = LSE_base[q_global];
            const float d_val = D_base[q_global];

            for (int k_row = 0; k_row < k_size; ++k_row) {
                const int k_global = k_start + k_row;

                if (causal && q_global < k_global) {
                    continue;
                }

                // Dequantize FP8 → FP32 and recompute attention score
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float q_val = fp8_e4m3_to_f32(Q_smem[q_row * HEAD_DIM + d], Q_scale);
                    float k_val = fp8_e4m3_to_f32(K_smem[k_row * HEAD_DIM + d], K_scale);
                    score += q_val * k_val;
                }
                score *= scale;

                const float prob = expf(score - lse_val);

                // Compute dP in FP32
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float dO_val = fp8_e4m3_to_f32(dO_smem[q_row * HEAD_DIM + d], dO_scale);
                    float v_val = fp8_e4m3_to_f32(V_smem[k_row * HEAD_DIM + d], V_scale);
                    dp += dO_val * v_val;
                }

                const float ds = prob * (dp - d_val) * scale;

                // Accumulate dV and dK in FP32
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float dO_val = fp8_e4m3_to_f32(dO_smem[q_row * HEAD_DIM + d], dO_scale);
                    float q_val = fp8_e4m3_to_f32(Q_smem[q_row * HEAD_DIM + d], Q_scale);
                    dV_local[d] += prob * dO_val;
                    dK_local[d] += ds * q_val;
                }

                // Compute dQ and write with atomic using dtype-safe helper
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float k_val = fp8_e4m3_to_f32(K_smem[k_row * HEAD_DIM + d], K_scale);
                    const float dq_val = ds * k_val;
                    atomic_add_dtype(&dQ_base[q_global * HEAD_DIM + d], dq_val);
                }
            }
        }

        __syncthreads();
    }

    // Write final dK and dV (convert FP32 -> FP8)
    for (int k_row = tid; k_row < k_size; k_row += blockDim.x) {
        const int k_global = k_start + k_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[k_global * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dK_local[d], dK_scale));
            dV_base[k_global * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dV_local[d], dV_scale));
        }
    }
}

extern "C" __global__ void flash_attention_v3_bwd_fp8_64(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale,
    const float dO_scale, const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_v3_bwd_fp8_kernel<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_v3_bwd_fp8_128(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale,
    const float dO_scale, const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_v3_bwd_fp8_kernel<16, 32, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

#endif  // __CUDA_ARCH__ >= 800
