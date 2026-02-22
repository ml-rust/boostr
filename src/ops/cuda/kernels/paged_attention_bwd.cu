// Paged Attention Backward Pass - vLLM-style non-contiguous KV cache
// Based on Flash Attention v2 backward with paged KV cache support
//
// Computes gradients for Q, K, V given gradient of output dO
// Uses block table indirection for non-contiguous KV cache access
//
// Key difference from standard Flash Attention backward:
// - K and V are stored in non-contiguous blocks
// - Block table maps logical positions to physical block addresses
// - Gradients dK and dV are accumulated using atomics for shared blocks

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

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
// Atomic Add for FP16
// ============================================================================

__device__ __forceinline__ void atomicAddHalf(__half* address, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(address, __float2half(val));
#else
    // Fallback for older architectures
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        hsum = __float2half_rn(__half2float(hsum) + val);
        old = (size_t)address & 2
            ? (old & 0xffff) | (hsum.x << 16)
            : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#endif
}

__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* address, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(address, __float2bfloat16(val));
#else
    // Fallback for older architectures
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short bits = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        float current = __uint_as_float(((unsigned int)bits) << 16);
        float sum = current + val;
        unsigned short new_bits = __float_as_uint(sum) >> 16;
        old = (size_t)address & 2
            ? (old & 0xffff) | (((unsigned int)new_bits) << 16)
            : (old & 0xffff0000) | new_bits;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#endif
}

// ============================================================================
// FP32 Paged Flash Attention Backward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_bwd_fp32_impl(
    const float* __restrict__ Q,           // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ K_blocks,    // [num_blocks, block_size, head_dim]
    const float* __restrict__ V_blocks,    // [num_blocks, block_size, head_dim]
    const float* __restrict__ O,           // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ dO,          // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ L,           // [batch, num_heads, seq_len_q] - logsumexp from forward
    const int* __restrict__ block_table,   // [batch, max_num_blocks]
    float* __restrict__ dQ,                // [batch, num_heads, seq_len_q, head_dim]
    float* __restrict__ dK_blocks,         // [num_blocks, block_size, head_dim]
    float* __restrict__ dV_blocks,         // [num_blocks, block_size, head_dim]
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
    float* Q_smem = smem;
    float* K_smem = smem + BLOCK_M * HEAD_DIM;
    float* V_smem = smem + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;
    float* dO_smem = smem + BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;
    float* O_smem = smem + 2 * BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;

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
    const float* O_base = O + head_offset;
    const float* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    float* dQ_base = dQ + head_offset;

    // Q tile indices
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q, O, dO tiles into shared memory
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem[row * HEAD_DIM + col] = Q_base[(q_start + row) * HEAD_DIM + col];
        O_smem[row * HEAD_DIM + col] = O_base[(q_start + row) * HEAD_DIM + col];
        dO_smem[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread handles one Q row
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Get logsumexp for this row
    float lse_val = 0.0f;
    if (is_valid_thread) {
        lse_val = L_base[q_start + q_row];
    }

    // Per-thread dQ accumulator
    float dQ_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dQ_local[d] = 0.0f;
    }

    // Compute D = rowsum(dO * O) for softmax backward
    float D_local = 0.0f;
    if (is_valid_thread) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            D_local += dO_smem[q_row * HEAD_DIM + d] * O_smem[q_row * HEAD_DIM + d];
        }
    }

    // Iterate over K/V tiles
    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles from paged blocks
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem[row * HEAD_DIM + col] = K_blocks[kv_offset + col];
            V_smem[row * HEAD_DIM + col] = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            // Compute attention scores and gradients for this tile
            for (int j = 0; j < k_tile_size; ++j) {
                const int k_idx = k_start + j;
                if (causal && (q_start + q_row) < k_idx) continue;

                // Compute Q @ K^T
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem[q_row * HEAD_DIM + d] * K_smem[j * HEAD_DIM + d];
                }
                score *= scale;

                // Compute softmax(score) using stored logsumexp
                float p = __expf(score - lse_val);

                // Compute dP = dO @ V^T
                float dP = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += dO_smem[q_row * HEAD_DIM + d] * V_smem[j * HEAD_DIM + d];
                }

                // Softmax backward: dS = P * (dP - D)
                float dS = p * (dP - D_local) * scale;

                // Accumulate dQ = dS @ K
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += dS * K_smem[j * HEAD_DIM + d];
                }

                // Accumulate dK and dV using atomics (paged blocks may be shared)
                const int kv_offset = get_paged_kv_offset(
                    block_table, batch_idx, max_num_blocks, k_idx, block_size, HEAD_DIM
                );

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    // dK += dS * Q
                    atomicAdd(&dK_blocks[kv_offset + d], dS * Q_smem[q_row * HEAD_DIM + d]);
                    // dV += P * dO
                    atomicAdd(&dV_blocks[kv_offset + d], p * dO_smem[q_row * HEAD_DIM + d]);
                }
            }
        }
        __syncthreads();
    }

    // Write dQ output
    if (is_valid_thread) {
        const int out_row = q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dQ_base[out_row * HEAD_DIM + d] = dQ_local[d];
        }
    }
}

// ============================================================================
// FP32 Kernel Entry Points
// ============================================================================

extern "C" __global__ void paged_flash_attention_bwd_64_fp32(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const float* O, const float* dO, const float* L,
    const int* block_table,
    float* dQ, float* dK_blocks, float* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_fp32_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_fp32(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const float* O, const float* dO, const float* L,
    const int* block_table,
    float* dQ, float* dK_blocks, float* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_fp32_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// FP16 Paged Flash Attention Backward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_bwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_blocks,
    const __half* __restrict__ V_blocks,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const float* __restrict__ L,
    const int* __restrict__ block_table,
    __half* __restrict__ dQ,
    __half* __restrict__ dK_blocks,
    __half* __restrict__ dV_blocks,
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

    __half* Q_smem = smem_fp16;
    __half* K_smem = smem_fp16 + BLOCK_M * HEAD_DIM;
    __half* V_smem = smem_fp16 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;
    __half* dO_smem = smem_fp16 + BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;
    __half* O_smem = smem_fp16 + 2 * BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __half* Q_base = Q + head_offset;
    const __half* O_base = O + head_offset;
    const __half* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    __half* dQ_base = dQ + head_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q, O, dO tiles
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem[row * HEAD_DIM + col] = Q_base[(q_start + row) * HEAD_DIM + col];
        O_smem[row * HEAD_DIM + col] = O_base[(q_start + row) * HEAD_DIM + col];
        dO_smem[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    float lse_val = 0.0f;
    if (is_valid_thread) {
        lse_val = L_base[q_start + q_row];
    }

    float dQ_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dQ_local[d] = 0.0f;
    }

    // Compute D = rowsum(dO * O)
    float D_local = 0.0f;
    if (is_valid_thread) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            D_local += __half2float(dO_smem[q_row * HEAD_DIM + d]) *
                       __half2float(O_smem[q_row * HEAD_DIM + d]);
        }
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
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem[row * HEAD_DIM + col] = K_blocks[kv_offset + col];
            V_smem[row * HEAD_DIM + col] = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            for (int j = 0; j < k_tile_size; ++j) {
                const int k_idx = k_start + j;
                if (causal && (q_start + q_row) < k_idx) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(Q_smem[q_row * HEAD_DIM + d]) *
                             __half2float(K_smem[j * HEAD_DIM + d]);
                }
                score *= scale;

                float p = __expf(score - lse_val);

                float dP = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += __half2float(dO_smem[q_row * HEAD_DIM + d]) *
                          __half2float(V_smem[j * HEAD_DIM + d]);
                }

                float dS = p * (dP - D_local) * scale;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += dS * __half2float(K_smem[j * HEAD_DIM + d]);
                }

                const int kv_offset = get_paged_kv_offset(
                    block_table, batch_idx, max_num_blocks, k_idx, block_size, HEAD_DIM
                );

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAddHalf(&dK_blocks[kv_offset + d],
                                  dS * __half2float(Q_smem[q_row * HEAD_DIM + d]));
                    atomicAddHalf(&dV_blocks[kv_offset + d],
                                  p * __half2float(dO_smem[q_row * HEAD_DIM + d]));
                }
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const int out_row = q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dQ_base[out_row * HEAD_DIM + d] = __float2half(dQ_local[d]);
        }
    }
}

extern "C" __global__ void paged_flash_attention_bwd_64_fp16(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const __half* O, const __half* dO, const float* L,
    const int* block_table,
    __half* dQ, __half* dK_blocks, __half* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_fp16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_fp16(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const __half* O, const __half* dO, const float* L,
    const int* block_table,
    __half* dQ, __half* dK_blocks, __half* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_fp16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// BF16 Paged Flash Attention Backward
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_flash_attention_bwd_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_blocks,
    const __nv_bfloat16* __restrict__ V_blocks,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK_blocks,
    __nv_bfloat16* __restrict__ dV_blocks,
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

    __nv_bfloat16* Q_smem = smem_bf16;
    __nv_bfloat16* K_smem = smem_bf16 + BLOCK_M * HEAD_DIM;
    __nv_bfloat16* V_smem = smem_bf16 + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;
    __nv_bfloat16* dO_smem = smem_bf16 + BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;
    __nv_bfloat16* O_smem = smem_bf16 + 2 * BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* O_base = O + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    __nv_bfloat16* dQ_base = dQ + head_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q, O, dO tiles
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem[row * HEAD_DIM + col] = Q_base[(q_start + row) * HEAD_DIM + col];
        O_smem[row * HEAD_DIM + col] = O_base[(q_start + row) * HEAD_DIM + col];
        dO_smem[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    float lse_val = 0.0f;
    if (is_valid_thread) {
        lse_val = L_base[q_start + q_row];
    }

    float dQ_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dQ_local[d] = 0.0f;
    }

    // Compute D = rowsum(dO * O)
    float D_local = 0.0f;
    if (is_valid_thread) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            D_local += __bfloat162float(dO_smem[q_row * HEAD_DIM + d]) *
                       __bfloat162float(O_smem[q_row * HEAD_DIM + d]);
        }
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
                block_table, batch_idx, max_num_blocks, token_idx, block_size, HEAD_DIM
            );

            K_smem[row * HEAD_DIM + col] = K_blocks[kv_offset + col];
            V_smem[row * HEAD_DIM + col] = V_blocks[kv_offset + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            for (int j = 0; j < k_tile_size; ++j) {
                const int k_idx = k_start + j;
                if (causal && (q_start + q_row) < k_idx) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __bfloat162float(Q_smem[q_row * HEAD_DIM + d]) *
                             __bfloat162float(K_smem[j * HEAD_DIM + d]);
                }
                score *= scale;

                float p = __expf(score - lse_val);

                float dP = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += __bfloat162float(dO_smem[q_row * HEAD_DIM + d]) *
                          __bfloat162float(V_smem[j * HEAD_DIM + d]);
                }

                float dS = p * (dP - D_local) * scale;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += dS * __bfloat162float(K_smem[j * HEAD_DIM + d]);
                }

                const int kv_offset = get_paged_kv_offset(
                    block_table, batch_idx, max_num_blocks, k_idx, block_size, HEAD_DIM
                );

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAddBF16(&dK_blocks[kv_offset + d],
                                  dS * __bfloat162float(Q_smem[q_row * HEAD_DIM + d]));
                    atomicAddBF16(&dV_blocks[kv_offset + d],
                                  p * __bfloat162float(dO_smem[q_row * HEAD_DIM + d]));
                }
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const int out_row = q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dQ_base[out_row * HEAD_DIM + d] = __float2bfloat16(dQ_local[d]);
        }
    }
}

extern "C" __global__ void paged_flash_attention_bwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    const int* block_table,
    __nv_bfloat16* dQ, __nv_bfloat16* dK_blocks, __nv_bfloat16* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_bf16_impl<64, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    const int* block_table,
    __nv_bfloat16* dQ, __nv_bfloat16* dK_blocks, __nv_bfloat16* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    paged_flash_attention_bwd_bf16_impl<128, 128, 64>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// ============================================================================
// Small Block Size Variants (fit in 48KB shared memory)
// ============================================================================
// Shared memory calculation for backward:
//   Q_smem + K_smem + V_smem + dO_smem + O_smem = (3*BLOCK_M + 2*BLOCK_N) * HEAD_DIM * sizeof(dtype)
// For 48KB (49152 bytes) limit:
//
// FP32 (4 bytes):
//   head_dim=64:  BLOCK_M=32, BLOCK_N=32 -> (96+64)*64*4 = 40960 bytes ✓
//   head_dim=128: BLOCK_M=16, BLOCK_N=16 -> (48+32)*128*4 = 40960 bytes ✓
//
// FP16/BF16 (2 bytes):
//   head_dim=64:  BLOCK_M=64, BLOCK_N=32 -> (192+64)*64*2 = 32768 bytes ✓
//   head_dim=128: BLOCK_M=32, BLOCK_N=32 -> (96+64)*128*2 = 40960 bytes ✓

// FP32 Small Variants
extern "C" __global__ void paged_flash_attention_bwd_64_fp32_small(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const float* O, const float* dO, const float* L,
    const int* block_table,
    float* dQ, float* dK_blocks, float* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=32, BLOCK_N=32: (3*32 + 2*32) * 64 * 4 = 40960 bytes
    paged_flash_attention_bwd_fp32_impl<64, 32, 32>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_fp32_small(
    const float* Q, const float* K_blocks, const float* V_blocks,
    const float* O, const float* dO, const float* L,
    const int* block_table,
    float* dQ, float* dK_blocks, float* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=16, BLOCK_N=16: (3*16 + 2*16) * 128 * 4 = 40960 bytes
    paged_flash_attention_bwd_fp32_impl<128, 16, 16>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// FP16 Small Variants
extern "C" __global__ void paged_flash_attention_bwd_64_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const __half* O, const __half* dO, const float* L,
    const int* block_table,
    __half* dQ, __half* dK_blocks, __half* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=64, BLOCK_N=32: (3*64 + 2*32) * 64 * 2 = 32768 bytes
    paged_flash_attention_bwd_fp16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_fp16_small(
    const __half* Q, const __half* K_blocks, const __half* V_blocks,
    const __half* O, const __half* dO, const float* L,
    const int* block_table,
    __half* dQ, __half* dK_blocks, __half* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=32, BLOCK_N=32: (3*32 + 2*32) * 128 * 2 = 40960 bytes
    paged_flash_attention_bwd_fp16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

// BF16 Small Variants
extern "C" __global__ void paged_flash_attention_bwd_64_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    const int* block_table,
    __nv_bfloat16* dQ, __nv_bfloat16* dK_blocks, __nv_bfloat16* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=64, BLOCK_N=32: (3*64 + 2*32) * 64 * 2 = 32768 bytes
    paged_flash_attention_bwd_bf16_impl<64, 64, 32>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}

extern "C" __global__ void paged_flash_attention_bwd_128_bf16_small(
    const __nv_bfloat16* Q, const __nv_bfloat16* K_blocks, const __nv_bfloat16* V_blocks,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    const int* block_table,
    __nv_bfloat16* dQ, __nv_bfloat16* dK_blocks, __nv_bfloat16* dV_blocks,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int max_num_blocks, int block_size, float scale, int causal
) {
    // BLOCK_M=32, BLOCK_N=32: (3*32 + 2*32) * 128 * 2 = 40960 bytes
    paged_flash_attention_bwd_bf16_impl<128, 32, 32>(
        Q, K_blocks, V_blocks, O, dO, L, block_table,
        dQ, dK_blocks, dV_blocks,
        batch_size, num_heads, seq_len_q, seq_len_k,
        max_num_blocks, block_size, scale, causal
    );
}
