// MQA/GQA Backward Pass
// Gradient routing for Multi-Query and Grouped-Query Attention
//
// Key Difference from MHA Backward:
// - MHA: Each head has independent K/V → dK/dV computed locally (no atomics)
// - MQA/GQA: Multiple Q heads share K/V heads → dK/dV must ACCUMULATE gradients from all sharing Q heads
//
// Gradient Routing:
// - dQ: Same as MHA (uses atomics, multiple K blocks contribute)
// - dK, dV: REQUIRES ATOMICS (multiple Q heads contribute to same KV head)
//   Example: MQA with 8 Q heads sharing 1 KV head → 8 CUDA blocks accumulate into same dK/dV
//
// Atomic Strategy:
// - FP32 atomics: atomicAdd (hardware support)
// - FP16/BF16: atomicAdd via __half2 or atomicCAS emulation
// - FP8: Accumulate in higher precision (BF16 on sm_80+, F32 otherwise)

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
// Atomic Add Helpers for FP16/BF16
// ============================================================================

#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ void atomicAddHalf(__half* address, __half val) {
    // TEMPORARY: Force CAS-based implementation instead of native atomicAdd
    // Native atomicAdd(__half*) seems to have issues with alignment or other problems
    // Use CAS on 32-bit aligned addresses for reliability
    unsigned int* address_as_uint = (unsigned int*)((size_t)address & ~2);
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    __half2* h2 = reinterpret_cast<__half2*>(&old);

    do {
        assumed = old;
        __half2 val_h2 = __halves2half2(val, val);
        size_t offset = (size_t)address & 2;
        if (offset) {
            h2->y = __hadd(h2->y, val);
        } else {
            h2->x = __hadd(h2->x, val);
        }
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(h2));
    } while (assumed != old);
}
#endif

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* address, __nv_bfloat16 val) {
    atomicAdd(address, val);
}
#endif

// ============================================================================
// Preprocessing Kernel: Compute D = rowsum(dO ⊙ O)
// ============================================================================

template<int HEAD_DIM>
__device__ void mqa_gqa_preprocess_bwd_impl(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_q_heads,
    const int seq_len_q
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len_q) return;

    const int offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const float* dO_row = dO + offset + q_pos * HEAD_DIM;
    const float* O_row = O + offset + q_pos * HEAD_DIM;

    // Compute row-wise dot product: D_i = sum_d(dO_i[d] * O_i[d])
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        sum += dO_row[d] * O_row[d];
    }

    const int d_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;
    D[d_offset + q_pos] = sum;
}

// Preprocessing kernel exports
extern "C" __global__ void mqa_gqa_preprocess_bwd_32_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q,
    const float scale_do, const float scale_o
) {
    // F32 preprocessing ignores scale parameters
    mqa_gqa_preprocess_bwd_impl<32>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

extern "C" __global__ void mqa_gqa_preprocess_bwd_64_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q,
    const float scale_do, const float scale_o
) {
    // F32 preprocessing ignores scale parameters
    mqa_gqa_preprocess_bwd_impl<64>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

extern "C" __global__ void mqa_gqa_preprocess_bwd_128_fp32(
    const float* dO, const float* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q,
    const float scale_do, const float scale_o
) {
    // F32 preprocessing ignores scale parameters
    mqa_gqa_preprocess_bwd_impl<128>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

// ============================================================================
// Main Backward Kernel - FP32
// ============================================================================

// MQA/GQA Backward Pass
// Grid: (batch_size * num_q_heads, num_k_blocks)
// Block: blockDim.x threads (128 or 256)
//
// CRITICAL: Each Q head processes its K/V block independently, but multiple Q heads
// share the same KV head → dK/dV accumulation requires ATOMICS
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_bwd_fp32_impl(
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
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
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

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;

    // GQA/MQA head mapping
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    // Base pointers
    const int q_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const int kv_offset = (batch_idx * num_kv_heads + kv_head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;

    const float* Q_base = Q + q_offset;
    const float* K_base = K + kv_offset;
    const float* V_base = V + kv_offset;
    const float* dO_base = dO + q_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    float* dQ_base = dQ + q_offset;
    float* dK_base = dK + kv_offset;
    float* dV_base = dV + kv_offset;

    // Load K and V tiles
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Per-thread accumulators for dK and dV
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
        q_block_start = k_block;  // Skip Q blocks before this K block
    }

    // Iterate over Q blocks
    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        // Load Q and dO tiles
        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Process all Q rows: each thread processes rows where (tid == q_row)
        // When q_tile_size > blockDim.x, threads with tid >= q_tile_size are idle
        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            // Only the assigned thread computes dQ for this Q row
            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                // Recompute QK score
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += Q_smem(q_row, d) * K_smem(k_col, d);
                }
                qk_score *= scale;

                // Recompute P = exp(score - LSE)
                const float p_val = __expf(qk_score - lse_val);

                // dP = dO @ V^T
                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += dO_smem(q_row, d) * V_smem(k_col, d);
                }

                // Softmax backward: dS = P * (dP - D) * scale
                const float ds_val = p_val * (dp_val - d_val) * scale;

                // Accumulate dQ: only the thread assigned to this Q row
                if (tid == q_row) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * K_smem(k_col, d);
                    }
                }

                // Accumulate dV: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * dO_smem(q_row, d);
                    }
                }

                // Accumulate dK: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dK_local[d] += ds_val * Q_smem(q_row, d);
                    }
                }
            }

            // Write dQ with atomics: only the thread assigned to this Q row
            if (tid == q_row) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV with ATOMICS (multiple Q heads contribute to same KV head)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAdd(&dK_base[(k_start + k_row) * HEAD_DIM + d], dK_local[d]);
            atomicAdd(&dV_base[(k_start + k_row) * HEAD_DIM + d], dV_local[d]);
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// ============================================================================
// FP16 Preprocessing Implementation (dtype-specific)
// ============================================================================

template<int HEAD_DIM>
__device__ void mqa_gqa_preprocess_bwd_fp16_impl(
    const __half* __restrict__ dO,
    const __half* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_q_heads,
    const int seq_len_q
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len_q) return;

    const int offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM + q_pos * HEAD_DIM;

    // Compute D[i] = sum_d (dO[i,d] * O[i,d])
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float do_val = __half2float(dO[offset + d]);
        float o_val = __half2float(O[offset + d]);
        sum += do_val * o_val;
    }

    const int d_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;
    D[d_offset + q_pos] = sum;
}

// ============================================================================
// FP16 Backward Implementation (dtype-specific, like forward)
// ============================================================================

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_bwd_fp16_impl(
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
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal
) {
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

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;

    // GQA/MQA head mapping
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    // Base pointers
    const int q_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const int kv_offset = (batch_idx * num_kv_heads + kv_head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;

    const __half* Q_base = Q + q_offset;
    const __half* K_base = K + kv_offset;
    const __half* V_base = V + kv_offset;
    const __half* dO_base = dO + q_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    __half* dQ_base = dQ + q_offset;
    __half* dK_base = dK + kv_offset;
    __half* dV_base = dV + kv_offset;

    // Load K and V tiles (convert F16 to F32 in shared memory)
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = __half2float(K_base[(k_start + row) * HEAD_DIM + col]);
        V_smem(row, col) = __half2float(V_base[(k_start + row) * HEAD_DIM + col]);
    }
    __syncthreads();

    // Per-thread accumulators for dK and dV
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
        q_block_start = k_block;  // Skip Q blocks before this K block
    }

    // Iterate over Q blocks
    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        // Load Q and dO tiles (convert F16 to F32 in shared memory)
        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = __half2float(Q_base[(q_start + row) * HEAD_DIM + col]);
            dO_smem(row, col) = __half2float(dO_base[(q_start + row) * HEAD_DIM + col]);
        }
        __syncthreads();

        // Process all Q rows
        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            // Only the assigned thread computes dQ for this Q row
            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                // Recompute QK score
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += Q_smem(q_row, d) * K_smem(k_col, d);
                }
                qk_score *= scale;

                // Recompute P = exp(score - LSE)
                const float p_val = __expf(qk_score - lse_val);

                // dP = dO @ V^T
                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += dO_smem(q_row, d) * V_smem(k_col, d);
                }

                // Softmax backward: dS = P * (dP - D) * scale
                const float ds_val = p_val * (dp_val - d_val) * scale;

                // Accumulate dQ: only the thread assigned to this Q row
                if (tid == q_row) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * K_smem(k_col, d);
                    }
                }

                // Accumulate dV: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * dO_smem(q_row, d);
                    }
                }

                // Accumulate dK: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dK_local[d] += ds_val * Q_smem(q_row, d);
                    }
                }
            }

            // Write dQ with atomics: only the thread assigned to this Q row
            if (tid == q_row) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAddHalf(&dQ_base[q_pos * HEAD_DIM + d], __float2half(dQ_local[d]));
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV with ATOMICS (multiple Q heads contribute to same KV head)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAddHalf(&dK_base[(k_start + k_row) * HEAD_DIM + d], __float2half(dK_local[d]));
            atomicAddHalf(&dV_base[(k_start + k_row) * HEAD_DIM + d], __float2half(dV_local[d]));
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// ============================================================================
// Kernel Exports - FP32
// ============================================================================

// ============================================================================
// Kernel Exports - Compile-time selection based on target GPU
// ============================================================================
// build.rs sets -DUSE_SMALL_BLOCKS for sm_86 and older GPUs
// Shared memory limits:
// - sm_86 (RTX 30xx): 99KB → use small blocks (64x64, 64x32)
// - sm_89+ (Hopper/Ada): 128KB+ → use large blocks (128x128, 128x64)
// ============================================================================

#ifdef USE_SMALL_BLOCKS

// Small block variants for GPUs with limited shared memory (sm_86 and older)
extern "C" __global__ void mqa_gqa_bwd_32_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=32: 64x64 blocks = 2*64*32 + 2*64*32 = 16KB
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=64: 64x32 blocks = 2*32*64 + 2*64*64 = 20KB
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<64, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=128: 64x32 blocks = 2*32*128 + 2*64*128 = 40KB
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<128, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

#else

// Large block variants for high-end GPUs (sm_89+)
extern "C" __global__ void mqa_gqa_bwd_32_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=32: 128x128 blocks
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=64: 128x128 blocks
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* LSE, const float* D,
    float* dQ, float* dK, float* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    // head_dim=128: 128x64 blocks
    // F32 kernels ignore scale parameters (no quantization)
    mqa_gqa_bwd_fp32_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

#endif

// ============================================================================
// Kernel Exports - FP16
// ============================================================================

#ifdef USE_SMALL_BLOCKS

// Small block variants for GPUs with limited shared memory (sm_86 and older)
extern "C" __global__ void mqa_gqa_bwd_32_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<64, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<128, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

#else

// Large block variants for high-end GPUs (sm_89+)
extern "C" __global__ void mqa_gqa_bwd_32_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO,
    const float* LSE, const float* D,
    __half* dQ, __half* dK, __half* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal
) {
    mqa_gqa_bwd_fp16_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal
    );
}

#endif

// ============================================================================
// Dtype-Generic Preprocessing Implementation
// ============================================================================

template<typename T, int HEAD_DIM>
__device__ void mqa_gqa_preprocess_bwd_dtype_impl(
    const T* __restrict__ dO,
    const T* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_q_heads,
    const int seq_len_q,
    const float scale_do,
    const float scale_o
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len_q) return;

    const int offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const T* dO_row = dO + offset + q_pos * HEAD_DIM;
    const T* O_row = O + offset + q_pos * HEAD_DIM;

    // Compute row-wise dot product: D_i = sum_d(dO_i[d] * O_i[d])
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float do_val = load_dtype(dO_row, d, scale_do);
        float o_val = load_dtype(O_row, d, scale_o);
        sum += do_val * o_val;
    }

    const int d_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;
    D[d_offset + q_pos] = sum;
}

// Preprocessing kernel exports - dtype-specific
// F16 preprocessing kernels (no scale parameters)
extern "C" __global__ void mqa_gqa_preprocess_bwd_32_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q
) {
    mqa_gqa_preprocess_bwd_fp16_impl<32>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

extern "C" __global__ void mqa_gqa_preprocess_bwd_64_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q
) {
    mqa_gqa_preprocess_bwd_fp16_impl<64>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

extern "C" __global__ void mqa_gqa_preprocess_bwd_128_fp16(
    const __half* dO, const __half* O, float* D,
    const int batch_size, const int num_q_heads, const int seq_len_q
) {
    mqa_gqa_preprocess_bwd_fp16_impl<128>(dO, O, D, batch_size, num_q_heads, seq_len_q);
}

// Generic dtype preprocessing kernels
#define EXPORT_PREPROCESS_BWD_DTYPE(T, HEAD_DIM, SUFFIX) \
extern "C" __global__ void mqa_gqa_preprocess_bwd_##HEAD_DIM##_##SUFFIX( \
    const T* dO, const T* O, float* D, \
    const int batch_size, const int num_q_heads, const int seq_len_q, \
    const float scale_do, const float scale_o \
) { \
    mqa_gqa_preprocess_bwd_dtype_impl<T, HEAD_DIM>(dO, O, D, batch_size, num_q_heads, seq_len_q, scale_do, scale_o); \
}

// BF16 preprocessing kernels
#if __CUDA_ARCH__ >= 800
EXPORT_PREPROCESS_BWD_DTYPE(__nv_bfloat16, 32, bf16)
EXPORT_PREPROCESS_BWD_DTYPE(__nv_bfloat16, 64, bf16)
EXPORT_PREPROCESS_BWD_DTYPE(__nv_bfloat16, 128, bf16)
#endif

// FP8 E4M3 preprocessing kernels
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e4m3, 32, fp8_e4m3)
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e4m3, 64, fp8_e4m3)
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e4m3, 128, fp8_e4m3)

// FP8 E5M2 preprocessing kernels
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e5m2, 32, fp8_e5m2)
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e5m2, 64, fp8_e5m2)
EXPORT_PREPROCESS_BWD_DTYPE(boostr_fp8_e5m2, 128, fp8_e5m2)

#undef EXPORT_PREPROCESS_BWD_DTYPE
// ============================================================================
// Dtype-Generic Main Backward Implementation
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_bwd_dtype_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ O,
    const T* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    T* __restrict__ dQ,
    T* __restrict__ dK,
    T* __restrict__ dV,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float scale_q,
    const float scale_k,
    const float scale_v,
    const float scale_o,
    const float scale_do,
    const float scale_dq,
    const float scale_dk,
    const float scale_dv
) {
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

    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;

    // GQA/MQA head mapping
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    // Base pointers
    const int q_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const int kv_offset = (batch_idx * num_kv_heads + kv_head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_q_heads + q_head_idx) * seq_len_q;

    const T* Q_base = Q + q_offset;
    const T* K_base = K + kv_offset;
    const T* V_base = V + kv_offset;
    const T* dO_base = dO + q_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    T* dQ_base = dQ + q_offset;
    T* dK_base = dK + kv_offset;
    T* dV_base = dV + kv_offset;

    // Load K and V tiles
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = load_dtype(K_base, (k_start + row) * HEAD_DIM + col, scale_k);
        V_smem(row, col) = load_dtype(V_base, (k_start + row) * HEAD_DIM + col, scale_v);
    }
    __syncthreads();

    // Per-thread accumulators for dK and dV
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
        q_block_start = k_block;  // Skip Q blocks before this K block
    }

    // Iterate over Q blocks
    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        // Load Q and dO tiles
        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem(row, col) = load_dtype(Q_base, (q_start + row) * HEAD_DIM + col, scale_q);
            dO_smem(row, col) = load_dtype(dO_base, (q_start + row) * HEAD_DIM + col, scale_do);
        }
        __syncthreads();

        // Process all Q rows: each thread processes rows where (tid == q_row)
        // When q_tile_size > blockDim.x, threads with tid >= q_tile_size are idle
        for (int q_row = 0; q_row < q_tile_size; ++q_row) {
            const int q_pos = q_start + q_row;
            if (q_pos >= seq_len_q) continue;

            const float lse_val = LSE_base[q_pos];
            const float d_val = D_base[q_pos];

            // Only the assigned thread computes dQ for this Q row
            float dQ_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dQ_local[d] = 0.0f;
            }

            for (int k_col = 0; k_col < k_tile_size; ++k_col) {
                const int k_pos = k_start + k_col;

                if (causal && q_pos < k_pos) continue;

                // Recompute QK score
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += Q_smem(q_row, d) * K_smem(k_col, d);
                }
                qk_score *= scale;

                // Recompute P = exp(score - LSE)
                const float p_val = __expf(qk_score - lse_val);

                // dP = dO @ V^T
                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += dO_smem(q_row, d) * V_smem(k_col, d);
                }

                // Softmax backward: dS = P * (dP - D) * scale
                const float ds_val = p_val * (dp_val - d_val) * scale;

                // Accumulate dQ: only the thread assigned to this Q row
                if (tid == q_row) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * K_smem(k_col, d);
                    }
                }

                // Accumulate dV: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * dO_smem(q_row, d);
                    }
                }

                // Accumulate dK: each thread accumulates for its assigned K row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dK_local[d] += ds_val * Q_smem(q_row, d);
                    }
                }
            }

            // Write dQ with atomics: only the thread assigned to this Q row
            if (tid == q_row) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomic_add_dtype(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d] * scale_dq);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV with ATOMICS (multiple Q heads contribute to same KV head)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomic_add_dtype(&dK_base[(k_start + k_row) * HEAD_DIM + d], dK_local[d] * scale_dk);
            atomic_add_dtype(&dV_base[(k_start + k_row) * HEAD_DIM + d], dV_local[d] * scale_dv);
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// ============================================================================
// Kernel Exports - BF16 with compile-time block selection
// ============================================================================

#if __CUDA_ARCH__ >= 800

#ifdef USE_SMALL_BLOCKS

// Small block BF16 kernels
extern "C" __global__ void mqa_gqa_bwd_32_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 64, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 128, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#else

// Large block BF16 kernels
extern "C" __global__ void mqa_gqa_bwd_32_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<__nv_bfloat16, 128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#endif
#endif  // __CUDA_ARCH__ >= 800

// ============================================================================
// Kernel Exports - FP8 E4M3 with compile-time block selection
// ============================================================================

#ifdef USE_SMALL_BLOCKS

// Small block FP8 E4M3 kernels
extern "C" __global__ void mqa_gqa_bwd_32_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 64, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 128, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#else

// Large block FP8 E4M3 kernels
extern "C" __global__ void mqa_gqa_bwd_32_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp8_e4m3(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO,
    const float* LSE, const float* D,
    boostr_fp8_e4m3* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e4m3, 128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#endif

// ============================================================================
// Kernel Exports - FP8 E5M2 with compile-time block selection
// ============================================================================

#ifdef USE_SMALL_BLOCKS

// Small block FP8 E5M2 kernels
extern "C" __global__ void mqa_gqa_bwd_32_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 64, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 128, 64, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#else

// Large block FP8 E5M2 kernels
extern "C" __global__ void mqa_gqa_bwd_32_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_64_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

extern "C" __global__ void mqa_gqa_bwd_128_fp8_e5m2(
    const boostr_fp8_e5m2* Q, const boostr_fp8_e5m2* K, const boostr_fp8_e5m2* V,
    const boostr_fp8_e5m2* O, const boostr_fp8_e5m2* dO,
    const float* LSE, const float* D,
    boostr_fp8_e5m2* dQ, boostr_fp8_e5m2* dK, boostr_fp8_e5m2* dV,
    const int batch_size, const int num_q_heads, const int num_kv_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float scale_q, const float scale_k, const float scale_v,
    const float scale_o, const float scale_do,
    const float scale_dq, const float scale_dk, const float scale_dv
) {
    mqa_gqa_bwd_dtype_impl<boostr_fp8_e5m2, 128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, scale, causal,
        scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
    );
}

#endif
