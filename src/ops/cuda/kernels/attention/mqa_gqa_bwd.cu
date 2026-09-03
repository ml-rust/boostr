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
// - Every kernel accumulates dQ/dK/dV into FP32 buffers with native
//   atomicAdd(float*), whatever the storage dtype of Q/K/V/dO is.
// - A low-precision atomic is a read-modify-write that re-rounds the running
//   sum on every add, so the error grows with the number of contributions.
//   GQA has many: num_q_heads / num_kv_heads blocks add into one dK/dV element,
//   and every K block adds into one dQ element.
// - The launcher casts the FP32 gradients down to the caller's dtype ONCE,
//   after the kernel, so each gradient element is rounded exactly once.
//
// One implementation and one entry-point signature, all dtypes:
// `mqa_gqa_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>` and
// `mqa_gqa_preprocess_bwd_impl<T, HEAD_DIM>` serve F32, F16, BF16 and both FP8
// formats. `load_dtype` from dtype_traits.cuh converts at the memory boundary;
// every tile is staged and every product accumulated in float.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Preprocessing Kernel: Compute D = rowsum(dO ⊙ O)
// ============================================================================

template<typename T, int HEAD_DIM>
__device__ void mqa_gqa_preprocess_bwd_impl(
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

// ============================================================================
// Preprocessing Kernel Entry Points
//
// ONE signature for every dtype, including the trailing dequant scales.
// `load_dtype` ignores them for every non-FP8 dtype, and the launcher passes
// 1.0f, so only the FP8 entries read them.
// ============================================================================

#define MQA_GQA_PREPROCESS_ENTRY(T, HEAD_DIM, SUFFIX)                          \
    extern "C" __global__ void mqa_gqa_preprocess_bwd_##HEAD_DIM##_##SUFFIX(   \
        const T* dO, const T* O, float* D,                                     \
        const int batch_size, const int num_q_heads, const int seq_len_q,      \
        const float scale_do, const float scale_o                              \
    ) {                                                                        \
        mqa_gqa_preprocess_bwd_impl<T, HEAD_DIM>(                              \
            dO, O, D, batch_size, num_q_heads, seq_len_q, scale_do, scale_o    \
        );                                                                     \
    }

MQA_GQA_PREPROCESS_ENTRY(float, 32, fp32)
MQA_GQA_PREPROCESS_ENTRY(float, 64, fp32)
MQA_GQA_PREPROCESS_ENTRY(float, 128, fp32)

MQA_GQA_PREPROCESS_ENTRY(__half, 32, fp16)
MQA_GQA_PREPROCESS_ENTRY(__half, 64, fp16)
MQA_GQA_PREPROCESS_ENTRY(__half, 128, fp16)

#if __CUDA_ARCH__ >= 800
MQA_GQA_PREPROCESS_ENTRY(__nv_bfloat16, 32, bf16)
MQA_GQA_PREPROCESS_ENTRY(__nv_bfloat16, 64, bf16)
MQA_GQA_PREPROCESS_ENTRY(__nv_bfloat16, 128, bf16)
#endif

MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e4m3, 32, fp8_e4m3)
MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e4m3, 64, fp8_e4m3)
MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e4m3, 128, fp8_e4m3)

MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e5m2, 32, fp8_e5m2)
MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e5m2, 64, fp8_e5m2)
MQA_GQA_PREPROCESS_ENTRY(boostr_fp8_e5m2, 128, fp8_e5m2)

#undef MQA_GQA_PREPROCESS_ENTRY

// ============================================================================
// Main Backward - templated implementation
//
// Grid: (batch_size * num_q_heads, num_k_blocks)
// Block: BLOCK_M threads; thread `tid` owns K row `tid` of the tile and, while
// a Q tile is resident, Q row `tid`.
//
// `dQ`, `dK` and `dV` are FP32 accumulators, NOT `T` buffers — see the atomic
// note at the top of this file. `scale_dq` / `scale_dk` / `scale_dv` scale each
// contribution before the FP32 atomic, so the cast that follows the kernel must
// not reapply them.
//
// K/V/Q/dO are staged as float regardless of `T`, so the dynamic shared memory
// the host requests is always (2*BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 4 bytes —
// see BWD_SMEM_ELEM_BYTES in src/ops/cuda/attention/mqa_gqa/block_config.rs.
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_bwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ O,
    const T* __restrict__ dO,
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
    float* dQ_base = dQ + q_offset;
    float* dK_base = dK + kv_offset;
    float* dV_base = dV + kv_offset;

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
        q_block_start = max(0, k_start - max(0, seq_len_k - seq_len_q)) / BLOCK_M;
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

                if (causal && q_pos + max(0, seq_len_k - seq_len_q) < k_pos) continue;

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
                    atomicAdd(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d] * scale_dq);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV with ATOMICS (multiple Q heads contribute to same KV head)
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAdd(&dK_base[(k_start + k_row) * HEAD_DIM + d], dK_local[d] * scale_dk);
            atomicAdd(&dV_base[(k_start + k_row) * HEAD_DIM + d], dV_local[d] * scale_dv);
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// ============================================================================
// Main Backward Entry Points - runtime block-size selection
//
// BOTH block-size variants are emitted unconditionally, as separate symbols:
//   mqa_gqa_bwd_{head_dim}_{dtype}       large blocks (128x128, 128x64)
//   mqa_gqa_bwd_{head_dim}_{dtype}_sm    small blocks (64x64, 64x32)
// `mqa_bwd_block_config` picks the symbol at runtime from the device's opt-in
// shared-memory limit; nothing here is compile-time gated on the GPU
// architecture, except the BF16 block, which needs sm_80.
//
// The gradient parameters are `float*` for every dtype: the launcher passes F32
// scratch buffers and casts them down to the caller's dtype once, afterwards.
//
// ONE signature for every dtype, including the eight trailing quantization
// scales. `load_dtype` ignores scale_q/k/v/o/do for every non-FP8 dtype, and
// scale_dq/dk/dv multiply each contribution before the FP32 atomic. The
// launcher passes 1.0f for all eight, so only the FP8 entries read them.
// ============================================================================

#define MQA_GQA_BWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)               \
    extern "C" __global__ void mqa_gqa_bwd_##HEAD_DIM##_##SUFFIX(              \
        const T* Q, const T* K, const T* V,                                    \
        const T* O, const T* dO,                                               \
        const float* LSE, const float* D,                                      \
        float* dQ, float* dK, float* dV,                                       \
        const int batch_size, const int num_q_heads, const int num_kv_heads,   \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal,                                   \
        const float scale_q, const float scale_k, const float scale_v,         \
        const float scale_o, const float scale_do,                             \
        const float scale_dq, const float scale_dk, const float scale_dv       \
    ) {                                                                        \
        mqa_gqa_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>(                       \
            Q, K, V, O, dO, LSE, D, dQ, dK, dV,                                \
            batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k,       \
            scale, causal, scale_q, scale_k, scale_v, scale_o, scale_do,       \
            scale_dq, scale_dk, scale_dv                                       \
        );                                                                     \
    }

// FP32 - large then small blocks
MQA_GQA_BWD_ENTRY(float, 32, 128, 128, fp32)
MQA_GQA_BWD_ENTRY(float, 64, 128, 128, fp32)
MQA_GQA_BWD_ENTRY(float, 128, 128, 64, fp32)
MQA_GQA_BWD_ENTRY(float, 32, 64, 64, fp32_sm)
MQA_GQA_BWD_ENTRY(float, 64, 64, 32, fp32_sm)
MQA_GQA_BWD_ENTRY(float, 128, 64, 32, fp32_sm)

// FP16 - large then small blocks
MQA_GQA_BWD_ENTRY(__half, 32, 128, 128, fp16)
MQA_GQA_BWD_ENTRY(__half, 64, 128, 128, fp16)
MQA_GQA_BWD_ENTRY(__half, 128, 128, 64, fp16)
MQA_GQA_BWD_ENTRY(__half, 32, 64, 64, fp16_sm)
MQA_GQA_BWD_ENTRY(__half, 64, 64, 32, fp16_sm)
MQA_GQA_BWD_ENTRY(__half, 128, 64, 32, fp16_sm)

// BF16 - large then small blocks
#if __CUDA_ARCH__ >= 800
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 32, 128, 128, bf16)
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 64, 128, 128, bf16)
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 128, 128, 64, bf16)
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 32, 64, 64, bf16_sm)
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 64, 64, 32, bf16_sm)
MQA_GQA_BWD_ENTRY(__nv_bfloat16, 128, 64, 32, bf16_sm)
#endif  // __CUDA_ARCH__ >= 800

// FP8 E4M3 - large then small blocks
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 32, 128, 128, fp8_e4m3)
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 64, 128, 128, fp8_e4m3)
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 128, 128, 64, fp8_e4m3)
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 32, 64, 64, fp8_e4m3_sm)
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 64, 64, 32, fp8_e4m3_sm)
MQA_GQA_BWD_ENTRY(boostr_fp8_e4m3, 128, 64, 32, fp8_e4m3_sm)

// FP8 E5M2 - large then small blocks
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 32, 128, 128, fp8_e5m2)
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 64, 128, 128, fp8_e5m2)
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 128, 128, 64, fp8_e5m2)
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 32, 64, 64, fp8_e5m2_sm)
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 64, 64, 32, fp8_e5m2_sm)
MQA_GQA_BWD_ENTRY(boostr_fp8_e5m2, 128, 64, 32, fp8_e5m2_sm)

#undef MQA_GQA_BWD_ENTRY
