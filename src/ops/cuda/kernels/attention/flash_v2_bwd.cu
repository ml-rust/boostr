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
// 1. **Parallelization**: grid is (batch*head, k_block), so each K/V block is owned
//    by exactly ONE CUDA block. dK and dV therefore accumulate in FP32 registers
//    and are written with a PLAIN STORE — no atomics anywhere on the dK/dV path.
//    dQ is the only atomic (several K blocks add into the same query row).
// 2. **Shared Memory**: Dynamic allocation, adaptive block sizes per GPU
// 3. **Numerical Stability**: FP32 accumulation even for FP16/BF16 inputs
// 4. **Causal Masking**: Skip Q blocks where q_pos < k_pos (entire block skip)
//
// One implementation and one entry-point signature, all dtypes:
// `flash_attention_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>` serves F32, F16 and
// BF16. FP8 lives in the separate `flash_v2_bwd_fp8.cu`.
//
// `dQ` is ALWAYS `float*`, whatever T is: K/V blocks atomicAdd into the same
// element and CUDA has no 2-byte float atomic. The launcher casts the FP32
// accumulator down to T after the kernel completes.
//
// SHARED MEMORY — the element type is the TENSOR DTYPE, not float. K/V/Q/dO
// stage verbatim as `T` and convert to float only when read back through
// `load_dtype`, so the dynamic allocation is
// (2*BLOCK_N + 2*BLOCK_M) * HEAD_DIM * sizeof(T), with NO bank-conflict padding
// of HEAD_DIM. That matches `compute_bwd_smem` in
// `src/ops/cuda/attention/flash_utils.rs`, which multiplies by
// `dtype.size_in_bytes()`. Staging as float instead would make the F16/BF16
// kernels write twice their allocation.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Preprocessing Kernel: Compute D = rowsum(dO ⊙ O)
// ============================================================================

// Compute D_i = sum_d(dO_i[d] * O_i[d]) for each query position, in FP32.
// Used in softmax backward: dS = P * (dP - D) * scale
template<typename T, int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_impl(
    const T* __restrict__ dO,
    const T* __restrict__ O,
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
    const T* dO_row = dO + offset + q_pos * HEAD_DIM;
    const T* O_row = O + offset + q_pos * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        sum += load_dtype(dO_row, d) * load_dtype(O_row, d);
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// ============================================================================
// Main Backward Kernel
// ============================================================================

// Grid: (batch_size * num_heads, num_k_blocks)
// Block: BLOCK_N threads
// Each CUDA block processes ONE K/V block and iterates over ALL Q blocks.
template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ O,
    const T* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    T* __restrict__ dK,
    T* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int window_size
) {
    // Dynamic shared memory layout, elements of type T:
    // [K: BLOCK_N x HEAD_DIM][V: BLOCK_N x HEAD_DIM]
    // [Q: BLOCK_M x HEAD_DIM][dO: BLOCK_M x HEAD_DIM]
    // A template cannot redeclare `extern __shared__` per instantiation, so the
    // tiles are carved out of one raw byte array.
    extern __shared__ __align__(16) unsigned char flash_bwd_smem_raw[];
    T* K_smem = reinterpret_cast<T*>(flash_bwd_smem_raw);
    T* V_smem = K_smem + BLOCK_N * HEAD_DIM;
    T* Q_smem = V_smem + BLOCK_N * HEAD_DIM;
    T* dO_smem = Q_smem + BLOCK_M * HEAD_DIM;

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

    const T* Q_base = Q + head_offset;
    const T* K_base = K + kv_head_offset;
    const T* V_base = V + kv_head_offset;
    const T* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    float* dQ_base = dQ + head_offset;
    T* dK_base = dK + kv_head_offset;
    T* dV_base = dV + kv_head_offset;

    // Load K and V tiles into shared memory (stay for the whole kernel)
    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem[row * HEAD_DIM + col] = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem[row * HEAD_DIM + col] = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // FP32 accumulators for dK and dV. No atomics needed — this K/V block is
    // computed by exactly one CUDA block.
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
        // Causal: skip Q blocks fully masked. Token positions, not block indices.
        q_block_start = max(0, k_start - max(0, seq_len_k - seq_len_q)) / BLOCK_M;
    }

    for (int q_block = q_block_start; q_block < q_block_end; ++q_block) {
        const int q_start = q_block * BLOCK_M;
        const int q_end = min(q_start + BLOCK_M, seq_len_q);
        const int q_tile_size = q_end - q_start;

        if (q_start >= seq_len_q) break;

        for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            Q_smem[row * HEAD_DIM + col] = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Each thread computes dQ for its assigned Q rows AND accumulates
        // dK/dV for its assigned K row.
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

                // Causal mask (position-level)
                if (causal && q_pos + max(0, seq_len_k - seq_len_q) < k_pos) continue;

                // Sliding window (inclusive of the current token): query row q_pos sits at
                // absolute position q_pos + key_offset, key_offset = max(0, seq_len_k - seq_len_q).
                // Same rule as the forward kernel in flash_v2.cu.
                if (window_size > 0 && k_pos < (q_pos + max(0, seq_len_k - seq_len_q)) - window_size + 1) continue;

                // Recompute attention score: Q @ K^T, in FP32
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    qk_score += load_dtype(Q_smem, q_row * HEAD_DIM + d) *
                                load_dtype(K_smem, k_col * HEAD_DIM + d);
                }
                qk_score *= scale;

                // Recompute attention probability: P = exp(score - LSE)
                const float p_val = __expf(qk_score - lse_val);

                // dP = dO @ V^T (gradient w.r.t. post-softmax probabilities)
                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dp_val += load_dtype(dO_smem, q_row * HEAD_DIM + d) *
                              load_dtype(V_smem, k_col * HEAD_DIM + d);
                }

                // Softmax backward: dS = P * (dP - D) * scale
                const float ds_val = p_val * (dp_val - d_val) * scale;

                // Accumulate dQ += dS * K (only for tid's Q row)
                if ((q_row % (int)blockDim.x) == tid) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dQ_local[d] += ds_val * load_dtype(K_smem, k_col * HEAD_DIM + d);
                    }
                }

                // Accumulate dV += P * dO and dK += dS * Q for K row k_row
                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dV_local[d] += p_val * load_dtype(dO_smem, q_row * HEAD_DIM + d);
                        dK_local[d] += ds_val * load_dtype(Q_smem, q_row * HEAD_DIM + d);
                    }
                }
            }

            // Write dQ with atomic adds into the FP32 accumulator
            if ((q_row % (int)blockDim.x) == tid) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Write dK and dV with a plain store — this K block is owned by one CUDA block.
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            store_dtype(dK_base, (k_start + k_row) * HEAD_DIM + d, dK_local[d]);
            store_dtype(dV_base, (k_start + k_row) * HEAD_DIM + d, dV_local[d]);
        }
    }
}

// ============================================================================
// Kernel Entry Points
//
// Three symbol families per dtype, over head_dim in {32, 64, 96, 128, 192, 256}:
//   flash_attention_preprocess_bwd_{head_dim}_{dtype}
//   flash_attention_bwd_{head_dim}_{dtype}          large block config
//   flash_attention_bwd_{head_dim}_sm_{dtype}       small block config
//
// The `_sm` entries use the same template with smaller BLOCK_M/BLOCK_N so the
// layout [K|V: BLOCK_N x HEAD_DIM][Q|dO: BLOCK_M x HEAD_DIM] fits GPUs with a
// limited opt-in shared-memory budget. `bwd_block_config` in
// `src/ops/cuda/attention/flash_utils.rs` picks the symbol and computes the
// allocation — keep the block sizes below in sync with it.
//
// Nothing here is compile-time gated on the GPU architecture: this translation
// unit builds at sm_75, and the BF16 entries must exist there. All BF16 work
// goes through `load_dtype` / `store_dtype`, whose conversions have a valid
// path at every architecture.
// ============================================================================

#define FLASH_BWD_PREPROCESS_ENTRY(T, HEAD_DIM, SUFFIX)                        \
    extern "C" __global__ void                                                 \
    flash_attention_preprocess_bwd_##HEAD_DIM##_##SUFFIX(                      \
        const T* dO, const T* O, float* D,                                     \
        const int batch_size, const int num_heads, const int seq_len           \
    ) {                                                                        \
        flash_attention_preprocess_bwd_impl<T, HEAD_DIM>(                      \
            dO, O, D, batch_size, num_heads, seq_len                           \
        );                                                                     \
    }

#define FLASH_BWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)                 \
    extern "C" __global__ void flash_attention_bwd_##HEAD_DIM##_##SUFFIX(      \
        const T* Q, const T* K, const T* V,                                    \
        const T* O, const T* dO, const float* LSE, const float* D,             \
        float* dQ, T* dK, T* dV,                                               \
        const int batch_size, const int num_heads,                             \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal, const int window_size             \
    ) {                                                                        \
        flash_attention_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>(               \
            Q, K, V, O, dO, LSE, D, dQ, dK, dV,                                \
            batch_size, num_heads, seq_len_q, seq_len_k,                       \
            scale, causal, window_size                                         \
        );                                                                     \
    }

// --- FP32 ---
FLASH_BWD_PREPROCESS_ENTRY(float, 32, fp32)
FLASH_BWD_PREPROCESS_ENTRY(float, 64, fp32)
FLASH_BWD_PREPROCESS_ENTRY(float, 96, fp32)
FLASH_BWD_PREPROCESS_ENTRY(float, 128, fp32)
FLASH_BWD_PREPROCESS_ENTRY(float, 192, fp32)
FLASH_BWD_PREPROCESS_ENTRY(float, 256, fp32)

FLASH_BWD_ENTRY(float, 32, 128, 128, fp32)
FLASH_BWD_ENTRY(float, 64, 128, 128, fp32)
FLASH_BWD_ENTRY(float, 96, 64, 128, fp32)
FLASH_BWD_ENTRY(float, 128, 128, 64, fp32)
FLASH_BWD_ENTRY(float, 192, 64, 64, fp32)
FLASH_BWD_ENTRY(float, 256, 64, 64, fp32)

FLASH_BWD_ENTRY(float, 32, 64, 64, sm_fp32)
FLASH_BWD_ENTRY(float, 64, 64, 64, sm_fp32)
FLASH_BWD_ENTRY(float, 96, 32, 32, sm_fp32)
FLASH_BWD_ENTRY(float, 128, 32, 32, sm_fp32)
FLASH_BWD_ENTRY(float, 192, 16, 16, sm_fp32)
FLASH_BWD_ENTRY(float, 256, 16, 16, sm_fp32)

// --- FP16 (FP16 I/O, FP32 accumulation) ---
FLASH_BWD_PREPROCESS_ENTRY(__half, 32, fp16)
FLASH_BWD_PREPROCESS_ENTRY(__half, 64, fp16)
FLASH_BWD_PREPROCESS_ENTRY(__half, 96, fp16)
FLASH_BWD_PREPROCESS_ENTRY(__half, 128, fp16)
FLASH_BWD_PREPROCESS_ENTRY(__half, 192, fp16)
FLASH_BWD_PREPROCESS_ENTRY(__half, 256, fp16)

FLASH_BWD_ENTRY(__half, 32, 128, 128, fp16)
FLASH_BWD_ENTRY(__half, 64, 128, 128, fp16)
FLASH_BWD_ENTRY(__half, 96, 64, 128, fp16)
FLASH_BWD_ENTRY(__half, 128, 128, 64, fp16)
FLASH_BWD_ENTRY(__half, 192, 64, 64, fp16)
FLASH_BWD_ENTRY(__half, 256, 64, 64, fp16)

FLASH_BWD_ENTRY(__half, 32, 64, 64, sm_fp16)
FLASH_BWD_ENTRY(__half, 64, 64, 64, sm_fp16)
FLASH_BWD_ENTRY(__half, 96, 32, 32, sm_fp16)
FLASH_BWD_ENTRY(__half, 128, 32, 32, sm_fp16)
FLASH_BWD_ENTRY(__half, 192, 16, 16, sm_fp16)
FLASH_BWD_ENTRY(__half, 256, 16, 16, sm_fp16)

// --- BF16 (BF16 I/O, FP32 accumulation) ---
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 32, bf16)
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 64, bf16)
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 96, bf16)
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 128, bf16)
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 192, bf16)
FLASH_BWD_PREPROCESS_ENTRY(__nv_bfloat16, 256, bf16)

FLASH_BWD_ENTRY(__nv_bfloat16, 32, 128, 128, bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 64, 128, 128, bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 96, 64, 128, bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 128, 128, 64, bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 192, 64, 64, bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 256, 64, 64, bf16)

FLASH_BWD_ENTRY(__nv_bfloat16, 32, 64, 64, sm_bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 64, 64, 64, sm_bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 96, 32, 32, sm_bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 128, 32, 32, sm_bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 192, 16, 16, sm_bf16)
FLASH_BWD_ENTRY(__nv_bfloat16, 256, 16, 16, sm_bf16)

#undef FLASH_BWD_PREPROCESS_ENTRY
#undef FLASH_BWD_ENTRY
