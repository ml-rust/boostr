// Flash Attention v2 - Production Implementation
// Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// Tri Dao, 2023 (https://arxiv.org/abs/2307.08691)
//
// Implementation matches PyTorch Flash Attention reference (research/flash-attention-main/)
//
// Key properties:
// 1. Padded shared memory strides (eliminates bank conflicts on power-of-2 dimensions)
// 2. Register-based FP32 accumulation for numerical stability, whatever T is
// 3. Native GQA: num_kv_heads can be less than num_heads (KV heads are broadcast)
// 4. Native sliding window: window_size restricts attention to local context
// 5. All head dimensions (32, 64, 96, 128, 192, 256)
//
// One implementation and one entry-point signature, all dtypes:
// `flash_attention_fwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>` serves F32, F16 and
// BF16. FP8 lives in the separate `flash_v2_fp8.cu`.
//
// Query positions are ABSOLUTE: query row `i` sits at sequence position
// `key_offset + i`, where `key_offset = seq_len_k - seq_len_q`. A KV-cached
// decode or chunked prefill passes seq_len_q < seq_len_k, and those queries are
// the LAST seq_len_q positions of the key sequence, so causal and sliding-window
// masking must both use that absolute position. Prefill (seq_len_q == seq_len_k)
// gives key_offset == 0 and leaves the masks unchanged. Same convention as
// `ops/impl_generic/attention/flash_standard.rs::build_attention_mask`.
//
// SHARED MEMORY - the element type is the TENSOR DTYPE, not float. Q/K/V stage
// verbatim as `T` and convert to float only when read back through `load_dtype`,
// so the dynamic allocation is
// (BLOCK_M + 2*BLOCK_N) * (HEAD_DIM + 1) * sizeof(T). The `+1` is bank-conflict
// padding on the head dimension and IS part of the allocation, unlike the
// backward layout. That matches `compute_smem` in
// `src/ops/cuda/attention/flash_utils.rs`, which multiplies by
// `dtype.size_in_bytes()`. Staging as float instead would make the F16/BF16
// kernels write twice their allocation.
//
// BLOCK_M is ALSO the thread count of the block: the launcher sets
// blockDim.x = BLOCK_M and each thread owns exactly one Q row
// (`is_valid_thread = q_row < q_tile_size`). A tile instantiated here is only
// correct when the launcher's block_dim matches its BLOCK_M - both come from
// `flash_utils::block_config`, the single source of the pair.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Flash Attention forward - templated implementation
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M))
// Block: BLOCK_M threads
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
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
    // Padded stride (+1) avoids bank conflicts on power-of-2 head dims.
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    // Dynamic shared memory layout, elements of type T:
    // [Q: BLOCK_M x HEAD_STRIDE][K: BLOCK_N x HEAD_STRIDE][V: BLOCK_N x HEAD_STRIDE]
    // A template cannot redeclare `extern __shared__` per instantiation, so the
    // tiles are carved out of one raw byte array.
    extern __shared__ __align__(16) unsigned char flash_fwd_smem_raw[];
    T* Q_smem_flat = reinterpret_cast<T*>(flash_fwd_smem_raw);
    T* K_smem_flat = Q_smem_flat + BLOCK_M * HEAD_STRIDE;
    T* V_smem_flat = K_smem_flat + BLOCK_N * HEAD_STRIDE;

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // GQA: Map query head to KV head (multiple Q heads share one KV head)
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Base pointers for this (batch, head). Q/O use num_heads, K/V use num_kv_heads.
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_kv_heads * seq_len_k * HEAD_DIM
                              + kv_head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const T* Q_base = Q + head_offset;
    const T* K_base = K + kv_head_offset;
    const T* V_base = V + kv_head_offset;
    T* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    // Q tile indices
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Absolute position of query row 0 (see the header note); 0 on prefill.
    const int key_offset = max(0, seq_len_k - seq_len_q);

    // Load Q tile into shared memory (all threads cooperate)
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem_flat[row * HEAD_STRIDE + col] = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread processes one Q row
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Per-thread accumulation in FP32 registers (all threads initialize, only
    // valid ones compute). FP32 accumulation regardless of T.
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

        // Sliding window optimization: skip entire K blocks outside the window.
        // Skip this K block only if it is outside the window for EVERY query in
        // the tile. That is governed by the FIRST query row, whose window reaches
        // furthest back - not the last, which reaches back the least. The
        // condition is block-uniform, so the `continue` past the `__syncthreads()`
        // below is taken by every thread of the block.
        if (window_size > 0) {
            int first_q_pos = key_offset + q_start;
            int min_k_needed = max(0, first_q_pos - window_size + 1);
            if (k_end - 1 < min_k_needed) {
                continue;  // Skip this K block entirely - outside window for all Q
            }
        }

        // Load K and V tiles into shared memory (ALL threads cooperate - critical!)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            K_smem_flat[row * HEAD_STRIDE + col] = K_base[(k_start + row) * HEAD_DIM + col];
            V_smem_flat[row * HEAD_STRIDE + col] = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Only valid threads (q_row < q_tile_size) compute attention
        if (is_valid_thread) {
            // First pass: compute max over this K tile
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = key_offset + q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

                // Compute Q @ K^T score in FP32
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += load_dtype(Q_smem_flat, q_row * HEAD_STRIDE + d) *
                             load_dtype(K_smem_flat, j * HEAD_STRIDE + d);
                }
                score *= scale;
                m_new = fmaxf(m_new, score);
            }

            // Online-softmax correction. A K block kept by the block-level skip can
            // still be fully masked for THIS query row; if it is also the row's
            // first block, m_new == m_local == -INFINITY and __expf(-inf - -inf) is
            // __expf(NaN) = NaN, poisoning O_local/l_local for the rest of the
            // kernel. alpha = 1 makes such a block an exact no-op (the second pass
            // below runs zero unmasked iterations). Any unmasked position yields a
            // finite score, so m_new == -INFINITY identifies exactly that case.
            // This must NOT become a `continue`: the update sits inside
            // `if (is_valid_thread)` and the loop ends with a `__syncthreads()`
            // outside it, so a per-thread divergent skip would break the barrier.
            const float alpha = (m_new == -INFINITY) ? 1.0f : __expf(m_local - m_new);

            // Rescale previous output in registers
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate weighted values and update l_local
            float l_new = alpha * l_local;
            for (int j = 0; j < k_tile_size; ++j) {
                const int q_pos = key_offset + q_start + q_row;
                const int k_pos = k_start + j;

                // Causal masking: skip if q_pos < k_pos
                if (causal && q_pos < k_pos) continue;

                // Sliding window: skip if k_pos < q_pos - window_size + 1
                if (window_size > 0 && k_pos < q_pos - window_size + 1) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += load_dtype(Q_smem_flat, q_row * HEAD_STRIDE + d) *
                             load_dtype(K_smem_flat, j * HEAD_STRIDE + d);
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                // Accumulate weighted V values in registers
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * load_dtype(V_smem_flat, j * HEAD_STRIDE + d);
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
            store_dtype(O_base, (q_start + q_row) * HEAD_DIM + d, O_local[d] * inv_l);
        }

        // Write logsumexp (for backward pass)
        L_base[q_start + q_row] = m_local + __logf(l_local);
    }
}

// ============================================================================
// Kernel Entry Points
//
// Two symbol families per dtype, over head_dim in {32, 64, 96, 128, 192, 256}:
//   flash_attention_fwd_{head_dim}_{dtype}          large block config
//   flash_attention_fwd_{head_dim}_sm_{dtype}       small block config
//
// The `_sm` entries use the same template with smaller BLOCK_M/BLOCK_N so the
// layout [Q: BLOCK_M x HEAD_STRIDE][K|V: BLOCK_N x HEAD_STRIDE] fits GPUs with a
// limited opt-in shared-memory budget. `block_config` in
// `src/ops/cuda/attention/flash_utils.rs` picks the symbol and computes the
// allocation - keep the block sizes below in sync with `block_config_large` and
// `block_config_small` there. A small config exists only for head_dim
// 96/128/192/256, matching `block_config_small`; 32 and 64 have none, and the
// launcher never builds an `_sm` name for them.
//
// Nothing here is compile-time gated on the GPU architecture: this translation
// unit builds at sm_75, and the BF16 entries must exist there. All BF16 work
// goes through `load_dtype` / `store_dtype`, whose conversions have a valid
// path at every architecture.
// ============================================================================

#define FLASH_FWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)                 \
    extern "C" __global__ void flash_attention_fwd_##HEAD_DIM##_##SUFFIX(      \
        const T* Q, const T* K, const T* V,                                    \
        T* O, float* L,                                                        \
        const int batch_size, const int num_heads, const int num_kv_heads,     \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal, const int window_size             \
    ) {                                                                        \
        flash_attention_fwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>(               \
            Q, K, V, O, L, batch_size, num_heads, num_kv_heads,                \
            seq_len_q, seq_len_k, scale, causal, window_size                   \
        );                                                                     \
    }

// --- FP32 ---
FLASH_FWD_ENTRY(float, 32, 128, 128, fp32)
FLASH_FWD_ENTRY(float, 64, 128, 128, fp32)
FLASH_FWD_ENTRY(float, 96, 64, 128, fp32)
FLASH_FWD_ENTRY(float, 128, 128, 64, fp32)
FLASH_FWD_ENTRY(float, 192, 64, 64, fp32)
FLASH_FWD_ENTRY(float, 256, 64, 64, fp32)

FLASH_FWD_ENTRY(float, 96, 32, 32, sm_fp32)
FLASH_FWD_ENTRY(float, 128, 64, 32, sm_fp32)
FLASH_FWD_ENTRY(float, 192, 32, 16, sm_fp32)
FLASH_FWD_ENTRY(float, 256, 16, 16, sm_fp32)

// --- FP16 (FP16 I/O, FP32 accumulation) ---
FLASH_FWD_ENTRY(__half, 32, 128, 128, fp16)
FLASH_FWD_ENTRY(__half, 64, 128, 128, fp16)
FLASH_FWD_ENTRY(__half, 96, 64, 128, fp16)
FLASH_FWD_ENTRY(__half, 128, 128, 64, fp16)
FLASH_FWD_ENTRY(__half, 192, 64, 64, fp16)
FLASH_FWD_ENTRY(__half, 256, 64, 64, fp16)

FLASH_FWD_ENTRY(__half, 96, 32, 32, sm_fp16)
FLASH_FWD_ENTRY(__half, 128, 64, 32, sm_fp16)
FLASH_FWD_ENTRY(__half, 192, 32, 16, sm_fp16)
FLASH_FWD_ENTRY(__half, 256, 16, 16, sm_fp16)

// --- BF16 (BF16 I/O, FP32 accumulation) ---
FLASH_FWD_ENTRY(__nv_bfloat16, 32, 128, 128, bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 64, 128, 128, bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 96, 64, 128, bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 128, 128, 64, bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 192, 64, 64, bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 256, 64, 64, bf16)

FLASH_FWD_ENTRY(__nv_bfloat16, 96, 32, 32, sm_bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 128, 64, 32, sm_bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 192, 32, 16, sm_bf16)
FLASH_FWD_ENTRY(__nv_bfloat16, 256, 16, 16, sm_bf16)

#undef FLASH_FWD_ENTRY
