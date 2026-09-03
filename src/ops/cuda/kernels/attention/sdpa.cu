// Scaled Dot-Product Attention (SDPA) - Fused Kernel
// Standard O(N²) attention but fused into a single kernel
// Used for MLA where K and V have different last dimensions
//
// Forward pass:
//   scores = Q @ K^T / sqrt(d_k)
//   attn = softmax(scores, dim=-1)
//   output = attn @ V
//
// Layout:
//   Q: [B, H, S_q, D_k]
//   K: [B, H, S_kv, D_k]  (note: D_k can differ from standard attention)
//   V: [B, H, S_kv, D_v]  (D_v can differ from D_k)
//   Output: [B, H, S_q, D_v]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// Shared memory layout, identical for every instantiation below:
//   [Q: BLOCK_M x head_dim_k][K: BLOCK_N x head_dim_k][V: BLOCK_N x head_dim_v]
// No +1 bank-conflict padding, and every tile is staged as `float` even for the
// FP16/BF16 entry points, which convert on load through `load_dtype`. The
// dynamic shared memory the host requests must therefore use sizeof(float) per
// element regardless of the input dtype -- see `sdpa_smem_size` in
// src/ops/cuda/attention/mla_block_config.rs.
//
// BLOCK_M is ALSO the thread count of the block: the launcher sets
// blockDim.x = BLOCK_M and each thread owns exactly one Q row
// (`is_valid_thread = tid < q_tile_size`). A tile instantiated here is only
// correct when the launcher's block_dim matches its BLOCK_M -- both come from
// `mla_block_config::block_config`, which is the single source of the pair.

// Length of the per-thread output accumulator. `head_dim_v` indexes it with no
// bounds check, so the launcher refuses head_dim_v beyond this.
#define SDPA_MAX_HEAD_DIM_V 256

// ============================================================================
// SDPA Forward - templated implementation
//
// `T` is the storage dtype (float / __half / __nv_bfloat16). All arithmetic
// and all shared-memory staging happen in float; `load_dtype`/`store_dtype`
// from dtype_traits.cuh handle the conversion at the memory boundary.
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M), 1)
// Block: (BLOCK_M, 1, 1)
// Each thread block processes one (batch, head) pair and iterates over K/V blocks
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N>
__device__ void sdpa_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim_k,
    const int head_dim_v,
    const float scale,
    const int causal
) {
    // Shared memory for loading Q, K, V tiles (always float, see note above)
    extern __shared__ char smem[];
    float* Q_smem = (float*)smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim_k;
    float* V_smem = K_smem + BLOCK_N * head_dim_k;

    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Base pointers for this (batch, head) pair
    const int head_offset_q = batch_idx * (num_heads * seq_len_q * head_dim_k) +
                              head_idx * (seq_len_q * head_dim_k);
    const int head_offset_kv = batch_idx * (num_heads * seq_len_k * head_dim_k) +
                               head_idx * (seq_len_k * head_dim_k);
    const int head_offset_v = batch_idx * (num_heads * seq_len_k * head_dim_v) +
                              head_idx * (seq_len_k * head_dim_v);
    const int head_offset_o = batch_idx * (num_heads * seq_len_q * head_dim_v) +
                              head_idx * (seq_len_q * head_dim_v);

    const T* Q_base = Q + head_offset_q;
    const T* K_base = K + head_offset_kv;
    const T* V_base = V + head_offset_v;
    T* O_base = O + head_offset_o;

    const int tid = threadIdx.x;

    // Load Q tile into shared memory, converting to float
    for (int i = tid; i < q_tile_size * head_dim_k; i += blockDim.x) {
        const int row = i / head_dim_k;
        const int col = i % head_dim_k;
        Q_smem[row * head_dim_k + col] =
            load_dtype(Q_base, (q_start + row) * head_dim_k + col);
    }
    __syncthreads();

    // Initialize output for this thread
    const bool is_valid_thread = (tid < q_tile_size);
    float O_local[SDPA_MAX_HEAD_DIM_V];
    float m_local = -INFINITY;
    float l_local = 0.0f;

    for (int d = 0; d < head_dim_v; ++d) {
        O_local[d] = 0.0f;
    }

    // Process K/V blocks
    const int num_kv_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles, converting to float
        for (int i = tid; i < k_tile_size * head_dim_k; i += blockDim.x) {
            const int row = i / head_dim_k;
            const int col = i % head_dim_k;
            K_smem[row * head_dim_k + col] =
                load_dtype(K_base, (k_start + row) * head_dim_k + col);
        }
        for (int i = tid; i < k_tile_size * head_dim_v; i += blockDim.x) {
            const int row = i / head_dim_v;
            const int col = i % head_dim_v;
            V_smem[row * head_dim_v + col] =
                load_dtype(V_base, (k_start + row) * head_dim_v + col);
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: find max with online softmax
            float m_new = m_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                // Causal mask
                if (causal && q_pos < k_pos) continue;

                // Compute Q @ K^T
                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                m_new = fmaxf(m_new, score);
            }

            // Update O with exp(m_old - m_new)
            const float alpha = __expf(m_local - m_new);
            for (int d = 0; d < head_dim_v; ++d) {
                O_local[d] *= alpha;
            }

            // Second pass: accumulate attention weights and V
            float l_new = alpha * l_local;
            for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
                const int q_pos = q_start + tid;
                const int k_pos = k_start + k_idx;

                // Causal mask
                if (causal && q_pos < k_pos) continue;

                // Compute score
                float score = 0.0f;
                for (int d = 0; d < head_dim_k; ++d) {
                    score += Q_smem[tid * head_dim_k + d] * K_smem[k_idx * head_dim_k + d];
                }
                score *= scale;

                // Softmax weight
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                // Accumulate V
                for (int d = 0; d < head_dim_v; ++d) {
                    O_local[d] += exp_score * V_smem[k_idx * head_dim_v + d];
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    // Finalize: divide by l_local (normalization) and convert back to T
    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        for (int d = 0; d < head_dim_v; ++d) {
            store_dtype(O_base, (q_start + tid) * head_dim_v + d, O_local[d] * inv_l);
        }
    }
}

// ============================================================================
// Kernel Entry Points
//
// Two tiles per dtype, selected by `mla_block_config::block_config`:
//
//   large (BLOCK_M=128, BLOCK_N=128), unsuffixed name:
//     4 * (128*D_k + 128*D_k + 128*D_v) bytes
//     = 196608 B at D_k=D_v=128, 262144 B at D_k=192/D_v=128
//   small (BLOCK_M=64, BLOCK_N=32), `_small` suffix:
//     4 * (64*D_k + 32*D_k + 32*D_v) bytes
//     = 65536 B at D_k=D_v=128, 90112 B at D_k=192/D_v=128
//
// The small tile is what makes a DeepSeek-V2/V3-shaped MLA
// (D_k = head_dim + rope_head_dim = 192, D_v = 128) runnable at all: the
// large tile's 256KB exceeds every opt-in shared-memory limit, while 88KB
// fits a ~99KB device.
// ============================================================================

#define SDPA_ENTRY(NAME, T, BM, BN)                                            \
    extern "C" __global__ void NAME(                                           \
        const T* __restrict__ Q,                                               \
        const T* __restrict__ K,                                               \
        const T* __restrict__ V,                                               \
        T* __restrict__ O,                                                     \
        const int batch_size,                                                  \
        const int num_heads,                                                   \
        const int seq_len_q,                                                   \
        const int seq_len_k,                                                   \
        const int head_dim_k,                                                  \
        const int head_dim_v,                                                  \
        const float scale,                                                     \
        const int causal                                                       \
    ) {                                                                        \
        sdpa_impl<T, BM, BN>(                                                  \
            Q, K, V, O, batch_size, num_heads, seq_len_q, seq_len_k,           \
            head_dim_k, head_dim_v, scale, causal                              \
        );                                                                     \
    }

SDPA_ENTRY(sdpa_f32, float, 128, 128)
SDPA_ENTRY(sdpa_f16, __half, 128, 128)
SDPA_ENTRY(sdpa_bf16, __nv_bfloat16, 128, 128)

SDPA_ENTRY(sdpa_f32_small, float, 64, 32)
SDPA_ENTRY(sdpa_f16_small, __half, 64, 32)
SDPA_ENTRY(sdpa_bf16_small, __nv_bfloat16, 64, 32)

#undef SDPA_ENTRY
