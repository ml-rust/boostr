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
// One implementation and one entry-point signature, all dtypes:
// `mqa_gqa_fwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>` serves F32, F16, BF16 and
// both FP8 formats.
//
// Shared memory:
// - Layout [Q: BLOCK_M x HEAD_STRIDE][K: BLOCK_N x HEAD_STRIDE]
//   [V: BLOCK_N x HEAD_STRIDE], HEAD_STRIDE = head_dim + 1 to avoid bank
//   conflicts. Unlike the backward layout, the forward pads by +1.
// - The ELEMENT TYPE is the tensor dtype for F32/F16/BF16, which stage
//   verbatim and convert to float on read, and `float` for FP8, which
//   dequantizes on load. So the requirement is
//   (BLOCK_M + 2*BLOCK_N) * (head_dim + 1) * sizeof(smem element),
//   and FP8 needs 4 bytes per element, not 1. `mqa_fwd_block_config` takes
//   that element size as its `elem_bytes` argument — the two must agree.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

#define SMEM_STRIDE(dim, pad) ((dim) + (pad))

// ============================================================================
// Shared-memory staging
// ============================================================================

// Element type of the Q/K/V tiles: the tensor dtype, except FP8, which cannot
// be multiplied in place and is dequantized to float at the load.
template<typename T>
using fwd_smem_t = std::conditional_t<DTypeTraits<T>::needs_scale, float, T>;

// Stage one element from global into shared memory. Non-FP8 dtypes copy
// verbatim — no conversion, no rounding — and are converted to float by
// `load_dtype` when read back out of shared memory.
template<typename T>
__device__ __forceinline__
fwd_smem_t<T> stage_dtype(const T* ptr, int idx, float scale) {
    if constexpr (DTypeTraits<T>::needs_scale) {
        return load_dtype(ptr, idx, scale);
    } else {
        return ptr[idx];
    }
}

// A template cannot redeclare `extern __shared__` per instantiation type, so
// the tiles are carved out of one raw byte array. The dynamic shared-memory
// base is the same address every dtype saw before.
extern __shared__ __align__(16) unsigned char mqa_gqa_fwd_smem_raw[];

// ============================================================================
// MQA/GQA Forward - templated implementation
//
// Grid: (batch_size * num_q_heads, ceil(seq_len_q / BLOCK_M))
// Block: BLOCK_M threads; thread `tid` owns Q row `tid` of the tile.
//
// `q_scale` / `k_scale` / `v_scale` / `o_scale` are the FP8 dequant/quant
// scales. `stage_dtype` and `store_dtype` ignore them for every non-FP8 dtype.
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void mqa_gqa_fwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
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

    using SmemT = fwd_smem_t<T>;
    SmemT* smem = reinterpret_cast<SmemT*>(mqa_gqa_fwd_smem_raw);

    SmemT* Q_smem_flat = smem;
    SmemT* K_smem_flat = smem + BLOCK_M * HEAD_STRIDE;
    SmemT* V_smem_flat = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_STRIDE + (j)]

    // Read one staged element back as float.
    #define Q_smem_f(i, j) load_dtype(Q_smem_flat, (i) * HEAD_STRIDE + (j))
    #define K_smem_f(i, j) load_dtype(K_smem_flat, (i) * HEAD_STRIDE + (j))
    #define V_smem_f(i, j) load_dtype(V_smem_flat, (i) * HEAD_STRIDE + (j))

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

    const T* Q_base = Q + q_offset;
    const T* K_base = K + kv_offset;
    const T* V_base = V + kv_offset;
    T* O_base = O + q_offset;
    float* L_base = L + lse_offset;

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = stage_dtype(Q_base, (q_start + row) * HEAD_DIM + col, q_scale);
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Per-thread accumulation, always in FP32
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
            K_smem(row, col) = stage_dtype(K_base, (k_start + row) * HEAD_DIM + col, k_scale);
            V_smem(row, col) = stage_dtype(V_base, (k_start + row) * HEAD_DIM + col, v_scale);
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (max(0, seq_len_k - seq_len_q) + q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem_f(q_row, d) * K_smem_f(j, d);
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
                if (causal && (max(0, seq_len_k - seq_len_q) + q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem_f(q_row, d) * K_smem_f(j, d);
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * V_smem_f(j, d);
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
            store_dtype(O_base, (q_start + q_row) * HEAD_DIM + d, O_local[d] * inv_l, o_scale);
        }

        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
    #undef Q_smem_f
    #undef K_smem_f
    #undef V_smem_f
}

// ============================================================================
// Kernel Entry Points - runtime block-size selection
//
// BOTH block-size variants are emitted unconditionally, as separate symbols:
//   mqa_gqa_fwd_{head_dim}_{dtype}       large blocks (128x128, 128x64)
//   mqa_gqa_fwd_{head_dim}_{dtype}_sm    small blocks (64x64, 64x32)
// `mqa_fwd_block_config` picks the symbol at runtime from the device's opt-in
// shared-memory limit; nothing here is compile-time gated on the GPU
// architecture.
//
// ONE signature for every dtype, including the four trailing quantization
// scales. `stage_dtype` and `store_dtype` ignore them for every non-FP8 dtype,
// and the launcher passes 1.0f, so only the FP8 entries read them.
// ============================================================================

#define MQA_GQA_FWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)               \
    extern "C" __global__ void mqa_gqa_fwd_##HEAD_DIM##_##SUFFIX(              \
        const T* Q, const T* K, const T* V,                                    \
        T* O, float* L,                                                        \
        const int batch_size, const int num_q_heads, const int num_kv_heads,   \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal,                                   \
        const float q_scale, const float k_scale,                              \
        const float v_scale, const float o_scale                               \
    ) {                                                                        \
        mqa_gqa_fwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>(                       \
            Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads,              \
            seq_len_q, seq_len_k, scale, causal,                               \
            q_scale, k_scale, v_scale, o_scale                                 \
        );                                                                     \
    }

// FP32 - large then small blocks
MQA_GQA_FWD_ENTRY(float, 32, 128, 128, fp32)
MQA_GQA_FWD_ENTRY(float, 64, 128, 128, fp32)
MQA_GQA_FWD_ENTRY(float, 128, 128, 64, fp32)
MQA_GQA_FWD_ENTRY(float, 32, 64, 64, fp32_sm)
MQA_GQA_FWD_ENTRY(float, 64, 64, 32, fp32_sm)
MQA_GQA_FWD_ENTRY(float, 128, 64, 32, fp32_sm)

// FP16 - large then small blocks
MQA_GQA_FWD_ENTRY(__half, 32, 128, 128, fp16)
MQA_GQA_FWD_ENTRY(__half, 64, 128, 128, fp16)
MQA_GQA_FWD_ENTRY(__half, 128, 128, 64, fp16)
MQA_GQA_FWD_ENTRY(__half, 32, 64, 64, fp16_sm)
MQA_GQA_FWD_ENTRY(__half, 64, 64, 32, fp16_sm)
MQA_GQA_FWD_ENTRY(__half, 128, 64, 32, fp16_sm)

// BF16 - large then small blocks
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 32, 128, 128, bf16)
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 64, 128, 128, bf16)
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 128, 128, 64, bf16)
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 32, 64, 64, bf16_sm)
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 64, 64, 32, bf16_sm)
MQA_GQA_FWD_ENTRY(__nv_bfloat16, 128, 64, 32, bf16_sm)

// FP8 E4M3 - large then small blocks. The FP8 tiles stage as `float`, so these
// need the same shared-memory bytes as the FP32 variants, not one quarter.
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 32, 128, 128, fp8_e4m3)
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 64, 128, 128, fp8_e4m3)
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 128, 128, 64, fp8_e4m3)
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 32, 64, 64, fp8_e4m3_sm)
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 64, 64, 32, fp8_e4m3_sm)
MQA_GQA_FWD_ENTRY(boostr_fp8_e4m3, 128, 64, 32, fp8_e4m3_sm)

// FP8 E5M2 - large then small blocks
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 32, 128, 128, fp8_e5m2)
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 64, 128, 128, fp8_e5m2)
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 128, 128, 64, fp8_e5m2)
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 32, 64, 64, fp8_e5m2_sm)
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 64, 64, 32, fp8_e5m2_sm)
MQA_GQA_FWD_ENTRY(boostr_fp8_e5m2, 128, 64, 32, fp8_e5m2_sm)

#undef MQA_GQA_FWD_ENTRY
