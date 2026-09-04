// Flash Attention v2 Forward - FP8 Kernels (separate translation unit)
//
// Split out of flash_v2.cu, which holds the general Turing-capable flash
// kernels and compiles at sm_75. These FP8 kernels need Ampere or newer, so
// this unit compiles at sm_80 (see build.rs) — no `__CUDA_ARCH__` guard, so a
// future arch mistake fails to build instead of silently dropping symbols.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// Padded shared-memory stride (same definition as flash_v2.cu).
#define SMEM_STRIDE(dim, pad) ((dim) + (pad))

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

    // Absolute position of query row 0 (see the header note); 0 on prefill.
    const int key_offset = max(0, seq_len_k - seq_len_q);

    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread owns one Q row. Threads past the tile MUST stay alive: the K/V
    // load below strides by blockDim.x, so every thread of the block is needed to
    // cover the tile. Returning early left K_smem/V_smem partly UNINITIALIZED
    // whenever q_tile_size < BLOCK_M, and the stale shared memory it read instead
    // made the output depend on whatever ran before. Same structure as
    // flash_v2.cu's `is_valid_thread`.
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

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

        if (is_valid_thread) {
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

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
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = (l_local == 0.0f) ? 1.0f : 1.0f / l_local;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            // Quantize FP32 → FP8 for output
            float out_val = O_local[d] * inv_l;
            uint8_t fp8_val = f32_to_fp8_e4m3_raw(out_val, o_scale);
            O_base[(q_start + q_row) * HEAD_DIM + d] = boostr_fp8_e4m3(fp8_val);
        }

        L_base[q_start + q_row] = m_local + __logf(l_local);
    }

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
