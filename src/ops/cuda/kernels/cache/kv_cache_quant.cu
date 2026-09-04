// Quantized KV Cache - FP8 compression for 2x memory savings
// Reduces KV cache from F32/FP16 → FP8 for long-context inference.
// BF16 input lives in kv_cache_quant_bf16.cu, which needs sm_80.
//
// Key features:
// 1. Per-token or per-head quantization with dynamic scales
// 2. On-the-fly dequantization during attention
// 3. 2x memory savings (FP16→FP8) with minimal quality loss
// 4. Compatible with Flash Attention v2
// 5. Uses centralized dtype_traits.cuh conversions for type safety

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "kv_cache_quant.cuh"

// ============================================================================
// FP8 Quantization/Dequantization
// ============================================================================

// FP32 quantization
extern "C" __global__ void quantize_kv_fp8_per_token_fp32(
    const float* input, boostr_fp8_e4m3* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_fp8_per_token_impl<float>(input, output, scales, num_tokens, head_dim);
}

// FP16 quantization
extern "C" __global__ void quantize_kv_fp8_per_token_fp16(
    const __half* input, boostr_fp8_e4m3* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_fp8_per_token_impl<__half>(input, output, scales, num_tokens, head_dim);
}

// ============================================================================
// Flash Attention with Quantized FP8 KV Cache
// ============================================================================

// Warp-level primitives
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

// Flash Attention with FP8 quantized KV cache (FP32 Q, FP8 K/V)
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_fp8_kv_impl(
    const float* __restrict__ Q,                  // [batch, heads, seq_len_q, head_dim] FP32
    const boostr_fp8_e4m3* __restrict__ K_quant,  // [batch, heads, seq_len_k, head_dim] FP8
    const boostr_fp8_e4m3* __restrict__ V_quant,  // [batch, heads, seq_len_k, head_dim] FP8
    const float* __restrict__ K_scales,           // [batch, heads, seq_len_k] or [batch, heads]
    const float* __restrict__ V_scales,           // [batch, heads, seq_len_k] or [batch, heads]
    float* __restrict__ O,                        // [batch, heads, seq_len_q, head_dim]
    float* __restrict__ L,                        // [batch, heads, seq_len_q]
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int per_token_scales  // 1 = per-token, 0 = per-head
) {
    extern __shared__ float smem[];

    float* Q_smem_flat = smem;
    float* K_smem_flat = smem + BLOCK_M * HEAD_DIM;
    float* V_smem_flat = smem + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;

    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int kv_head_offset = batch_idx * num_heads * seq_len_k * HEAD_DIM
                              + head_idx * seq_len_k * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const float* Q_base = Q + head_offset;
    const boostr_fp8_e4m3* K_base = K_quant + kv_head_offset;
    const boostr_fp8_e4m3* V_base = V_quant + kv_head_offset;
    float* O_base = O + head_offset;
    float* L_base = L + lse_offset;

    // Scales base pointers
    const float* K_scales_base = per_token_scales
        ? (K_scales + batch_idx * num_heads * seq_len_k + head_idx * seq_len_k)
        : (K_scales + batch_idx * num_heads + head_idx);
    const float* V_scales_base = per_token_scales
        ? (V_scales + batch_idx * num_heads * seq_len_k + head_idx * seq_len_k)
        : (V_scales + batch_idx * num_heads + head_idx);

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    // Load Q tile
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

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

        // Load and dequantize K and V tiles using dtype_traits conversions
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            // Dequantize K from FP8 using centralized conversion function
            float k_scale = per_token_scales ? K_scales_base[token_idx] : K_scales_base[0];
            K_smem(row, col) = fp8_e4m3_to_f32((uint8_t)K_base[token_idx * HEAD_DIM + col], k_scale);

            // Dequantize V from FP8 using centralized conversion function
            float v_scale = per_token_scales ? V_scales_base[token_idx] : V_scales_base[0];
            V_smem(row, col) = fp8_e4m3_to_f32((uint8_t)V_base[token_idx * HEAD_DIM + col], v_scale);
        }
        __syncthreads();

        if (is_valid_thread) {
            // First pass: compute max
            float m_new = m_local;
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
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
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                const float exp_score = __expf(score - m_new);
                l_new += exp_score;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += exp_score * V_smem(j, d);
                }
            }

            m_local = m_new;
            l_local = l_new;
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const float inv_l = 1.0f / l_local;
        const int out_row = q_start + q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_base[out_row * HEAD_DIM + d] = O_local[d] * inv_l;
        }

        L_base[out_row] = m_local + __logf(l_local);
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
}

// Kernel entry points
extern "C" __global__ void flash_attention_fp8_kv_64(
    const float* Q, const boostr_fp8_e4m3* K_quant, const boostr_fp8_e4m3* V_quant,
    const float* K_scales, const float* V_scales,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    float scale, int causal, int per_token_scales
) {
    flash_attention_fp8_kv_impl<64, 128, 64>(
        Q, K_quant, V_quant, K_scales, V_scales, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        scale, causal, per_token_scales
    );
}

extern "C" __global__ void flash_attention_fp8_kv_128(
    const float* Q, const boostr_fp8_e4m3* K_quant, const boostr_fp8_e4m3* V_quant,
    const float* K_scales, const float* V_scales,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    float scale, int causal, int per_token_scales
) {
    flash_attention_fp8_kv_impl<128, 128, 64>(
        Q, K_quant, V_quant, K_scales, V_scales, O, L,
        batch_size, num_heads, seq_len_q, seq_len_k,
        scale, causal, per_token_scales
    );
}

// ============================================================================
// Per-Head Quantization (Coarser granularity, faster)
// ============================================================================

extern "C" __global__ void quantize_kv_fp8_per_head_fp32(
    const float* input, boostr_fp8_e4m3* output, float* scales,
    int num_heads, int seq_len, int head_dim
) {
    quantize_kv_fp8_per_head_impl<float>(input, output, scales, num_heads, seq_len, head_dim);
}

extern "C" __global__ void quantize_kv_fp8_per_head_fp16(
    const __half* input, boostr_fp8_e4m3* output, float* scales,
    int num_heads, int seq_len, int head_dim
) {
    quantize_kv_fp8_per_head_impl<__half>(input, output, scales, num_heads, seq_len, head_dim);
}

// ============================================================================
// INT8 Quantization/Dequantization (per-token, symmetric)
// ============================================================================

// FP32 INT8 quantization
extern "C" __global__ void quantize_kv_int8_per_token_fp32(
    const float* input, int8_t* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_int8_per_token_impl<float>(input, output, scales, num_tokens, head_dim);
}

// FP16 INT8 quantization
extern "C" __global__ void quantize_kv_int8_per_token_fp16(
    const __half* input, int8_t* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_int8_per_token_impl<__half>(input, output, scales, num_tokens, head_dim);
}

// INT8 per-token dequantization: output[i] = quantized[i] * scale[token]
extern "C" __global__ void dequantize_kv_int8_per_token_fp32(
    const int8_t* __restrict__ quantized,
    float* __restrict__ output,
    const float* __restrict__ scales,
    int num_tokens,
    int head_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    const int8_t* token_in = quantized + token_idx * head_dim;
    float* token_out = output + token_idx * head_dim;
    float scale = scales[token_idx];

    for (int i = tid; i < head_dim; i += blockDim.x) {
        token_out[i] = (float)token_in[i] * scale;
    }
}
