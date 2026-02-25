// INT4 Quantized KV Cache - 4x memory savings for long-context inference
//
// Reduces KV cache from FP16/BF16 → INT4 for extreme context length support.
// Trade-off: Higher compression (4x vs 2x) but more quantization error.
//
// Key features:
// 1. Per-group quantization with asymmetric min-max scaling
// 2. Group size: 32, 64, or 128 elements for quality/overhead trade-off
// 3. Packed INT4: 2 values per byte for efficient memory access
// 4. On-the-fly dequantization during attention
// 5. Compatible with Flash Attention v2
//
// Memory layout:
// - INT4 values: packed as 2 per byte, low nibble first
// - Scales/zeros: one pair per group (float16 each = 4 bytes overhead per group)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cfloat>

// Use standardized dtype infrastructure
#include "dtype_traits.cuh"

// ============================================================================
// INT4 Packing/Unpacking Utilities
// ============================================================================

// Pack two INT4 values (0-15) into one byte
__device__ __forceinline__ uint8_t pack_int4(int val0, int val1) {
    return (uint8_t)((val0 & 0xF) | ((val1 & 0xF) << 4));
}

// Unpack two INT4 values from one byte
__device__ __forceinline__ void unpack_int4(uint8_t packed, int& val0, int& val1) {
    val0 = packed & 0xF;        // Low nibble
    val1 = (packed >> 4) & 0xF; // High nibble
}

// Quantize FP32 value to INT4 (0-15) with scale and zero point
// Formula: q = clamp(round((x - zero) / scale), 0, 15)
__device__ __forceinline__ int quantize_to_int4(float x, float scale, float zero) {
    float q = (x - zero) / scale;
    q = roundf(q);
    q = fmaxf(0.0f, fminf(15.0f, q));
    return (int)q;
}

// Dequantize INT4 (0-15) to FP32 with scale and zero point
// Formula: x = q * scale + zero
__device__ __forceinline__ float dequantize_from_int4(int q, float scale, float zero) {
    return (float)q * scale + zero;
}

// ============================================================================
// Per-Group INT4 Quantization
// ============================================================================

// Quantize KV cache to INT4 with per-group asymmetric min-max scaling
// Input: [num_tokens, head_dim]
// Output: [num_tokens, head_dim/2] (packed INT4) + [num_groups, 2] (scale, zero per group)
// Where num_groups = num_tokens * ceil(head_dim / group_size)

// Template implementation using dtype_traits.cuh
template<typename T>
__device__ void quantize_kv_int4_per_group_impl(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size
) {
    int token_idx = blockIdx.x;
    int group_in_token = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    int groups_per_token = (head_dim + group_size - 1) / group_size;
    if (group_in_token >= groups_per_token) return;

    int group_start = group_in_token * group_size;
    int group_end = min(group_start + group_size, head_dim);
    int group_len = group_end - group_start;

    const T* token_input = input + token_idx * head_dim + group_start;

    __shared__ float smin[256];
    __shared__ float smax[256];

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // Use load_dtype from dtype_traits.cuh for type-agnostic loading
    for (int i = tid; i < group_len; i += blockDim.x) {
        float val = load_dtype<T>(token_input, i);
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    // Parallel reduction for min/max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    __shared__ float scale;
    __shared__ float zero;

    if (tid == 0) {
        float min_val = smin[0];
        float max_val = smax[0];
        float range = max_val - min_val;

        if (range < 1e-8f) {
            scale = 1.0f;
            zero = min_val;
        } else {
            scale = range / 15.0f;
            zero = min_val;
        }

        int global_group_idx = token_idx * groups_per_token + group_in_token;
        scales[global_group_idx] = scale;
        zeros[global_group_idx] = zero;
    }
    __syncthreads();

    uint8_t* token_output = output + token_idx * (head_dim / 2) + (group_start / 2);

    // Quantize and pack pairs of values
    for (int i = tid * 2; i < group_len; i += blockDim.x * 2) {
        float val0 = 0.0f, val1 = 0.0f;

        if (i < group_len) {
            val0 = load_dtype<T>(token_input, i);
        }
        if (i + 1 < group_len) {
            val1 = load_dtype<T>(token_input, i + 1);
        }

        int q0 = quantize_to_int4(val0, scale, zero);
        int q1 = quantize_to_int4(val1, scale, zero);

        token_output[i / 2] = pack_int4(q0, q1);
    }
}

// Extern C wrappers for each dtype
extern "C" __global__ void quantize_kv_int4_per_group_fp32(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size
) {
    quantize_kv_int4_per_group_impl<float>(
        input, output, scales, zeros, num_tokens, head_dim, group_size
    );
}

extern "C" __global__ void quantize_kv_int4_per_group_fp16(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size
) {
    quantize_kv_int4_per_group_impl<__half>(
        input, output, scales, zeros, num_tokens, head_dim, group_size
    );
}

extern "C" __global__ void quantize_kv_int4_per_group_bf16(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size
) {
    quantize_kv_int4_per_group_impl<__nv_bfloat16>(
        input, output, scales, zeros, num_tokens, head_dim, group_size
    );
}

// ============================================================================
// Dequantization Kernel (for debugging/fallback)
// ============================================================================

template<typename T>
__device__ void dequantize_kv_int4_per_group_impl(
    const uint8_t* __restrict__ input,     // [num_tokens, head_dim/2] packed INT4
    const float* __restrict__ scales,      // [num_groups]
    const float* __restrict__ zeros,       // [num_groups]
    T* __restrict__ output,                // [num_tokens, head_dim]
    int num_tokens,
    int head_dim,
    int group_size
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    int groups_per_token = (head_dim + group_size - 1) / group_size;
    const uint8_t* token_input = input + token_idx * (head_dim / 2);
    T* token_output = output + token_idx * head_dim;

    for (int i = tid * 2; i < head_dim; i += blockDim.x * 2) {
        // Determine which group this element belongs to
        int group_idx = i / group_size;
        int global_group_idx = token_idx * groups_per_token + group_idx;

        float scale = scales[global_group_idx];
        float zero = zeros[global_group_idx];

        // Unpack INT4 values
        int q0, q1;
        unpack_int4(token_input[i / 2], q0, q1);

        // Dequantize
        float val0 = dequantize_from_int4(q0, scale, zero);
        float val1 = dequantize_from_int4(q1, scale, zero);

        // Store using store_dtype from dtype_traits.cuh
        if (i < head_dim) {
            store_dtype<T>(token_output, i, val0);
        }
        if (i + 1 < head_dim) {
            store_dtype<T>(token_output, i + 1, val1);
        }
    }
}

extern "C" __global__ void dequantize_kv_int4_per_group_fp32(
    const uint8_t* input, const float* scales, const float* zeros,
    float* output, int num_tokens, int head_dim, int group_size
) {
    dequantize_kv_int4_per_group_impl<float>(
        input, scales, zeros, output, num_tokens, head_dim, group_size
    );
}

extern "C" __global__ void dequantize_kv_int4_per_group_fp16(
    const uint8_t* input, const float* scales, const float* zeros,
    __half* output, int num_tokens, int head_dim, int group_size
) {
    dequantize_kv_int4_per_group_impl<__half>(
        input, scales, zeros, output, num_tokens, head_dim, group_size
    );
}

extern "C" __global__ void dequantize_kv_int4_per_group_bf16(
    const uint8_t* input, const float* scales, const float* zeros,
    __nv_bfloat16* output, int num_tokens, int head_dim, int group_size
) {
    dequantize_kv_int4_per_group_impl<__nv_bfloat16>(
        input, scales, zeros, output, num_tokens, head_dim, group_size
    );
}

// ============================================================================
// Flash Attention with INT4 Quantized KV Cache
// ============================================================================

// Warp-level primitives
__device__ __forceinline__ float warp_reduce_max_int4(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_int4(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Flash Attention with INT4 quantized KV cache
// Q in FP32, K/V in INT4 with on-the-fly dequantization
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_int4_kv_impl(
    const float* __restrict__ Q,                  // [batch, heads, seq_len_q, head_dim] FP32
    const uint8_t* __restrict__ K_quant,          // [batch, heads, seq_len_k, head_dim/2] INT4
    const uint8_t* __restrict__ V_quant,          // [batch, heads, seq_len_k, head_dim/2] INT4
    const __half* __restrict__ K_scales,          // [batch, heads, num_groups_k]
    const __half* __restrict__ K_zeros,           // [batch, heads, num_groups_k]
    const __half* __restrict__ V_scales,          // [batch, heads, num_groups_v]
    const __half* __restrict__ V_zeros,           // [batch, heads, num_groups_v]
    float* __restrict__ O,                        // [batch, heads, seq_len_q, head_dim]
    float* __restrict__ L,                        // [batch, heads, seq_len_q]
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int group_size,
    const float scale,
    const int causal
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

    // Offsets for FP32 Q and O
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;

    // Offsets for INT4 K/V (packed, so head_dim/2)
    const int kv_head_offset = batch_idx * num_heads * seq_len_k * (HEAD_DIM / 2)
                              + head_idx * seq_len_k * (HEAD_DIM / 2);

    // Offsets for scales/zeros
    const int groups_per_token = (HEAD_DIM + group_size - 1) / group_size;
    const int total_groups = seq_len_k * groups_per_token;
    const int scale_offset = batch_idx * num_heads * total_groups
                            + head_idx * total_groups;

    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const float* Q_base = Q + head_offset;
    const uint8_t* K_base = K_quant + kv_head_offset;
    const uint8_t* V_base = V_quant + kv_head_offset;
    const __half* K_scales_base = K_scales + scale_offset;
    const __half* K_zeros_base = K_zeros + scale_offset;
    const __half* V_scales_base = V_scales + scale_offset;
    const __half* V_zeros_base = V_zeros + scale_offset;
    float* O_base = O + head_offset;
    float* L_base = L + lse_offset;

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

        // Load and dequantize K and V tiles from INT4
        for (int i = tid; i < k_tile_size * (HEAD_DIM / 2); i += blockDim.x) {
            const int row = i / (HEAD_DIM / 2);
            const int packed_col = i % (HEAD_DIM / 2);
            const int token_idx = k_start + row;
            const int elem_col = packed_col * 2;

            // Determine group index for these elements
            int group_idx = elem_col / group_size;
            int global_group_idx = token_idx * groups_per_token + group_idx;

            // Get scale and zero for K
            float k_scale = __half2float(K_scales_base[global_group_idx]);
            float k_zero = __half2float(K_zeros_base[global_group_idx]);

            // Get scale and zero for V
            float v_scale = __half2float(V_scales_base[global_group_idx]);
            float v_zero = __half2float(V_zeros_base[global_group_idx]);

            // Unpack and dequantize K
            int k_q0, k_q1;
            unpack_int4(K_base[token_idx * (HEAD_DIM / 2) + packed_col], k_q0, k_q1);
            K_smem(row, elem_col) = dequantize_from_int4(k_q0, k_scale, k_zero);
            K_smem(row, elem_col + 1) = dequantize_from_int4(k_q1, k_scale, k_zero);

            // Unpack and dequantize V
            int v_q0, v_q1;
            unpack_int4(V_base[token_idx * (HEAD_DIM / 2) + packed_col], v_q0, v_q1);
            V_smem(row, elem_col) = dequantize_from_int4(v_q0, v_scale, v_zero);
            V_smem(row, elem_col + 1) = dequantize_from_int4(v_q1, v_scale, v_zero);
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
extern "C" __global__ void flash_attention_int4_kv_64(
    const float* Q, const uint8_t* K_quant, const uint8_t* V_quant,
    const __half* K_scales, const __half* K_zeros,
    const __half* V_scales, const __half* V_zeros,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int group_size, float scale, int causal
) {
    flash_attention_int4_kv_impl<64, 128, 64>(
        Q, K_quant, V_quant, K_scales, K_zeros, V_scales, V_zeros,
        O, L, batch_size, num_heads, seq_len_q, seq_len_k,
        group_size, scale, causal
    );
}

extern "C" __global__ void flash_attention_int4_kv_128(
    const float* Q, const uint8_t* K_quant, const uint8_t* V_quant,
    const __half* K_scales, const __half* K_zeros,
    const __half* V_scales, const __half* V_zeros,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int group_size, float scale, int causal
) {
    flash_attention_int4_kv_impl<128, 64, 64>(
        Q, K_quant, V_quant, K_scales, K_zeros, V_scales, V_zeros,
        O, L, batch_size, num_heads, seq_len_q, seq_len_k,
        group_size, scale, causal
    );
}

// Smaller block size version for 48KB shared memory limit
extern "C" __global__ void flash_attention_int4_kv_64_small(
    const float* Q, const uint8_t* K_quant, const uint8_t* V_quant,
    const __half* K_scales, const __half* K_zeros,
    const __half* V_scales, const __half* V_zeros,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int group_size, float scale, int causal
) {
    flash_attention_int4_kv_impl<64, 64, 32>(
        Q, K_quant, V_quant, K_scales, K_zeros, V_scales, V_zeros,
        O, L, batch_size, num_heads, seq_len_q, seq_len_k,
        group_size, scale, causal
    );
}

extern "C" __global__ void flash_attention_int4_kv_128_small(
    const float* Q, const uint8_t* K_quant, const uint8_t* V_quant,
    const __half* K_scales, const __half* K_zeros,
    const __half* V_scales, const __half* V_zeros,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int group_size, float scale, int causal
) {
    flash_attention_int4_kv_impl<128, 32, 32>(
        Q, K_quant, V_quant, K_scales, K_zeros, V_scales, V_zeros,
        O, L, batch_size, num_heads, seq_len_q, seq_len_k,
        group_size, scale, causal
    );
}

// ============================================================================
// Incremental Append with Quantization
// For autoregressive generation: quantize and append new K/V to cache
// ============================================================================

template<typename T>
__device__ void append_kv_int4_impl(
    const T* __restrict__ new_k,           // [batch, num_heads, head_dim]
    const T* __restrict__ new_v,           // [batch, num_heads, head_dim]
    uint8_t* __restrict__ k_cache,         // [batch, num_heads, max_seq, head_dim/2]
    uint8_t* __restrict__ v_cache,         // [batch, num_heads, max_seq, head_dim/2]
    __half* __restrict__ k_scales,         // [batch, num_heads, max_seq * groups_per_token]
    __half* __restrict__ k_zeros,          // [batch, num_heads, max_seq * groups_per_token]
    __half* __restrict__ v_scales,         // [batch, num_heads, max_seq * groups_per_token]
    __half* __restrict__ v_zeros,          // [batch, num_heads, max_seq * groups_per_token]
    int batch_size,
    int num_heads,
    int position,                          // Current position to write
    int head_dim,
    int max_seq_len,
    int group_size
) {
    int batch_head_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;

    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;

    if (batch_idx >= batch_size) return;

    int groups_per_token = (head_dim + group_size - 1) / group_size;
    if (group_idx >= groups_per_token) return;

    int group_start = group_idx * group_size;
    int group_end = min(group_start + group_size, head_dim);
    int group_len = group_end - group_start;

    // Get pointers for this batch/head
    const T* k_in = new_k + batch_idx * num_heads * head_dim + head_idx * head_dim + group_start;
    const T* v_in = new_v + batch_idx * num_heads * head_dim + head_idx * head_dim + group_start;

    // Cache offsets
    int cache_offset = batch_idx * num_heads * max_seq_len * (head_dim / 2)
                     + head_idx * max_seq_len * (head_dim / 2)
                     + position * (head_dim / 2)
                     + (group_start / 2);
    uint8_t* k_out = k_cache + cache_offset;
    uint8_t* v_out = v_cache + cache_offset;

    // Scale offset
    int scale_offset = batch_idx * num_heads * max_seq_len * groups_per_token
                     + head_idx * max_seq_len * groups_per_token
                     + position * groups_per_token
                     + group_idx;

    // Find min/max for K and V in this group
    __shared__ float k_min_s[64], k_max_s[64];
    __shared__ float v_min_s[64], v_max_s[64];

    float k_min = FLT_MAX, k_max = -FLT_MAX;
    float v_min = FLT_MAX, v_max = -FLT_MAX;

    // Use load_dtype from dtype_traits.cuh
    for (int i = tid; i < group_len; i += blockDim.x) {
        float kv = load_dtype<T>(k_in, i);
        float vv = load_dtype<T>(v_in, i);
        k_min = fminf(k_min, kv);
        k_max = fmaxf(k_max, kv);
        v_min = fminf(v_min, vv);
        v_max = fmaxf(v_max, vv);
    }

    if (tid < 64) {
        k_min_s[tid] = k_min;
        k_max_s[tid] = k_max;
        v_min_s[tid] = v_min;
        v_max_s[tid] = v_max;
    }
    __syncthreads();

    // Reduce
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            k_min_s[tid] = fminf(k_min_s[tid], k_min_s[tid + s]);
            k_max_s[tid] = fmaxf(k_max_s[tid], k_max_s[tid + s]);
            v_min_s[tid] = fminf(v_min_s[tid], v_min_s[tid + s]);
            v_max_s[tid] = fmaxf(v_max_s[tid], v_max_s[tid + s]);
        }
        __syncthreads();
    }

    __shared__ float k_scale, k_zero, v_scale, v_zero;
    if (tid == 0) {
        float k_range = k_max_s[0] - k_min_s[0];
        k_scale = (k_range > 1e-8f) ? (k_range / 15.0f) : 1.0f;
        k_zero = k_min_s[0];
        k_scales[scale_offset] = __float2half(k_scale);
        k_zeros[scale_offset] = __float2half(k_zero);

        float v_range = v_max_s[0] - v_min_s[0];
        v_scale = (v_range > 1e-8f) ? (v_range / 15.0f) : 1.0f;
        v_zero = v_min_s[0];
        v_scales[scale_offset] = __float2half(v_scale);
        v_zeros[scale_offset] = __float2half(v_zero);
    }
    __syncthreads();

    // Quantize and pack using load_dtype
    for (int i = tid * 2; i < group_len; i += blockDim.x * 2) {
        float kv0 = 0.0f, kv1 = 0.0f, vv0 = 0.0f, vv1 = 0.0f;

        if (i < group_len) {
            kv0 = load_dtype<T>(k_in, i);
            vv0 = load_dtype<T>(v_in, i);
        }
        if (i + 1 < group_len) {
            kv1 = load_dtype<T>(k_in, i + 1);
            vv1 = load_dtype<T>(v_in, i + 1);
        }

        int k_q0 = quantize_to_int4(kv0, k_scale, k_zero);
        int k_q1 = quantize_to_int4(kv1, k_scale, k_zero);
        int v_q0 = quantize_to_int4(vv0, v_scale, v_zero);
        int v_q1 = quantize_to_int4(vv1, v_scale, v_zero);

        k_out[i / 2] = pack_int4(k_q0, k_q1);
        v_out[i / 2] = pack_int4(v_q0, v_q1);
    }
}

extern "C" __global__ void append_kv_int4_fp32(
    const float* new_k, const float* new_v,
    uint8_t* k_cache, uint8_t* v_cache,
    __half* k_scales, __half* k_zeros,
    __half* v_scales, __half* v_zeros,
    int batch_size, int num_heads, int position,
    int head_dim, int max_seq_len, int group_size
) {
    append_kv_int4_impl<float>(
        new_k, new_v, k_cache, v_cache,
        k_scales, k_zeros, v_scales, v_zeros,
        batch_size, num_heads, position, head_dim, max_seq_len, group_size
    );
}

extern "C" __global__ void append_kv_int4_fp16(
    const __half* new_k, const __half* new_v,
    uint8_t* k_cache, uint8_t* v_cache,
    __half* k_scales, __half* k_zeros,
    __half* v_scales, __half* v_zeros,
    int batch_size, int num_heads, int position,
    int head_dim, int max_seq_len, int group_size
) {
    append_kv_int4_impl<__half>(
        new_k, new_v, k_cache, v_cache,
        k_scales, k_zeros, v_scales, v_zeros,
        batch_size, num_heads, position, head_dim, max_seq_len, group_size
    );
}

extern "C" __global__ void append_kv_int4_bf16(
    const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
    uint8_t* k_cache, uint8_t* v_cache,
    __half* k_scales, __half* k_zeros,
    __half* v_scales, __half* v_zeros,
    int batch_size, int num_heads, int position,
    int head_dim, int max_seq_len, int group_size
) {
    append_kv_int4_impl<__nv_bfloat16>(
        new_k, new_v, k_cache, v_cache,
        k_scales, k_zeros, v_scales, v_zeros,
        batch_size, num_heads, position, head_dim, max_seq_len, group_size
    );
}
