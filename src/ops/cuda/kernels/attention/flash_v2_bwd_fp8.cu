// Flash Attention v2 Backward - FP8 Kernels (separate translation unit)
//
// Split out of flash_v2_bwd.cu: these kernels are guarded by
// `#if __CUDA_ARCH__ >= 800`, but flash_v2_bwd.cu is compiled at sm_75 because
// lines outside this block are the general Turing-capable flash kernels.
// Compiled at sm_75 the guard erased every FP8 symbol from the PTX, so
// `flash_attention_bwd_*_fp8` was never found at runtime on any GPU.
// As its own translation unit this file is compiled at sm_80 (see build.rs)
// and the guard now documents a real requirement instead of erasing the file.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// FP8 Backward Kernels - For Ampere/Hopper GPUs (FP8 I/O, FP32 accumulation)
// ============================================================================

#if __CUDA_ARCH__ >= 800  // Ampere and newer

// Preprocessing for FP8
template<int HEAD_DIM>
__device__ void flash_attention_preprocess_bwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ dO,
    const boostr_fp8_e4m3* __restrict__ O,
    float* __restrict__ D,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float dO_scale,
    const float O_scale
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || q_pos >= seq_len) return;

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const boostr_fp8_e4m3* dO_row = dO + offset + q_pos * HEAD_DIM;
    const boostr_fp8_e4m3* O_row = O + offset + q_pos * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float dO_val = fp8_e4m3_to_f32(dO_row[d], dO_scale);
        float O_val = fp8_e4m3_to_f32(O_row[d], O_scale);
        sum += dO_val * O_val;
    }

    const int d_offset = (batch_idx * num_heads + head_idx) * seq_len;
    D[d_offset + q_pos] = sum;
}

// Main backward kernel for FP8
// `dQ` is an FP32 accumulator holding the DEQUANTIZED gradient, NOT an FP8
// buffer: K/V blocks atomicAdd into the same element and CUDA has no 1-byte
// float atomic. The launcher applies `dQ_scale` and quantizes afterwards, so
// `dQ_scale` is unused here.
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void flash_attention_bwd_fp8_impl(
    const boostr_fp8_e4m3* __restrict__ Q,
    const boostr_fp8_e4m3* __restrict__ K,
    const boostr_fp8_e4m3* __restrict__ V,
    const boostr_fp8_e4m3* __restrict__ O,
    const boostr_fp8_e4m3* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    boostr_fp8_e4m3* __restrict__ dK,
    boostr_fp8_e4m3* __restrict__ dV,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float Q_scale,
    const float K_scale,
    const float V_scale,
    const float dO_scale,
    const float dQ_scale,
    const float dK_scale,
    const float dV_scale
) {
    extern __shared__ boostr_fp8_e4m3 smem_fp8[];

    boostr_fp8_e4m3* K_smem_flat = smem_fp8;
    boostr_fp8_e4m3* V_smem_flat = smem_fp8 + BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* Q_smem_flat = smem_fp8 + 2 * BLOCK_N * HEAD_DIM;
    boostr_fp8_e4m3* dO_smem_flat = smem_fp8 + 2 * BLOCK_N * HEAD_DIM + BLOCK_M * HEAD_DIM;

    #define K_smem(i, j) K_smem_flat[(i) * HEAD_DIM + (j)]
    #define V_smem(i, j) V_smem_flat[(i) * HEAD_DIM + (j)]
    #define Q_smem(i, j) Q_smem_flat[(i) * HEAD_DIM + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_DIM + (j)]

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int k_block = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int k_start = k_block * BLOCK_N;
    const int k_end = min(k_start + BLOCK_N, seq_len_k);
    const int k_tile_size = k_end - k_start;

    if (batch_idx >= batch_size || k_start >= seq_len_k) return;

    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_idx * num_heads + head_idx) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_idx * num_heads + head_idx) * seq_len_q;

    const boostr_fp8_e4m3* Q_base = Q + head_offset;
    const boostr_fp8_e4m3* K_base = K + kv_head_offset;
    const boostr_fp8_e4m3* V_base = V + kv_head_offset;
    const boostr_fp8_e4m3* dO_base = dO + head_offset;
    const float* LSE_base = LSE + lse_offset;
    const float* D_base = D + lse_offset;
    float* dQ_base = dQ + head_offset;
    boostr_fp8_e4m3* dK_base = dK + kv_head_offset;
    boostr_fp8_e4m3* dV_base = dV + kv_head_offset;

    for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        K_smem(row, col) = K_base[(k_start + row) * HEAD_DIM + col];
        V_smem(row, col) = V_base[(k_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

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
            Q_smem(row, col) = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_smem(row, col) = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

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

                if (causal && q_pos + max(0, seq_len_k - seq_len_q) < k_pos) continue;

                // Dequantize FP8 → FP32 for computation
                float qk_score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), Q_scale);
                    float k_val = fp8_e4m3_to_f32(K_smem(k_col, d), K_scale);
                    qk_score += q_val * k_val;
                }
                qk_score *= scale;

                const float p_val = __expf(qk_score - lse_val);

                float dp_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float dO_val = fp8_e4m3_to_f32(dO_smem(q_row, d), dO_scale);
                    float v_val = fp8_e4m3_to_f32(V_smem(k_col, d), V_scale);
                    dp_val += dO_val * v_val;
                }

                const float ds_val = p_val * (dp_val - d_val) * scale;

                if ((q_row % (int)blockDim.x) == tid) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        float k_val = fp8_e4m3_to_f32(K_smem(k_col, d), K_scale);
                        dQ_local[d] += ds_val * k_val;
                    }
                }

                if (k_col == k_row && k_row < k_tile_size) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        float dO_val = fp8_e4m3_to_f32(dO_smem(q_row, d), dO_scale);
                        float q_val = fp8_e4m3_to_f32(Q_smem(q_row, d), Q_scale);
                        dV_local[d] += p_val * dO_val;
                        dK_local[d] += ds_val * q_val;
                    }
                }
            }

            if ((q_row % (int)blockDim.x) == tid) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    atomicAdd(&dQ_base[q_pos * HEAD_DIM + d], dQ_local[d]);
                }
            }
        }

        __syncthreads();
    }

    // Quantize FP32 → FP8 for output
    if (k_row < k_tile_size && (k_start + k_row) < seq_len_k) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dK_base[(k_start + k_row) * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dK_local[d], dK_scale));
            dV_base[(k_start + k_row) * HEAD_DIM + d] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(dV_local[d], dV_scale));
        }
    }

    #undef K_smem
    #undef V_smem
    #undef Q_smem
    #undef dO_smem
}

// FP8 kernel instantiations
extern "C" __global__ void flash_attention_preprocess_bwd_64_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) {
    flash_attention_preprocess_bwd_fp8_impl<64>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale);
}

extern "C" __global__ void flash_attention_bwd_64_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<64, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_128_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) {
    flash_attention_preprocess_bwd_fp8_impl<128>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale);
}

extern "C" __global__ void flash_attention_bwd_128_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<128, 128, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_32_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<32>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_32_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<32, 128, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_96_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<96>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_96_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<96, 64, 128>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_192_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<192>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_192_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<192, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

extern "C" __global__ void flash_attention_preprocess_bwd_256_fp8(
    const boostr_fp8_e4m3* dO, const boostr_fp8_e4m3* O, float* D,
    const int batch_size, const int num_heads, const int seq_len,
    const float dO_scale, const float O_scale
) { flash_attention_preprocess_bwd_fp8_impl<256>(dO, O, D, batch_size, num_heads, seq_len, dO_scale, O_scale); }

extern "C" __global__ void flash_attention_bwd_256_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<256, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// ============================================================================
// Small-shared-memory Backward Instantiations - FP8
// Same `flash_attention_bwd_fp8_impl` template with smaller BLOCK_M/BLOCK_N.
// Selected by `bwd_block_config` in src/ops/cuda/attention/flash_utils.rs —
// keep the block sizes in sync.
// ============================================================================

// head_dim=32, BLOCK_M=64, BLOCK_N=64
extern "C" __global__ void flash_attention_bwd_32_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<32, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// head_dim=64, BLOCK_M=64, BLOCK_N=64
extern "C" __global__ void flash_attention_bwd_64_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<64, 64, 64>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// head_dim=96, BLOCK_M=32, BLOCK_N=32
extern "C" __global__ void flash_attention_bwd_96_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<96, 32, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// head_dim=128, BLOCK_M=32, BLOCK_N=32
extern "C" __global__ void flash_attention_bwd_128_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<128, 32, 32>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// head_dim=192, BLOCK_M=16, BLOCK_N=16
extern "C" __global__ void flash_attention_bwd_192_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<192, 16, 16>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// head_dim=256, BLOCK_M=16, BLOCK_N=16
extern "C" __global__ void flash_attention_bwd_256_sm_fp8(
    const boostr_fp8_e4m3* Q, const boostr_fp8_e4m3* K, const boostr_fp8_e4m3* V,
    const boostr_fp8_e4m3* O, const boostr_fp8_e4m3* dO, const float* LSE, const float* D,
    float* dQ, boostr_fp8_e4m3* dK, boostr_fp8_e4m3* dV,
    const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_k,
    const float scale, const int causal,
    const float Q_scale, const float K_scale, const float V_scale, const float dO_scale,
    const float dQ_scale, const float dK_scale, const float dV_scale
) {
    flash_attention_bwd_fp8_impl<256, 16, 16>(
        Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        batch_size, num_heads, seq_len_q, seq_len_k, scale, causal,
        Q_scale, K_scale, V_scale, dO_scale, dQ_scale, dK_scale, dV_scale
    );
}

// ============================================================================
// Diagnostic probe: E4M3 round trip AS COMPILED IN THIS TRANSLATION UNIT
// ============================================================================
//
// Exercises the same `f32_to_fp8_e4m3_raw` / `fp8_e4m3_to_f32` pair the
// backward kernel above uses, at the same arch (sm_80) and under the same
// `--use_fast_math` flag. Tests compare its two outputs against numr's cast:
// `raw` isolates the ENCODER, `dec` isolates the DECODER. A disagreement names
// which converter is wrong; agreement rules both out and moves the search
// upstream into the gradient itself.
//
// `scale` is 1.0f so the probe measures rounding only, not the scale multiply.
extern "C" __global__ void fp8_e4m3_roundtrip_probe(
    const float* __restrict__ in,
    boostr_fp8_e4m3* __restrict__ raw,
    float* __restrict__ dec,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint8_t byte = f32_to_fp8_e4m3_raw(in[idx], 1.0f);
    raw[idx] = boostr_fp8_e4m3(byte);
    dec[idx] = fp8_e4m3_to_f32(byte, 1.0f);
}

// Same probe for E5M2, which shares the converter family and therefore shares
// every rounding, carry, and subnormal path with E4M3.
extern "C" __global__ void fp8_e5m2_roundtrip_probe(
    const float* __restrict__ in,
    boostr_fp8_e5m2* __restrict__ raw,
    float* __restrict__ dec,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint8_t byte = f32_to_fp8_e5m2_raw(in[idx], 1.0f);
    raw[idx] = boostr_fp8_e5m2(byte);
    dec[idx] = fp8_e5m2_to_f32(byte, 1.0f);
}

#endif  // __CUDA_ARCH__ >= 800
