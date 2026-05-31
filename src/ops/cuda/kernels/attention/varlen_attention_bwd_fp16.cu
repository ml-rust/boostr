// VarLen (packed) flash-attention backward — FP16 path.
//
// Split from varlen_attention_bwd.cu (which now holds the FP32 path) to keep
// each kernel translation unit within the file-size budget. Compiled as its own
// module (VARLEN_ATTENTION_BWD_FP16_MODULE); the FP16 kernel symbols are
// unchanged so the Rust dispatcher loads them by the same names.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "atomics.cuh"

// ============================================================================
// VarLen Attention Backward - FP16, full GQA, fixed atomics
// ============================================================================
//
// All computation is done in fp32 per-thread.  Only the final dQ write and
// the dK/dV atomic scatters involve half-precision conversion.
// atomicAddHalf from atomics.cuh is used for dK/dV (replaces the broken
// `atomicAdd(reinterpret_cast<float*>(__half_ptr), val)` from the old code).

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void varlen_flash_attention_bwd_fp16_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const float*  __restrict__ L,
    const __half* __restrict__ grad_O,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ cu_seqlens_k,
    __half* __restrict__ grad_Q,
    __half* __restrict__ grad_K,
    __half* __restrict__ grad_V,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int causal
) {
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    extern __shared__ __half smem_fp16[];

    __half* Q_smem_flat  = smem_fp16;
    __half* K_smem_flat  = smem_fp16 + BLOCK_M * HEAD_STRIDE;
    __half* V_smem_flat  = smem_fp16 + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;
    __half* dO_smem_flat = smem_fp16 + BLOCK_M * HEAD_STRIDE + 2 * BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j)  Q_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define K_smem(i, j)  K_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define V_smem(i, j)  V_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_STRIDE + (j)]

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x % num_heads;
    const int remaining = blockIdx.x / num_heads;
    const int num_q_blocks_per_batch = (max_seqlen_q + BLOCK_M - 1) / BLOCK_M;
    const int batch_idx = remaining / num_q_blocks_per_batch;
    const int q_block_in_batch = remaining % num_q_blocks_per_batch;

    if (batch_idx >= batch_size) return;

    // GQA head mapping
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;

    const int seq_start_q = cu_seqlens_q[batch_idx];
    const int seq_end_q   = cu_seqlens_q[batch_idx + 1];
    const int seq_len_q   = seq_end_q - seq_start_q;

    const int seq_start_k = cu_seqlens_k[batch_idx];
    const int seq_end_k   = cu_seqlens_k[batch_idx + 1];
    const int seq_len_k   = seq_end_k - seq_start_k;

    const int q_start = q_block_in_batch * BLOCK_M;
    const int q_end   = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    if (q_start >= seq_len_q) return;

    const __half* Q_head  = Q      + head_idx    * HEAD_DIM;
    const __half* K_head  = K      + kv_head_idx * HEAD_DIM;
    const __half* V_head  = V      + kv_head_idx * HEAD_DIM;
    const __half* O_head  = O      + head_idx    * HEAD_DIM;
    const __half* dO_head = grad_O + head_idx    * HEAD_DIM;
    __half*       dQ_head = grad_Q + head_idx    * HEAD_DIM;
    __half*       dK_head = grad_K + kv_head_idx * HEAD_DIM;
    __half*       dV_head = grad_V + kv_head_idx * HEAD_DIM;

    // Load Q and grad_O tiles
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int global_q_pos = seq_start_q + q_start + row;
        Q_smem(row, col)  = Q_head[global_q_pos  * num_heads * HEAD_DIM + col];
        dO_smem(row, col) = dO_head[global_q_pos * num_heads * HEAD_DIM + col];
    }
    __syncthreads();

    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // Accumulate dQ in fp32 for precision
    float dQ_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dQ_local[d] = 0.0f;
    }

    float lse = 0.0f;
    float D   = 0.0f;
    if (is_valid_thread) {
        const int global_q_pos = seq_start_q + q_start + q_row;
        lse = L[global_q_pos * num_heads + head_idx];

        for (int d = 0; d < HEAD_DIM; ++d) {
            float o_val  = __half2float(O_head[global_q_pos * num_heads * HEAD_DIM + d]);
            float do_val = __half2float(dO_smem(q_row, d));
            D += o_val * do_val;
        }
    }

    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start     = k_block * BLOCK_N;
        const int k_end       = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // K/V row stride uses num_kv_heads (GQA layout)
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int global_k_pos = seq_start_k + k_start + row;
            K_smem(row, col) = K_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
            V_smem(row, col) = V_head[global_k_pos * num_kv_heads * HEAD_DIM + col];
        }
        __syncthreads();

        if (is_valid_thread) {
            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(Q_smem(q_row, d)) * __half2float(K_smem(j, d));
                }
                score *= scale;

                const float prob = __expf(score - lse);

                float grad_prob = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    grad_prob += __half2float(V_smem(j, d)) * __half2float(dO_smem(q_row, d));
                }

                const float grad_score = prob * (grad_prob - D);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += scale * grad_score * __half2float(K_smem(j, d));
                }

                // Atomic scatter dK and dV into the kv_head's buffer.
                // Accumulate per-thread in fp32 first, then convert and atomicAddHalf.
                // This avoids the broken reinterpret_cast-to-float* pattern.
                const int global_k_pos = seq_start_k + k_start + j;
#if __CUDA_ARCH__ >= 700
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float dk_val = scale * grad_score * __half2float(Q_smem(q_row, d));
                    atomicAddHalf(
                        &dK_head[global_k_pos * num_kv_heads * HEAD_DIM + d],
                        __float2half(dk_val)
                    );

                    float dv_val = prob * __half2float(dO_smem(q_row, d));
                    atomicAddHalf(
                        &dV_head[global_k_pos * num_kv_heads * HEAD_DIM + d],
                        __float2half(dv_val)
                    );
                }
#endif
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const int global_q_pos = seq_start_q + q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dQ_head[global_q_pos * num_heads * HEAD_DIM + d] = __float2half(dQ_local[d]);
        }
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
    #undef dO_smem
}

// ============================================================================
// FP16 Kernel Entry Points
// ============================================================================

extern "C" __global__ void varlen_flash_attention_bwd_64_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const float* L, const __half* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* grad_Q, __half* grad_K, __half* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp16_impl<64, 128, 64>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_bwd_128_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const float* L, const __half* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* grad_Q, __half* grad_K, __half* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp16_impl<128, 128, 64>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

// head_dim=256: use the same small tiles as the fwd 256 fp16 kernel
extern "C" __global__ void varlen_flash_attention_bwd_256_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const float* L, const __half* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    __half* grad_Q, __half* grad_K, __half* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp16_impl<256, 32, 32>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}
