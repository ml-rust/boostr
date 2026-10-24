// VarLen (packed) flash-attention backward — half-precision path (F16, BF16).
//
// `varlen_attention_bwd.cu` holds the FP32 path. This file compiles as its
// own module (VARLEN_ATTENTION_BWD_HALF_MODULE). One template on the element
// type T serves both 2-byte dtypes. Entry names end in `_fp16` for `__half`
// and `_bf16` for `__nv_bfloat16`. No native bf16 arithmetic runs here, so
// the module builds at sm_75.
//
// Causal convention: ABSOLUTE (bottom-right) alignment, per sequence.
// Within sequence s, seq_len_q = cu_seqlens_q[s+1] - cu_seqlens_q[s] and
// seq_len_k = cu_seqlens_k[s+1] - cu_seqlens_k[s], so the query rows of that
// sequence are the LAST seq_len_q positions of its seq_len_k keys and local
// query row r sits at absolute position key_offset + r, with a PER-SEQUENCE
// key_offset = seq_len_k - seq_len_q. Key ki is masked when ki > key_offset + r.
// A full prefill (seq_len_q == seq_len_k) gives key_offset == 0. Same
// convention as `ops/impl_generic/attention/flash_standard.rs::build_attention_mask`
// and `kernels/attention/flash_v2.cu`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// VarLen Attention Backward - half-precision inputs, full GQA
// ============================================================================
//
// Every thread computes in fp32. Q, K, V, O and dO are T. dQ is stored once
// as T. dK and dV are FLOAT buffers: the launcher allocates them as zeroed
// F32 scratch and casts them to T's dtype after the launch. No partial sum
// is rounded to T before the full reduction ends.
//
// Two-phase tile loop. Phase 1 gives each thread one Q token row of the tile
// and accumulates dQ locally. It needs no atomics: one block owns each Q row.
// Phase 2 transposes ownership. Each thread takes one K row of the tile,
// sweeps the tile's Q rows into fp32 registers, and emits ONE float atomicAdd
// per (k_row, head_dim element) for each of dK and dV. Atomics stay: under
// GQA several Q heads scatter into the same KV head, and each Q block of the
// sequence adds to the same K rows.

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void varlen_flash_attention_bwd_half_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ O,
    const float* __restrict__ L,
    const T* __restrict__ grad_O,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    T* __restrict__ grad_Q,
    float* __restrict__ grad_K,   // F32 scratch, zeroed by the launcher
    float* __restrict__ grad_V,   // F32 scratch, zeroed by the launcher
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int causal
) {
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    // Byte-typed so the `__half` and `__nv_bfloat16` instantiations share one
    // extern declaration.
    extern __shared__ __align__(16) unsigned char varlen_bwd_half_smem[];
    T* smem = reinterpret_cast<T*>(varlen_bwd_half_smem);

    T* Q_smem_flat  = smem;
    T* K_smem_flat  = smem + BLOCK_M * HEAD_STRIDE;
    T* V_smem_flat  = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;
    T* dO_smem_flat = smem + BLOCK_M * HEAD_STRIDE + 2 * BLOCK_N * HEAD_STRIDE;

    #define Q_smem(i, j)  Q_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define K_smem(i, j)  K_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define V_smem(i, j)  V_smem_flat[(i)  * HEAD_STRIDE + (j)]
    #define dO_smem(i, j) dO_smem_flat[(i) * HEAD_STRIDE + (j)]
    #define SMEM_F(base, i, j) load_dtype<T>(base, (i) * HEAD_STRIDE + (j))

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
    // Absolute (bottom-right) causal alignment, per sequence — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

    const int q_start = q_block_in_batch * BLOCK_M;
    const int q_end   = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    if (q_start >= seq_len_q) return;

    const T* Q_head  = Q      + head_idx    * HEAD_DIM;
    const T* K_head  = K      + kv_head_idx * HEAD_DIM;
    const T* V_head  = V      + kv_head_idx * HEAD_DIM;
    const T* O_head  = O      + head_idx    * HEAD_DIM;
    const T* dO_head = grad_O + head_idx    * HEAD_DIM;
    T*       dQ_head = grad_Q + head_idx    * HEAD_DIM;
    float*   dK_head = grad_K + kv_head_idx * HEAD_DIM;
    float*   dV_head = grad_V + kv_head_idx * HEAD_DIM;

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

    // D = rowsum(grad_O * O) and the forward logsumexp, staged in STATIC shared
    // memory rather than per-thread registers: the dK/dV phase below transposes
    // ownership to K rows, and a K-row-owning thread needs both values for
    // every Q row of the tile. Adds 2 * BLOCK_M * sizeof(float) bytes on top of
    // the dynamic allocation sized by `varlen_attention_block_config.rs`.
    __shared__ float D_smem[BLOCK_M];
    __shared__ float lse_smem[BLOCK_M];
    for (int row = tid; row < q_tile_size; row += blockDim.x) {
        const int global_q_pos = seq_start_q + q_start + row;
        float d_acc = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            d_acc += load_dtype<T>(O_head, global_q_pos * num_heads * HEAD_DIM + d) *
                     SMEM_F(dO_smem_flat, row, d);
        }
        D_smem[row] = d_acc;
        lse_smem[row] = L[global_q_pos * num_heads + head_idx];
    }
    __syncthreads();

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

        // grad_Q phase: this thread owns Q row `q_row` and sweeps the K tile.
        if (is_valid_thread) {
            const float lse = lse_smem[q_row];
            const float D   = D_smem[q_row];

            for (int j = 0; j < k_tile_size; ++j) {
                if (causal && (key_offset + q_start + q_row) < (k_start + j)) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += SMEM_F(Q_smem_flat, q_row, d) * SMEM_F(K_smem_flat, j, d);
                }
                score *= scale;

                const float prob = __expf(score - lse);

                float grad_prob = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    grad_prob += SMEM_F(V_smem_flat, j, d) * SMEM_F(dO_smem_flat, q_row, d);
                }

                const float grad_score = prob * (grad_prob - D);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += scale * grad_score * SMEM_F(K_smem_flat, j, d);
                }
            }
        }

        // grad_K/grad_V phase: ownership transposes. This thread owns K row
        // `k_row` of the tile and sweeps the tile's Q rows into fp32
        // registers. It emits ONE float atomicAdd per (k_row, head_dim
        // element) for each of dK and dV. The scores are recomputed here.
        // Staging P and dS as BLOCK_M x BLOCK_N floats breaks the
        // shared-memory budget that the `_small` tiles respect. Recomputing
        // costs only the score dot-product flops.
        for (int k_row = tid; k_row < k_tile_size; k_row += blockDim.x) {
            const int k_idx = k_start + k_row;

            float dK_local[HEAD_DIM];
            float dV_local[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dK_local[d] = 0.0f;
                dV_local[d] = 0.0f;
            }

            for (int qr = 0; qr < q_tile_size; ++qr) {
                if (causal && (key_offset + q_start + qr) < k_idx) continue;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += SMEM_F(Q_smem_flat, qr, d) * SMEM_F(K_smem_flat, k_row, d);
                }
                score *= scale;

                const float prob = __expf(score - lse_smem[qr]);

                float grad_prob = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    grad_prob += SMEM_F(V_smem_flat, k_row, d) * SMEM_F(dO_smem_flat, qr, d);
                }

                const float grad_score = prob * (grad_prob - D_smem[qr]);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dK_local[d] += scale * grad_score * SMEM_F(Q_smem_flat, qr, d);
                    dV_local[d] += prob * SMEM_F(dO_smem_flat, qr, d);
                }
            }

            const int global_k_pos = seq_start_k + k_idx;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomicAdd(&dK_head[global_k_pos * num_kv_heads * HEAD_DIM + d], dK_local[d]);
                atomicAdd(&dV_head[global_k_pos * num_kv_heads * HEAD_DIM + d], dV_local[d]);
            }
        }
        __syncthreads();
    }

    if (is_valid_thread) {
        const int global_q_pos = seq_start_q + q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            store_dtype<T>(dQ_head, global_q_pos * num_heads * HEAD_DIM + d, dQ_local[d]);
        }
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
    #undef dO_smem
    #undef SMEM_F
}

// ============================================================================
// Kernel Entry Points
//
// VARLEN_BWD_HALF_ENTRY(T, SUFFIX, HD, BM, BN, TILE) emits
// `varlen_flash_attention_bwd_<HD>_<SUFFIX><TILE>` for one tile config. The
// tiles match the forward entries in varlen_attention_fwd_half.cu, which
// lists the smem byte counts. head_dim=64 needs no small variant: its
// backward needs 49920 B dynamic plus 1024 B of static D_smem / lse_smem at
// the large tile, under a 99KB opt-in device.
// ============================================================================

#define VARLEN_BWD_HALF_ENTRY(T, SUFFIX, HD, BM, BN, TILE)                     \
    extern "C" __global__ void varlen_flash_attention_bwd_##HD##_##SUFFIX##TILE( \
        const T* Q, const T* K, const T* V,                                    \
        const T* O, const float* L, const T* grad_O,                           \
        const int* cu_seqlens_q, const int* cu_seqlens_k,                      \
        T* grad_Q, float* grad_K, float* grad_V,                               \
        int batch_size, int num_heads, int num_kv_heads,                       \
        int max_seqlen_q, int max_seqlen_k, float scale, int causal            \
    ) {                                                                        \
        varlen_flash_attention_bwd_half_impl<T, HD, BM, BN>(                   \
            Q, K, V, O, L, grad_O,                                             \
            cu_seqlens_q, cu_seqlens_k,                                        \
            grad_Q, grad_K, grad_V,                                            \
            batch_size, num_heads, num_kv_heads,                               \
            max_seqlen_q, max_seqlen_k, scale, causal                          \
        );                                                                     \
    }

#define VARLEN_BWD_HALF_DTYPE(T, SUFFIX)                                       \
    VARLEN_BWD_HALF_ENTRY(T, SUFFIX, 64, 128, 64, )                            \
    VARLEN_BWD_HALF_ENTRY(T, SUFFIX, 128, 128, 64, )                           \
    VARLEN_BWD_HALF_ENTRY(T, SUFFIX, 128, 32, 32, _small)                      \
    VARLEN_BWD_HALF_ENTRY(T, SUFFIX, 256, 32, 32, )

VARLEN_BWD_HALF_DTYPE(__half, fp16)
VARLEN_BWD_HALF_DTYPE(__nv_bfloat16, bf16)

#undef VARLEN_BWD_HALF_DTYPE
#undef VARLEN_BWD_HALF_ENTRY
