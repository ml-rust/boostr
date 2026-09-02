// VarLen (Ragged) Flash Attention Backward Pass
// Based on Flash Attention v2 backward with cumulative sequence length indexing
//
// Key features:
// 1. Packed sequences with cu_seqlens indexing
// 2. Eliminates padding overhead in gradient computation
// 3. Full GQA support: num_kv_heads <= num_heads; dK/dV scatter via atomicAdd
// 4. FP16 uses correct atomicAddHalf (no reinterpret_cast corruption)
// 5. head_dim 64 / 128 / 256 supported
//
// Backward pass computes:
//   grad_Q = scale * grad_scores @ K
//   grad_K = scale * grad_scores^T @ Q   (accumulated via atomicAdd, GQA scatter)
//   grad_V = probs^T @ grad_output       (accumulated via atomicAdd, GQA scatter)
//
// Where grad_scores = probs * (grad_probs - D)  and  D = rowsum(grad_O * O)
//
// Causal convention: ABSOLUTE (bottom-right) alignment, per sequence.
// Within sequence s, seq_len_q = cu_seqlens_q[s+1] - cu_seqlens_q[s] and
// seq_len_k = cu_seqlens_k[s+1] - cu_seqlens_k[s], so the query rows of that
// sequence are the LAST seq_len_q positions of its seq_len_k keys and local
// query row r sits at absolute position key_offset + r, with a PER-SEQUENCE
// key_offset = seq_len_k - seq_len_q. Key ki is masked when ki > key_offset + r.
// A full prefill (seq_len_q == seq_len_k) gives key_offset == 0, leaving the
// rule identical to the previous top-left form. Same convention as
// `ops/impl_generic/attention/flash_standard.rs::build_attention_mask` and
// `kernels/attention/flash_v2.cu`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// VarLen Attention Backward - FP32, full GQA
// ============================================================================
//
// Two-phase tile loop. Phase 1 gives each thread one Q token row of the tile
// and accumulates dQ locally (no atomics: each Q row is owned by exactly one
// block). Phase 2 transposes ownership — each thread takes one K row of the
// tile, sweeps the tile's Q rows into fp32 registers, and emits ONE atomicAdd
// per (k_row, head_dim element) for each of dK and dV. Atomics cannot be
// dropped: under GQA several Q heads scatter into the same KV head, and each Q
// block of the sequence contributes to the same K rows.

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void varlen_flash_attention_bwd_fp32_impl(
    const float* __restrict__ Q,           // [total_tokens_q, num_heads, head_dim]
    const float* __restrict__ K,           // [total_tokens_k, num_kv_heads, head_dim]
    const float* __restrict__ V,           // [total_tokens_k, num_kv_heads, head_dim]
    const float* __restrict__ O,           // [total_tokens_q, num_heads, head_dim]
    const float* __restrict__ L,           // [total_tokens_q, num_heads] (logsumexp from forward)
    const float* __restrict__ grad_O,      // [total_tokens_q, num_heads, head_dim]
    const int* __restrict__ cu_seqlens_q,  // [batch_size + 1]
    const int* __restrict__ cu_seqlens_k,  // [batch_size + 1]
    float* __restrict__ grad_Q,            // [total_tokens_q, num_heads, head_dim]
    float* __restrict__ grad_K,            // [total_tokens_k, num_kv_heads, head_dim]
    float* __restrict__ grad_V,            // [total_tokens_k, num_kv_heads, head_dim]
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,                // GQA: <= num_heads; MHA: == num_heads
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int causal
) {
    // +1 smem padding (mirrors fwd HEAD_STRIDE)
    constexpr int HEAD_STRIDE = HEAD_DIM + 1;

    extern __shared__ float smem[];

    float* Q_smem_flat  = smem;
    float* K_smem_flat  = smem + BLOCK_M * HEAD_STRIDE;
    float* V_smem_flat  = smem + BLOCK_M * HEAD_STRIDE + BLOCK_N * HEAD_STRIDE;
    float* dO_smem_flat = smem + BLOCK_M * HEAD_STRIDE + 2 * BLOCK_N * HEAD_STRIDE;
    // smem size = (2*BLOCK_M + 2*BLOCK_N) * HEAD_STRIDE * sizeof(float)

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

    // GQA head mapping: kv_head_idx = head_idx * num_kv_heads / num_heads
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;

    // Get sequence boundaries
    const int seq_start_q = cu_seqlens_q[batch_idx];
    const int seq_end_q   = cu_seqlens_q[batch_idx + 1];
    const int seq_len_q   = seq_end_q - seq_start_q;

    const int seq_start_k = cu_seqlens_k[batch_idx];
    const int seq_end_k   = cu_seqlens_k[batch_idx + 1];
    const int seq_len_k   = seq_end_k - seq_start_k;
    // Absolute (bottom-right) causal alignment, per sequence — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

    // Local Q block position
    const int q_start = q_block_in_batch * BLOCK_M;
    const int q_end   = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;

    if (q_start >= seq_len_q) return;

    // Base pointers — Q/dQ/O/dO use num_heads stride; K/V/dK/dV use num_kv_heads
    const float* Q_head  = Q      + head_idx    * HEAD_DIM;
    const float* K_head  = K      + kv_head_idx * HEAD_DIM;
    const float* V_head  = V      + kv_head_idx * HEAD_DIM;
    const float* O_head  = O      + head_idx    * HEAD_DIM;
    const float* dO_head = grad_O + head_idx    * HEAD_DIM;
    float*       dQ_head = grad_Q + head_idx    * HEAD_DIM;
    float*       dK_head = grad_K + kv_head_idx * HEAD_DIM;
    float*       dV_head = grad_V + kv_head_idx * HEAD_DIM;

    // Load Q tile and grad_O tile into shared memory
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

    // Initialize grad_Q accumulator
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
            d_acc += O_head[global_q_pos * num_heads * HEAD_DIM + d] * dO_smem(row, d);
        }
        D_smem[row] = d_acc;
        lse_smem[row] = L[global_q_pos * num_heads + head_idx];
    }
    __syncthreads();

    // Iterate over K/V tiles
    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start     = k_block * BLOCK_N;
        const int k_end       = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles — K/V row stride uses num_kv_heads (GQA layout)
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

                // Recompute score and prob
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem(q_row, d) * K_smem(j, d);
                }
                score *= scale;
                const float prob = __expf(score - lse);

                // grad_prob = V[j] · dO[q_row]
                float grad_prob = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    grad_prob += V_smem(j, d) * dO_smem(q_row, d);
                }

                // Softmax backward: grad_score = prob * (grad_prob - D)
                const float grad_score = prob * (grad_prob - D);

                // Accumulate grad_Q locally
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += scale * grad_score * K_smem(j, d);
                }
            }
        }

        // grad_K/grad_V phase: ownership transposes — this thread owns K row
        // `k_row` of the tile and sweeps the tile's Q rows, summing into fp32
        // registers and emitting ONE atomic per (k_row, head_dim element)
        // instead of one per (q_row, k_row, head_dim element). The scores are
        // recomputed here rather than staged in shared memory: staging P and
        // dS as BLOCK_M x BLOCK_N floats would break the shared-memory budget
        // the `_small` tiles exist to respect, while recomputing costs only
        // the score dot-product flops and no extra shared memory.
        // Atomics remain required: under GQA several Q heads scatter into the
        // same KV head, and each Q block of the sequence adds its own share.
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
                    score += Q_smem(qr, d) * K_smem(k_row, d);
                }
                score *= scale;
                const float prob = __expf(score - lse_smem[qr]);

                float grad_prob = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    grad_prob += V_smem(k_row, d) * dO_smem(qr, d);
                }

                const float grad_score = prob * (grad_prob - D_smem[qr]);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dK_local[d] += scale * grad_score * Q_smem(qr, d);
                    dV_local[d] += prob * dO_smem(qr, d);
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

    // Write grad_Q (no atomics needed: each Q row is owned by exactly one block)
    if (is_valid_thread) {
        const int global_q_pos = seq_start_q + q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dQ_head[global_q_pos * num_heads * HEAD_DIM + d] = dQ_local[d];
        }
    }

    #undef Q_smem
    #undef K_smem
    #undef V_smem
    #undef dO_smem
}

// ============================================================================
// FP32 Kernel Entry Points
// ============================================================================

extern "C" __global__ void varlen_flash_attention_bwd_64_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp32_impl<64, 128, 64>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_bwd_128_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp32_impl<128, 128, 64>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

// ============================================================================
// HEAD_DIM=64/128, small tile fallback (BLOCK_M=32/16) — see the matching
// forward small-tile comment in varlen_attention.cu for the smem byte counts.
// ============================================================================

extern "C" __global__ void varlen_flash_attention_bwd_64_fp32_small(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp32_impl<64, 32, 32>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

extern "C" __global__ void varlen_flash_attention_bwd_128_fp32_small(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp32_impl<128, 16, 16>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}

// head_dim=256: use the same small tiles as the fwd 256 fp32 kernel
extern "C" __global__ void varlen_flash_attention_bwd_256_fp32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* grad_O,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    float* grad_Q, float* grad_K, float* grad_V,
    int batch_size, int num_heads, int num_kv_heads,
    int max_seqlen_q, int max_seqlen_k, float scale, int causal
) {
    varlen_flash_attention_bwd_fp32_impl<256, 16, 16>(
        Q, K, V, O, L, grad_O,
        cu_seqlens_q, cu_seqlens_k,
        grad_Q, grad_K, grad_V,
        batch_size, num_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k, scale, causal
    );
}
