// Paged Attention Backward Pass - vLLM-style non-contiguous KV cache
// Based on Flash Attention v2 backward with paged KV cache support
//
// Computes gradients for Q, K, V given gradient of output dO
// Uses block table indirection for non-contiguous KV cache access
//
// Key difference from standard Flash Attention backward:
// - K and V are stored in non-contiguous blocks
// - Block table maps logical positions to physical block addresses
// - Gradients dK and dV are accumulated using atomics for shared blocks
//
// dK/dV ownership: the tile loop runs two phases. The dQ phase gives each
// thread one Q row; the dK/dV phase transposes that and gives each thread one
// K row of the tile, summing over the tile's Q rows into FP32 registers and
// emitting ONE atomic per (k_row, head_dim element). The scores are recomputed
// in the second phase rather than staged in shared memory, so the shared-memory
// layout is unchanged — the same trade `mqa_gqa_bwd.cu` makes. Atomics cannot
// be dropped entirely: under causal masking and GQA a paged K/V slot receives
// contributions from several query blocks and several query heads.
//
// Causal convention: ABSOLUTE (bottom-right) alignment. The block table indexes
// keys by their absolute position in the sequence, and the seq_len_q query rows
// are the LAST positions of that seq_len_k context, so query row r sits at
// absolute position key_offset + r with key_offset = seq_len_k - seq_len_q.
// Key j is masked when j > key_offset + r. A full prefill
// (seq_len_q == seq_len_k) gives key_offset == 0, leaving the rule identical to
// the previous top-left form. Same convention as
// `ops/impl_generic/attention/flash_standard.rs::build_attention_mask` and
// `kernels/attention/flash_v2.cu`.
//
// One implementation and one entry-point signature, all dtypes:
// `paged_attention_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>` serves F32, F16 and
// BF16. There is no FP8 paged backward kernel.
//
// SHARED MEMORY — the element type is the TENSOR DTYPE, not float. Q/K/V/dO/O
// stage verbatim as `T` and convert to float only when read back through
// `load_dtype`, so the dynamic allocation is
// (3*BLOCK_M + 2*BLOCK_N) * HEAD_DIM * sizeof(T), with NO bank-conflict
// padding (unlike flash/varlen, paged does not pad HEAD_DIM). That matches
// `bwd_smem_size` in `paged_attention_bwd_block_config.rs`, which multiplies by
// `dtype.size_in_bytes()`. Staging as float instead would make the F16/BF16
// kernels write twice their allocation. Two STATIC shared arrays, `D_smem` and
// `lse_smem`, add 2 * BLOCK_M floats on top and are NOT part of that dynamic
// figure.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "paged_attention.cuh"

// ============================================================================
// Half-precision atomics
//
// dK/dV scatter uses `atomic_add_dtype` from dtype_traits.cuh, which carries a
// real pre-Ampere CAS fallback for every dtype. This file kept private
// `atomicAddHalf`/`atomicAddBF16` copies only while that header's BF16 body sat
// behind `#if __CUDA_ARCH__ >= 800` with no `#else` and compiled to nothing at
// the sm_75 `build.rs` targets here. It does not any more, so there is one
// implementation.
// ============================================================================

// A template cannot redeclare `extern __shared__` per instantiation type, so
// the tiles are carved out of one raw byte array. The dynamic shared-memory
// base is the same address every dtype saw before.
extern __shared__ __align__(16) unsigned char paged_bwd_smem_raw[];

// ============================================================================
// Paged Flash Attention Backward - templated implementation
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M))
// Block: BLOCK_M threads.
//
// Layouts:
//   Q/O/dO/dQ: [batch, num_heads, seq_len_q, head_dim]
//   K/V/dK/dV blocks: [num_blocks, block_size, num_kv_heads, head_dim]
//   L: [batch, num_heads, seq_len_q] logsumexp from the forward pass
//   block_table: [batch, max_num_blocks]
// ============================================================================

template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__device__ void paged_attention_bwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K_blocks,
    const T* __restrict__ V_blocks,
    const T* __restrict__ O,
    const T* __restrict__ dO,
    const float* __restrict__ L,
    const int* __restrict__ block_table,
    T* __restrict__ dQ,
    T* __restrict__ dK_blocks,
    T* __restrict__ dV_blocks,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int max_num_blocks,
    const int block_size,
    const float scale,
    const int causal
) {
    // Shared memory element type is T — see the note in the file header.
    T* smem = reinterpret_cast<T*>(paged_bwd_smem_raw);

    T* Q_smem = smem;
    T* K_smem = smem + BLOCK_M * HEAD_DIM;
    T* V_smem = smem + BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM;
    T* dO_smem = smem + BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;
    T* O_smem = smem + 2 * BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    // GQA/MQA: each KV head serves (num_heads / num_kv_heads) query heads.
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Base pointers for this (batch, head)
    const int head_offset = batch_idx * num_heads * seq_len_q * HEAD_DIM
                           + head_idx * seq_len_q * HEAD_DIM;
    const int lse_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    const T* Q_base = Q + head_offset;
    const T* O_base = O + head_offset;
    const T* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    T* dQ_base = dQ + head_offset;

    // Q tile indices
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len_q);
    const int q_tile_size = q_end - q_start;
    // Absolute (bottom-right) causal alignment — see file header.
    const int key_offset = max(0, seq_len_k - seq_len_q);

    // Load Q, O, dO tiles into shared memory
    for (int i = tid; i < q_tile_size * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        Q_smem[row * HEAD_DIM + col] = Q_base[(q_start + row) * HEAD_DIM + col];
        O_smem[row * HEAD_DIM + col] = O_base[(q_start + row) * HEAD_DIM + col];
        dO_smem[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
    }
    __syncthreads();

    // Each thread owns one Q row for the dQ phase.
    const int q_row = tid;
    const bool is_valid_thread = (q_row < q_tile_size);

    // D = rowsum(dO * O) for softmax backward, plus the forward logsumexp.
    // Staged in shared memory because the dK/dV phase below transposes
    // ownership to K rows and needs both for every Q row of the tile.
    __shared__ float D_smem[BLOCK_M];
    __shared__ float lse_smem[BLOCK_M];
    for (int row = tid; row < q_tile_size; row += blockDim.x) {
        float d_acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            d_acc += load_dtype(dO_smem, row * HEAD_DIM + d) *
                     load_dtype(O_smem, row * HEAD_DIM + d);
        }
        D_smem[row] = d_acc;
        lse_smem[row] = L_base[q_start + row];
    }
    __syncthreads();

    // Per-thread dQ accumulator
    float dQ_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        dQ_local[d] = 0.0f;
    }

    // Iterate over K/V tiles
    const int num_k_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len_k);
        const int k_tile_size = k_end - k_start;

        // Load K and V tiles from paged blocks
        for (int i = tid; i < k_tile_size * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int token_idx = k_start + row;

            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, token_idx, block_size, num_kv_heads, kv_head_idx, HEAD_DIM
            );

            K_smem[row * HEAD_DIM + col] = K_blocks[kv_offset + col];
            V_smem[row * HEAD_DIM + col] = V_blocks[kv_offset + col];
        }
        __syncthreads();

        // dQ phase: this thread owns Q row `q_row` and sweeps the K tile.
        if (is_valid_thread) {
            const float lse_val = lse_smem[q_row];
            const float D_local = D_smem[q_row];

            for (int j = 0; j < k_tile_size; ++j) {
                const int k_idx = k_start + j;
                if (causal && (key_offset + q_start + q_row) < k_idx) continue;

                // Compute Q @ K^T
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += load_dtype(Q_smem, q_row * HEAD_DIM + d) *
                             load_dtype(K_smem, j * HEAD_DIM + d);
                }
                score *= scale;

                // Compute softmax(score) using stored logsumexp
                float p = __expf(score - lse_val);

                // Compute dP = dO @ V^T
                float dP = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += load_dtype(dO_smem, q_row * HEAD_DIM + d) *
                          load_dtype(V_smem, j * HEAD_DIM + d);
                }

                // Softmax backward: dS = P * (dP - D)
                float dS = p * (dP - D_local) * scale;

                // Accumulate dQ = dS @ K
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dQ_local[d] += dS * load_dtype(K_smem, j * HEAD_DIM + d);
                }
            }
        }

        // dK/dV phase: ownership transposes — this thread owns K row `k_row`
        // and sweeps the Q tile, accumulating in FP32 registers so a low
        // precision store is rounded once, at the single atomic per (k_row, d).
        // The scores are recomputed rather than staged in shared memory (same
        // trade as `mqa_gqa_bwd.cu`), so the shared-memory layout is unchanged.
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
                    score += load_dtype(Q_smem, qr * HEAD_DIM + d) *
                             load_dtype(K_smem, k_row * HEAD_DIM + d);
                }
                score *= scale;

                float p = __expf(score - lse_smem[qr]);

                float dP = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += load_dtype(dO_smem, qr * HEAD_DIM + d) *
                          load_dtype(V_smem, k_row * HEAD_DIM + d);
                }

                float dS = p * (dP - D_smem[qr]) * scale;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dK_local[d] += dS * load_dtype(Q_smem, qr * HEAD_DIM + d);
                    dV_local[d] += p * load_dtype(dO_smem, qr * HEAD_DIM + d);
                }
            }

            // Paged blocks may be shared across query blocks and query heads,
            // so the single write per element still has to be atomic.
            const int kv_offset = get_paged_kv_offset(
                block_table, batch_idx, max_num_blocks, k_idx, block_size, num_kv_heads, kv_head_idx, HEAD_DIM
            );

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomic_add_dtype(&dK_blocks[kv_offset + d], dK_local[d]);
                atomic_add_dtype(&dV_blocks[kv_offset + d], dV_local[d]);
            }
        }
        __syncthreads();
    }

    // Write dQ output
    if (is_valid_thread) {
        const int out_row = q_start + q_row;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            store_dtype(dQ_base, out_row * HEAD_DIM + d, dQ_local[d]);
        }
    }
}

// ============================================================================
// Kernel Entry Points - runtime block-size selection
//
// BOTH tile variants are emitted unconditionally, as separate symbols:
//   paged_flash_attention_bwd_{head_dim}_{dtype}         large tile (128x64)
//   paged_flash_attention_bwd_{head_dim}_{dtype}_small   small tile
// `bwd_block_config` picks the symbol at runtime; the large tile is reachable
// only through `BOOSTR_PAGED_BWD_TILE=large`. Nothing here is compile-time
// gated on the GPU architecture — the BF16 entries must exist at sm_75, where
// `atomic_add_dtype` takes its CAS fallback.
//
// The small tiles differ per dtype because the shared-memory element is T:
// F32 gets half the rows F16/BF16 do at the same 48KB budget. These pairs must
// stay in sync with `bwd_block_config_small` in
// `paged_attention_bwd_block_config.rs`.
//
// For the 48KB (49152 byte) limit, dynamic + static shared memory:
//   F32  head_dim=64:  32x32 -> (96+64)*64*4  = 40960 + 256 = 41216 bytes
//   F32  head_dim=128: 16x16 -> (48+32)*128*4 = 40960 + 128 = 41088 bytes
//   half head_dim=64:  64x32 -> (192+64)*64*2 = 32768 + 512 = 33280 bytes
//   half head_dim=128: 32x32 -> (96+64)*128*2 = 40960 + 256 = 41216 bytes
// ============================================================================

#define PAGED_BWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)                 \
    extern "C" __global__ void paged_flash_attention_bwd_##HEAD_DIM##_##SUFFIX(\
        const T* Q, const T* K_blocks, const T* V_blocks,                      \
        const T* O, const T* dO, const float* L,                               \
        const int* block_table,                                                \
        T* dQ, T* dK_blocks, T* dV_blocks,                                     \
        int batch_size, int num_heads, int num_kv_heads,                       \
        int seq_len_q, int seq_len_k,                                          \
        int max_num_blocks, int block_size, float scale, int causal            \
    ) {                                                                        \
        paged_attention_bwd_impl<T, HEAD_DIM, BLOCK_M, BLOCK_N>(               \
            Q, K_blocks, V_blocks, O, dO, L, block_table,                      \
            dQ, dK_blocks, dV_blocks,                                          \
            batch_size, num_heads, num_kv_heads, seq_len_q, seq_len_k,         \
            max_num_blocks, block_size, scale, causal                          \
        );                                                                     \
    }

// Large tile (BLOCK_M=128, BLOCK_N=64), identical for every dtype.
PAGED_BWD_ENTRY(float, 64, 128, 64, fp32)
PAGED_BWD_ENTRY(float, 128, 128, 64, fp32)
PAGED_BWD_ENTRY(__half, 64, 128, 64, fp16)
PAGED_BWD_ENTRY(__half, 128, 128, 64, fp16)
PAGED_BWD_ENTRY(__nv_bfloat16, 64, 128, 64, bf16)
PAGED_BWD_ENTRY(__nv_bfloat16, 128, 128, 64, bf16)

// Small tiles, sized per dtype to fit 48KB shared memory.
PAGED_BWD_ENTRY(float, 64, 32, 32, fp32_small)
PAGED_BWD_ENTRY(float, 128, 16, 16, fp32_small)
PAGED_BWD_ENTRY(__half, 64, 64, 32, fp16_small)
PAGED_BWD_ENTRY(__half, 128, 32, 32, fp16_small)
PAGED_BWD_ENTRY(__nv_bfloat16, 64, 64, 32, bf16_small)
PAGED_BWD_ENTRY(__nv_bfloat16, 128, 32, 32, bf16_small)

#undef PAGED_BWD_ENTRY
