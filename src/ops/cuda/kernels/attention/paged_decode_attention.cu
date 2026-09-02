// Paged decode attention kernel — S_q=1 specialized with block_table lookup
//
// Same algorithm as decode_attention.cu but reads K/V from paged blocks
// via block_table indirection. GQA-aware: kv_h = h / (num_heads / num_kv_heads).
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Q layout: [B, num_heads, 1, D]
// Output: [B, num_heads, 1, D], LSE: [B, num_heads, 1] (always F32)
//
// Q/K/V/O carry the tensor dtype; the softmax state and the accumulator are
// always F32.
//
// Two grid shapes, matching decode_attention.cu:
//   - Whole-sequence: one block per (batch, Q head). The grid is then
//     `batch * num_heads` however long the KV sequence is.
//   - Split-KV: grid (batch * num_heads, num_splits), each block owning a
//     contiguous run of KV blocks and writing an unnormalized partial with its
//     own (m, l). The combine pass is shared with the contiguous decode kernel
//     (`decode_attention_{D}_fp32_combine`) — the partials have the same layout.
//
// Splitting on KV blocks rather than tokens keeps every slice boundary aligned
// with a block_table entry, so a slice never straddles a page.

#include "../dtype_traits.cuh"
#include "decode_warp_merge.cuh"
#include <stdint.h>

// ============================================================================
// Online-softmax pass over a run of KV blocks
// ============================================================================
//
// One WARP owns one KV position, matching decode_attention.cu. Each lane holds
// `D / 32` of the head's dimensions, so a position's dot product is a single
// warp reduction with no shared memory and no barrier. Warps split the offsets
// WITHIN a page rather than striding across pages, so they share one block
// table entry and no per-position division is needed.
//
// See decode_attention.cu for why the block-per-position shape this replaced
// was the wrong one.

// Accumulates KV blocks `[blk_begin, blk_end)` into this thread's output
// dimension, leaving the accumulator UNNORMALIZED alongside the running max `m`
// and denominator `l`.
//
// Every thread of the block must call this with the same range: the merge at
// the end is block-wide.
template<typename T, int D>
__device__ __forceinline__ void paged_decode_core(
    const T* __restrict__ q_row,
    const T* __restrict__ K_blocks,
    const T* __restrict__ V_blocks,
    const int* __restrict__ bt,
    int kv_h, int num_kv_heads, int seq_len_k, int block_size,
    int blk_begin, int blk_end, float scale,
    float& acc, float& m, float& l
) {
    constexpr int NW = D / DECODE_LANES;
    constexpr int VPT = D / DECODE_LANES;

    const int tid = threadIdx.x;
    const int warp_id = tid / DECODE_LANES;
    const int lane_id = tid % DECODE_LANES;

    __shared__ float smem_acc[NW][D];
    __shared__ float smem_m[NW];
    __shared__ float smem_l[NW];

    float q_reg[VPT];
    #pragma unroll
    for (int u = 0; u < VPT; u++)
        q_reg[u] = convert_dtype<float>(q_row[lane_id + u * DECODE_LANES]) * scale;

    float w_acc[VPT];
    #pragma unroll
    for (int u = 0; u < VPT; u++)
        w_acc[u] = 0.0f;
    float w_m = -INFINITY;
    float w_l = 0.0f;

    const int kv_stride = num_kv_heads * D;

    for (int blk = blk_begin; blk < blk_end; blk++) {
        const int physical_block = bt[blk];
        const int tokens_in_block = min(block_size, seq_len_k - blk * block_size);

        const size_t block_base = (size_t)physical_block * block_size * num_kv_heads * D
                                + (size_t)kv_h * D;

        for (int off = warp_id; off < tokens_in_block; off += NW) {
            const size_t base = block_base + (size_t)off * kv_stride + lane_id;

            float dot = 0.0f;
            #pragma unroll
            for (int u = 0; u < VPT; u++)
                dot += q_reg[u] * convert_dtype<float>(K_blocks[base + u * DECODE_LANES]);

            #pragma unroll
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, s);

            float m_new = fmaxf(w_m, dot);
            float exp_old = expf(w_m - m_new);
            float exp_new = expf(dot - m_new);

            #pragma unroll
            for (int u = 0; u < VPT; u++) {
                float v_val = convert_dtype<float>(V_blocks[base + u * DECODE_LANES]);
                w_acc[u] = w_acc[u] * exp_old + v_val * exp_new;
            }
            w_l = w_l * exp_old + exp_new;
            w_m = m_new;
        }
    }

    #pragma unroll
    for (int u = 0; u < VPT; u++)
        smem_acc[warp_id][lane_id + u * DECODE_LANES] = w_acc[u];
    if (lane_id == 0) {
        smem_m[warp_id] = w_m;
        smem_l[warp_id] = w_l;
    }
    __syncthreads();

    decode_merge_warps<D>(&smem_acc[0][0], smem_m, smem_l, acc, m, l);
}

// ============================================================================
// Whole-sequence kernel: one block per (batch, Q head)
// ============================================================================

template<typename T, int D>
__device__ __forceinline__ void paged_decode_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K_blocks,
    const T* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    T* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const int* bt = block_table + b * max_num_blocks;
    const int num_kv_blocks = (seq_len_k + block_size - 1) / block_size;

    float acc, m, l;
    paged_decode_core<T, D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                            seq_len_k, block_size, 0, num_kv_blocks, scale, acc, m, l);

    O[(size_t)(b * num_heads + h) * D + tid] =
        convert_dtype<T>((l > 0.0f) ? acc / l : 0.0f);
    if (tid == 0)
        LSE[b * num_heads + h] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Split-KV kernel: grid (batch * num_heads, num_splits)
// ============================================================================

// `partial_o` is [B * num_heads, num_splits, D] and `partial_ml` is
// [B * num_heads, num_splits, 2] — the layout the shared combine kernel reads.
template<typename T, int D>
__device__ __forceinline__ void paged_decode_split_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K_blocks,
    const T* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale, int num_splits
) {
    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const int* bt = block_table + b * max_num_blocks;

    const int num_kv_blocks = (seq_len_k + block_size - 1) / block_size;
    const int blocks_per_split = (num_kv_blocks + num_splits - 1) / num_splits;
    const int blk_begin = split * blocks_per_split;
    const int blk_end = min(blk_begin + blocks_per_split, num_kv_blocks);

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    // Block-uniform: the bounds derive only from block indices, so the barriers
    // inside the core stay reachable by the whole block.
    if (blk_begin < blk_end)
        paged_decode_core<T, D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                                seq_len_k, block_size, blk_begin, blk_end, scale, acc, m, l);

    const size_t slot = (size_t)bh * num_splits + split;
    partial_o[slot * D + tid] = acc;
    if (tid == 0) {
        partial_ml[slot * 2] = m;
        partial_ml[slot * 2 + 1] = l;
    }
}

// ============================================================================
// Entry points, one set per (head_dim, dtype)
// ============================================================================

#define PAGED_DECODE_KERNELS(D, SUFFIX, T)                                          \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX(                   \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    T* __restrict__ O, float* __restrict__ LSE,                                     \
    int num_heads, int num_kv_heads, int seq_len_k, int max_num_blocks,             \
    int block_size, float scale                                                     \
) {                                                                                 \
    paged_decode_impl<T, D>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,  \
                            num_kv_heads, seq_len_k, max_num_blocks, block_size,    \
                            scale);                                                 \
}                                                                                   \
                                                                                    \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX##_graph(           \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    T* __restrict__ O, float* __restrict__ LSE,                                     \
    int num_heads, int num_kv_heads, const int* __restrict__ seq_len_k_ptr,         \
    int max_num_blocks, int block_size, float scale                                 \
) {                                                                                 \
    paged_decode_impl<T, D>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,  \
                            num_kv_heads, *seq_len_k_ptr, max_num_blocks,           \
                            block_size, scale);                                     \
}                                                                                   \
                                                                                    \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX##_split(           \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    float* __restrict__ partial_o, float* __restrict__ partial_ml,                  \
    int num_heads, int num_kv_heads, int seq_len_k, int max_num_blocks,             \
    int block_size, float scale, int num_splits                                     \
) {                                                                                 \
    paged_decode_split_impl<T, D>(Q, K_blocks, V_blocks, block_table, partial_o,    \
                                  partial_ml, num_heads, num_kv_heads, seq_len_k,   \
                                  max_num_blocks, block_size, scale, num_splits);   \
}

PAGED_DECODE_KERNELS(64, fp32, float)
PAGED_DECODE_KERNELS(128, fp32, float)
PAGED_DECODE_KERNELS(64, fp16, __half)
PAGED_DECODE_KERNELS(128, fp16, __half)
PAGED_DECODE_KERNELS(64, bf16, __nv_bfloat16)
PAGED_DECODE_KERNELS(128, bf16, __nv_bfloat16)
