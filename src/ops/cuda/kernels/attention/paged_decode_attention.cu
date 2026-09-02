// Paged decode attention kernel — S_q=1 specialized with block_table lookup
//
// Same algorithm as decode_attention.cu but reads K/V from paged blocks
// via block_table indirection. GQA-aware: kv_h = h / (num_heads / num_kv_heads).
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Q layout: [B, num_heads, 1, D]
// Output: [B, num_heads, 1, D], LSE: [B, num_heads, 1]
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

#include <cuda_runtime.h>
#include <stdint.h>

// Positions consumed per barrier round. The block reduces one dot product per
// position through shared memory, so a round costs two `__syncthreads()`
// whatever it covers; batching positions amortizes those barriers and keeps
// that many independent K loads in flight.
#define PAGED_DECODE_PER_ITER 4

// ============================================================================
// Online-softmax pass over a run of KV blocks
// ============================================================================

// Accumulates KV blocks `[blk_begin, blk_end)` into this thread's output
// dimension, leaving the accumulator UNNORMALIZED alongside the running max `m`
// and denominator `l`.
//
// Every thread of the block must call this with the same range: it contains
// block-wide barriers.
template<int D>
__device__ __forceinline__ void paged_decode_core(
    const float* __restrict__ q_row,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ bt,
    int kv_h, int num_kv_heads, int seq_len_k, int block_size,
    int blk_begin, int blk_end, float scale,
    float& acc, float& m, float& l
) {
    constexpr int NW = D / 32;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem_qk[PAGED_DECODE_PER_ITER][NW];

    const float q_val = q_row[tid] * scale;

    acc = 0.0f;
    m = -INFINITY;
    l = 0.0f;

    const int kv_stride = num_kv_heads * D;

    for (int blk = blk_begin; blk < blk_end; blk++) {
        const int physical_block = bt[blk];
        const int tokens_in_block = min(block_size, seq_len_k - blk * block_size);

        const size_t block_base = (size_t)physical_block * block_size * num_kv_heads * D
                                + (size_t)kv_h * D;

        int off = 0;
        for (; off + PAGED_DECODE_PER_ITER <= tokens_in_block; off += PAGED_DECODE_PER_ITER) {
            float qk[PAGED_DECODE_PER_ITER];
            #pragma unroll
            for (int r = 0; r < PAGED_DECODE_PER_ITER; r++)
                qk[r] = q_val * K_blocks[block_base + (size_t)(off + r) * kv_stride + tid];

            #pragma unroll
            for (int r = 0; r < PAGED_DECODE_PER_ITER; r++) {
                #pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    qk[r] += __shfl_down_sync(0xFFFFFFFF, qk[r], s);
            }

            // Guards the previous round's readers against this round's writes.
            __syncthreads();
            if (lane_id == 0) {
                #pragma unroll
                for (int r = 0; r < PAGED_DECODE_PER_ITER; r++)
                    smem_qk[r][warp_id] = qk[r];
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < PAGED_DECODE_PER_ITER; r++) {
                float dot = 0.0f;
                #pragma unroll
                for (int w = 0; w < NW; w++)
                    dot += smem_qk[r][w];

                float v_val = V_blocks[block_base + (size_t)(off + r) * kv_stride + tid];
                float m_new = fmaxf(m, dot);
                float exp_old = expf(m - m_new);
                float exp_new = expf(dot - m_new);

                acc = acc * exp_old + v_val * exp_new;
                l = l * exp_old + exp_new;
                m = m_new;
            }
        }

        for (; off < tokens_in_block; off++) {
            float qk = q_val * K_blocks[block_base + (size_t)off * kv_stride + tid];

            #pragma unroll
            for (int s = 16; s > 0; s >>= 1)
                qk += __shfl_down_sync(0xFFFFFFFF, qk, s);

            __syncthreads();
            if (lane_id == 0) smem_qk[0][warp_id] = qk;
            __syncthreads();

            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < NW; w++)
                dot += smem_qk[0][w];

            float v_val = V_blocks[block_base + (size_t)off * kv_stride + tid];
            float m_new = fmaxf(m, dot);
            float exp_old = expf(m - m_new);
            float exp_new = expf(dot - m_new);

            acc = acc * exp_old + v_val * exp_new;
            l = l * exp_old + exp_new;
            m = m_new;
        }
    }
}

// ============================================================================
// Whole-sequence kernel: one block per (batch, Q head)
// ============================================================================

template<int D>
__device__ __forceinline__ void paged_decode_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
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

    const float* q_row = Q + (size_t)(b * num_heads + h) * D;
    const int* bt = block_table + b * max_num_blocks;
    const int num_kv_blocks = (seq_len_k + block_size - 1) / block_size;

    float acc, m, l;
    paged_decode_core<D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                         seq_len_k, block_size, 0, num_kv_blocks, scale, acc, m, l);

    O[(size_t)(b * num_heads + h) * D + tid] = (l > 0.0f) ? acc / l : 0.0f;
    if (tid == 0)
        LSE[b * num_heads + h] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Split-KV kernel: grid (batch * num_heads, num_splits)
// ============================================================================

// `partial_o` is [B * num_heads, num_splits, D] and `partial_ml` is
// [B * num_heads, num_splits, 2] — the layout the shared combine kernel reads.
template<int D>
__device__ __forceinline__ void paged_decode_split_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
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

    const float* q_row = Q + (size_t)(b * num_heads + h) * D;
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
        paged_decode_core<D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                             seq_len_k, block_size, blk_begin, blk_end, scale, acc, m, l);

    const size_t slot = (size_t)bh * num_splits + split;
    partial_o[slot * D + tid] = acc;
    if (tid == 0) {
        partial_ml[slot * 2] = m;
        partial_ml[slot * 2 + 1] = l;
    }
}

// ============================================================================
// Non-graph entry points: seq_len_k as plain int
// ============================================================================

extern "C" __global__ void paged_decode_attention_128_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale
) {
    paged_decode_impl<128>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,
                           num_kv_heads, seq_len_k, max_num_blocks, block_size, scale);
}

extern "C" __global__ void paged_decode_attention_64_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale
) {
    paged_decode_impl<64>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,
                          num_kv_heads, seq_len_k, max_num_blocks, block_size, scale);
}

extern "C" __global__ void paged_decode_attention_128_fp32_split(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale, int num_splits
) {
    paged_decode_split_impl<128>(Q, K_blocks, V_blocks, block_table, partial_o, partial_ml,
                                 num_heads, num_kv_heads, seq_len_k, max_num_blocks,
                                 block_size, scale, num_splits);
}

extern "C" __global__ void paged_decode_attention_64_fp32_split(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale, int num_splits
) {
    paged_decode_split_impl<64>(Q, K_blocks, V_blocks, block_table, partial_o, partial_ml,
                                num_heads, num_kv_heads, seq_len_k, max_num_blocks,
                                block_size, scale, num_splits);
}

// ============================================================================
// Graph-mode entry points: seq_len_k from device pointer
// ============================================================================

extern "C" __global__ void paged_decode_attention_128_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    const int* __restrict__ seq_len_k_ptr,
    int max_num_blocks,
    int block_size, float scale
) {
    paged_decode_impl<128>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,
                           num_kv_heads, *seq_len_k_ptr, max_num_blocks, block_size, scale);
}

extern "C" __global__ void paged_decode_attention_64_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    const int* __restrict__ seq_len_k_ptr,
    int max_num_blocks,
    int block_size, float scale
) {
    paged_decode_impl<64>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,
                          num_kv_heads, *seq_len_k_ptr, max_num_blocks, block_size, scale);
}
