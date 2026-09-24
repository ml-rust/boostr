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
//
// Per-row key end: `kv_end` is a `[B]` I32 device array or null, taken by every
// entry point. Row `b`'s keys stop at `min(kv_end[b], seq_len_k)`, read once
// per block, so one block table holds rows of different lengths and the last,
// partly filled page of a short row is clamped to that row's own end rather
// than the batch's. Null leaves every row ending at `seq_len_k`, which is the
// exact arithmetic of a call that passes no per-row end.
//
// With per-row ends the split grid cuts each row's own page run by
// `paged_decode_row_slice`, a rule of the host's `fill` count
// (`ops/cuda/attention/decode_split.rs`) and the row's page count alone, never
// of the batch size, of `gridDim.y` or of another row. The host sizes the grid
// from the block-table capacity, the upper bound on any row's page run, so a
// shorter row's trailing slices come out empty and the number and order of its
// real folds is the same batched or alone.

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
// `row_end` is this row's own last key: it clamps the last, partly filled page
// so a row never reads past its keys into whatever the padding entries of its
// block table point at.
//
// Every thread of the block must call this with the same range: the merge at
// the end is block-wide.
template<typename T, int D>
__device__ __forceinline__ void paged_decode_core(
    const T* __restrict__ q_row,
    const T* __restrict__ K_blocks,
    const T* __restrict__ V_blocks,
    const int* __restrict__ bt,
    int kv_h, int num_kv_heads, int row_end, int block_size,
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
        const int tokens_in_block = min(block_size, row_end - blk * block_size);

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

// Token-unit slice quantum and slice cap, matching DECODE_SLICE_CHUNK and
// DECODE_MAX_SLICES in decode_attention.cu and DECODE_MIN_CHUNK /
// DECODE_MAX_SPLITS in ops/cuda/attention/decode_split.rs.
#define PAGED_DECODE_SLICE_CHUNK 32
#define PAGED_DECODE_MAX_SLICES 32

// Last key of row `b`, clamped to the key axis. A null `kv_end` leaves it at
// `seq_len_k`, the bound the row loop had without a per-row end.
__device__ __forceinline__ int paged_decode_row_end(
    const int* __restrict__ kv_end, int b, int seq_len_k
) {
    return (kv_end != nullptr) ? min(kv_end[b], seq_len_k) : seq_len_k;
}

// Fewest KV pages a slice may own: the token quantum rounded up to whole
// pages, so a slice boundary lands on a block-table entry.
__device__ __forceinline__ int paged_decode_min_chunk_blocks(int block_size) {
    const int page = (block_size > 0) ? block_size : 1;
    const int chunk = (PAGED_DECODE_SLICE_CHUNK + page - 1) / page;
    return (chunk > 0) ? chunk : 1;
}

// Slice `split` of a row owning KV pages `[0, row_blocks)`. The row wants
// `fill` slices, clamped to what its page run allows: at least
// `paged_decode_min_chunk_blocks` pages each and at most
// PAGED_DECODE_MAX_SLICES, one slice when fewer than two chunks exist. Each
// slice is `ceil(row_blocks / want)` pages rounded up to the chunk. Mirrors
// `paged_row_splits` in ops/cuda/attention/decode_split.rs, which sizes the
// grid from the block-table capacity; a shorter row's trailing slices are
// empty. Reads neither the batch size nor `gridDim.y`, so a row's cut is the
// same whatever it is batched with.
__device__ __forceinline__ void paged_decode_row_slice(
    int row_blocks, int fill, int split, int block_size, int& begin, int& end
) {
    if (row_blocks <= 0) {
        begin = 0;
        end = 0;
        return;
    }
    const int chunk = paged_decode_min_chunk_blocks(block_size);
    const int max_splits = min(row_blocks / chunk, PAGED_DECODE_MAX_SLICES);
    const int want = (max_splits < 2) ? 1 : min(max(fill, 1), max_splits);
    const int per = (row_blocks + want - 1) / want;
    const int len = (per + chunk - 1) / chunk * chunk;
    begin = split * len;
    end = min(begin + len, row_blocks);
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
    int block_size, float scale,
    const int* __restrict__ kv_end
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const int* bt = block_table + b * max_num_blocks;
    const int row_end = paged_decode_row_end(kv_end, b, seq_len_k);
    const int num_kv_blocks = (row_end + block_size - 1) / block_size;

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    // Block-uniform: an empty page run skips the core and its barriers
    // together. A row whose end is zero has no keys at all.
    if (num_kv_blocks > 0)
        paged_decode_core<T, D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                                row_end, block_size, 0, num_kv_blocks, scale, acc, m, l);

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
//
// With a null `kv_end` every row runs to `seq_len_k` and the page run is cut
// into `num_splits` equal parts, the arithmetic of a call without per-row
// ends. With per-row ends each row cuts its own page run by
// `paged_decode_row_slice`; slices past that run write `m = -inf, l = 0`,
// which the combine pass drops without contributing to the global maximum.
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
    int block_size, float scale, int num_splits,
    const int* __restrict__ kv_end, int fill
) {
    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const int* bt = block_table + b * max_num_blocks;

    const int row_end = paged_decode_row_end(kv_end, b, seq_len_k);
    const int num_kv_blocks = (row_end + block_size - 1) / block_size;
    int blk_begin;
    int blk_end;
    if (kv_end != nullptr) {
        paged_decode_row_slice(num_kv_blocks, fill, split, block_size, blk_begin, blk_end);
    } else {
        const int blocks_per_split = (num_kv_blocks + num_splits - 1) / num_splits;
        blk_begin = split * blocks_per_split;
        blk_end = min(blk_begin + blocks_per_split, num_kv_blocks);
    }

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    // Block-uniform: the bounds derive only from block indices, so the barriers
    // inside the core stay reachable by the whole block.
    if (blk_begin < blk_end)
        paged_decode_core<T, D>(q_row, K_blocks, V_blocks, bt, kv_h, num_kv_heads,
                                row_end, block_size, blk_begin, blk_end, scale, acc, m, l);

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

// Non-graph and graph variants take the same arguments apart from `seq_len_k`:
// a plain int here, a device pointer in the `_graph` entries, dereferenced once
// as the row-independent clamp in `min(kv_end[b], seq_len_k)`. Every entry
// takes `kv_end`; null in it gives the arithmetic of a call without per-row
// ends.
#define PAGED_DECODE_KERNELS(D, SUFFIX, T)                                          \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX(                   \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    T* __restrict__ O, float* __restrict__ LSE,                                     \
    int num_heads, int num_kv_heads, int seq_len_k, int max_num_blocks,             \
    int block_size, float scale, const int* __restrict__ kv_end                     \
) {                                                                                 \
    paged_decode_impl<T, D>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,  \
                            num_kv_heads, seq_len_k, max_num_blocks, block_size,    \
                            scale, kv_end);                                         \
}                                                                                   \
                                                                                    \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX##_graph(           \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    T* __restrict__ O, float* __restrict__ LSE,                                     \
    int num_heads, int num_kv_heads, const int* __restrict__ seq_len_k_ptr,         \
    int max_num_blocks, int block_size, float scale,                                \
    const int* __restrict__ kv_end                                                  \
) {                                                                                 \
    paged_decode_impl<T, D>(Q, K_blocks, V_blocks, block_table, O, LSE, num_heads,  \
                            num_kv_heads, *seq_len_k_ptr, max_num_blocks,           \
                            block_size, scale, kv_end);                             \
}                                                                                   \
                                                                                    \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX##_split(           \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    float* __restrict__ partial_o, float* __restrict__ partial_ml,                  \
    int num_heads, int num_kv_heads, int seq_len_k, int max_num_blocks,             \
    int block_size, float scale, int num_splits,                                    \
    const int* __restrict__ kv_end, int fill                                        \
) {                                                                                 \
    paged_decode_split_impl<T, D>(Q, K_blocks, V_blocks, block_table, partial_o,    \
                                  partial_ml, num_heads, num_kv_heads, seq_len_k,   \
                                  max_num_blocks, block_size, scale, num_splits,    \
                                  kv_end, fill);                                    \
}                                                                                   \
                                                                                    \
extern "C" __global__ void paged_decode_attention_##D##_##SUFFIX##_split_graph(     \
    const T* __restrict__ Q, const T* __restrict__ K_blocks,                        \
    const T* __restrict__ V_blocks, const int* __restrict__ block_table,            \
    float* __restrict__ partial_o, float* __restrict__ partial_ml,                  \
    int num_heads, int num_kv_heads, const int* __restrict__ seq_len_k_ptr,         \
    int max_num_blocks, int block_size, float scale, int num_splits,                \
    const int* __restrict__ kv_end, int fill                                        \
) {                                                                                 \
    paged_decode_split_impl<T, D>(Q, K_blocks, V_blocks, block_table, partial_o,    \
                                  partial_ml, num_heads, num_kv_heads,              \
                                  *seq_len_k_ptr, max_num_blocks, block_size,       \
                                  scale, num_splits, kv_end, fill);                 \
}

PAGED_DECODE_KERNELS(64, fp32, float)
PAGED_DECODE_KERNELS(128, fp32, float)
PAGED_DECODE_KERNELS(64, fp16, __half)
PAGED_DECODE_KERNELS(128, fp16, __half)
PAGED_DECODE_KERNELS(64, bf16, __nv_bfloat16)
PAGED_DECODE_KERNELS(128, bf16, __nv_bfloat16)
