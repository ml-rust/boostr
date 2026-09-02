// Decode attention kernel — optimized for S_q=1 (autoregressive decoding)
//
// Inspired by llama.cpp fattn-vec and vLLM's paged_attention.
// For contiguous KV cache (no paging).
//
// Layout: Q [B, num_heads, 1, D], K/V [B, num_kv_heads, seq_k, D]
// Output: O [B, num_heads, 1, D], LSE [B, num_heads, 1] (always F32)
//
// Q/K/V/O carry the tensor dtype; the softmax state and the accumulator are
// always F32. Serving runs in F16/BF16, so a decode kernel that existed only in
// F32 sent every real request to the tiled prefill kernel instead.
//
// Two grid shapes:
//   - Whole-sequence: one block per (batch, head). The grid is then
//     `batch * num_heads` however long the KV sequence is, so a small batch
//     leaves most of the device idle at long context.
//   - Split-KV: grid (batch * num_heads, num_splits). Each block owns a
//     contiguous slice of the KV sequence and writes an unnormalized partial
//     accumulator with its own `(m, l)` softmax statistics; a combine pass
//     merges the slices. The host picks `num_splits` from the device's compute
//     unit count, so the grid grows with the sequence until the device is full.
//
// Two variants of each:
//   - Non-graph: seq_len_k passed as plain int kernel arg (zero overhead)
//   - Graph-mode (_graph suffix): seq_len_k_ptr is a device pointer to i32,
//     kv_seq_stride is the memory stride (capacity >= seq_len_k)
//
// Sliding window: `window_size == 0` disables it. Otherwise a key `j` is masked
// when `j + window_size <= i`, where `i` is the query's absolute position. Decode
// is single-token, so `i == seq_len_k - 1` and the surviving keys are exactly
// `j >= seq_len_k - window_size` — a contiguous suffix, so the loop just starts
// later. Matches the kernel contract in
// ops/impl_generic/attention/flash_standard.rs.

#include "../dtype_traits.cuh"
#include "decode_warp_merge.cuh"

// ============================================================================
// Online-softmax pass over a contiguous KV range
// ============================================================================
//
// One WARP owns one KV position. Each lane holds `D / 32` of the head's
// dimensions, so a position's dot product is a single warp reduction with no
// shared memory and no barrier.
//
// The obvious alternative — the whole block cooperating on one position — was
// measured and is much worse: every warp redundantly repeats the same scalar
// reduction, the same `expf` pair and the same `m`/`l` update, then pays a
// shared-memory round trip and two `__syncthreads()` per position. That
// redundancy was over half of all executed instructions, and it made the
// grouped-query case compute-bound at 83% SM throughput while DRAM sat at 54%.
//
// Each warp therefore accumulates over a strided subset of the range and ends
// with its own `(acc, m, l)`. Merging those is the same rescaling the split-KV
// combine does, once per block instead of once per position.

// Accumulates `[pos_begin, pos_end)` into this thread's output dimension.
// Leaves the accumulator UNNORMALIZED alongside the running max `m` and
// denominator `l`, so the same pass serves both the whole-sequence kernel
// (which divides immediately) and a split slice (which defers to the combine).
//
// Every thread of the block must call this with the same range: the merge at
// the end is block-wide.
template<typename T, int D>
__device__ __forceinline__ void decode_attention_core(
    const T* __restrict__ q_row,
    const T* __restrict__ k_base,
    const T* __restrict__ v_base,
    int pos_begin, int pos_end, float scale,
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

    // Lane `l` holds dimensions `l, l + 32, l + 64, ...` so each of the VPT
    // loads is 32 consecutive elements across the warp — one coalesced request.
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

    for (int pos = pos_begin + warp_id; pos < pos_end; pos += NW) {
        const size_t k_off = (size_t)pos * D + lane_id;

        float dot = 0.0f;
        #pragma unroll
        for (int u = 0; u < VPT; u++)
            dot += q_reg[u] * convert_dtype<float>(k_base[k_off + u * DECODE_LANES]);

        // Butterfly reduction: every lane ends with the full sum, so no
        // broadcast step is needed.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

        float m_new = fmaxf(w_m, dot);
        float exp_old = expf(w_m - m_new);
        float exp_new = expf(dot - m_new);

        const size_t v_off = (size_t)pos * D + lane_id;
        #pragma unroll
        for (int u = 0; u < VPT; u++) {
            float v_val = convert_dtype<float>(v_base[v_off + u * DECODE_LANES]);
            w_acc[u] = w_acc[u] * exp_old + v_val * exp_new;
        }
        w_l = w_l * exp_old + exp_new;
        w_m = m_new;
    }

    // Publish this warp's partial, indexed by the dimension each lane holds.
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
// Whole-sequence kernel: one block per (batch, head)
// ============================================================================

template<typename T, int D>
__device__ __forceinline__ void decode_attention_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale, int window_size
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const T* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;
    const T* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;

    // Sliding window keeps the last `window_size` keys; `0` disables it.
    const int pos_start = (window_size > 0) ? max(0, seq_len_k - window_size) : 0;

    float acc, m, l;
    decode_attention_core<T, D>(q_row, k_base, v_base, pos_start, seq_len_k, scale, acc, m, l);

    O[(size_t)(b * num_heads + h) * D + tid] =
        convert_dtype<T>((l > 0.0f) ? acc / l : 0.0f);
    if (tid == 0)
        LSE[b * num_heads + h] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Split-KV kernel: grid (batch * num_heads, num_splits)
// ============================================================================

// `partial_o` is [B * num_heads, num_splits, D] and `partial_ml` is
// [B * num_heads, num_splits, 2] holding `(m, l)` per slice, both always F32.
// An empty slice writes `m = -inf, l = 0`, which the combine pass drops without
// contributing to the global maximum.
template<typename T, int D>
__device__ __forceinline__ void decode_attention_split_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale, int window_size, int num_splits
) {
    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const T* q_row = Q + (size_t)(b * num_heads + h) * D;
    const T* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;
    const T* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;

    const int pos_start = (window_size > 0) ? max(0, seq_len_k - window_size) : 0;
    const int span = seq_len_k - pos_start;
    const int chunk = (span + num_splits - 1) / num_splits;
    const int begin = pos_start + split * chunk;
    const int end = min(begin + chunk, seq_len_k);

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    // Block-uniform: `begin` and `end` derive only from block indices, so the
    // barriers inside the core stay reachable by the whole block.
    if (begin < end)
        decode_attention_core<T, D>(q_row, k_base, v_base, begin, end, scale, acc, m, l);

    const size_t slot = (size_t)bh * num_splits + split;
    partial_o[slot * D + tid] = acc;
    if (tid == 0) {
        partial_ml[slot * 2] = m;
        partial_ml[slot * 2 + 1] = l;
    }
}

// Merges the per-slice partials into the final output and log-sum-exp.
// One block per (batch, head); `num_splits` is small, so each thread walks the
// slices directly rather than reducing through shared memory.
template<typename T, int D>
__device__ __forceinline__ void decode_attention_combine_impl(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_ml,
    T* __restrict__ O,
    float* __restrict__ LSE,
    int num_splits
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    const float* ml = partial_ml + (size_t)bh * num_splits * 2;
    const float* po = partial_o + (size_t)bh * num_splits * D;

    float m_max = -INFINITY;
    for (int s = 0; s < num_splits; s++)
        if (ml[s * 2 + 1] > 0.0f) m_max = fmaxf(m_max, ml[s * 2]);

    float acc = 0.0f;
    float l_total = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float l_s = ml[s * 2 + 1];
        if (l_s <= 0.0f) continue;
        float w = expf(ml[s * 2] - m_max);
        acc += po[s * D + tid] * w;
        l_total += l_s * w;
    }

    O[(size_t)bh * D + tid] = convert_dtype<T>((l_total > 0.0f) ? acc / l_total : 0.0f);
    if (tid == 0)
        LSE[bh] = (l_total > 0.0f) ? (m_max + logf(l_total)) : -INFINITY;
}

// ============================================================================
// Entry points, one set per (head_dim, dtype)
// ============================================================================

// The non-graph variants are only dispatched for window_size == 0 (see
// ops/cuda/attention/flash.rs); windowed non-graph decode goes to flash_v2.
#define DECODE_ATTENTION_KERNELS(D, SUFFIX, T)                                     \
extern "C" __global__ void decode_attention_##D##_##SUFFIX(                        \
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,     \
    T* __restrict__ O, float* __restrict__ LSE,                                    \
    int num_heads, int num_kv_heads, int seq_len_k, int kv_seq_stride, float scale  \
) {                                                                                \
    decode_attention_impl<T, D>(Q, K, V, O, LSE, num_heads, num_kv_heads,          \
                                seq_len_k, kv_seq_stride, scale, 0);               \
}                                                                                  \
                                                                                   \
extern "C" __global__ void decode_attention_##D##_##SUFFIX##_graph(                \
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,     \
    T* __restrict__ O, float* __restrict__ LSE,                                    \
    int num_heads, int num_kv_heads, const int* seq_len_k_ptr,                     \
    int kv_seq_stride, float scale, int window_size                                \
) {                                                                                \
    decode_attention_impl<T, D>(Q, K, V, O, LSE, num_heads, num_kv_heads,          \
                                *seq_len_k_ptr, kv_seq_stride, scale, window_size);\
}                                                                                  \
                                                                                   \
extern "C" __global__ void decode_attention_##D##_##SUFFIX##_split(                \
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,     \
    float* __restrict__ partial_o, float* __restrict__ partial_ml,                 \
    int num_heads, int num_kv_heads, int seq_len_k, int kv_seq_stride,             \
    float scale, int num_splits                                                    \
) {                                                                                \
    decode_attention_split_impl<T, D>(Q, K, V, partial_o, partial_ml, num_heads,   \
                                      num_kv_heads, seq_len_k, kv_seq_stride,      \
                                      scale, 0, num_splits);                       \
}                                                                                  \
                                                                                   \
extern "C" __global__ void decode_attention_##D##_##SUFFIX##_combine(              \
    const float* __restrict__ partial_o, const float* __restrict__ partial_ml,     \
    T* __restrict__ O, float* __restrict__ LSE, int num_splits                     \
) {                                                                                \
    decode_attention_combine_impl<T, D>(partial_o, partial_ml, O, LSE, num_splits);\
}

DECODE_ATTENTION_KERNELS(64, fp32, float)
DECODE_ATTENTION_KERNELS(128, fp32, float)
DECODE_ATTENTION_KERNELS(64, fp16, __half)
DECODE_ATTENTION_KERNELS(128, fp16, __half)
DECODE_ATTENTION_KERNELS(64, bf16, __nv_bfloat16)
DECODE_ATTENTION_KERNELS(128, bf16, __nv_bfloat16)
