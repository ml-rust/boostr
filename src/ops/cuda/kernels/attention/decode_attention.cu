// Decode attention kernel — optimized for S_q=1 (autoregressive decoding)
//
// Inspired by llama.cpp fattn-vec and vLLM's paged_attention.
// For contiguous KV cache (no paging).
//
// Layout: Q [B, num_heads, 1, D], K/V [B, num_kv_heads, seq_k, D]
// Output: O [B, num_heads, 1, D], LSE [B, num_heads, 1]
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

// Positions consumed per barrier round. The block reduces one dot product per
// position through shared memory, so a round costs two `__syncthreads()`
// whatever it covers; batching positions amortizes those barriers and keeps
// that many independent K loads in flight.
#define DECODE_PER_ITER 4

// ============================================================================
// Online-softmax pass over a contiguous KV range
// ============================================================================

// Accumulates `[pos_begin, pos_end)` into this thread's output dimension.
// Returns the running accumulator UNNORMALIZED, alongside the running max `m`
// and denominator `l`, so the same pass serves both the whole-sequence kernel
// (which divides immediately) and a split slice (which defers to the combine).
//
// Every thread of the block must call this with the same range: it contains
// block-wide barriers.
template<int D>
__device__ __forceinline__ void decode_attention_core(
    const float* __restrict__ q_row,
    const float* __restrict__ k_base,
    const float* __restrict__ v_base,
    int pos_begin, int pos_end, float scale,
    float& acc, float& m, float& l
) {
    constexpr int NW = D / 32;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem_qk[DECODE_PER_ITER][NW];

    const float q_val = q_row[tid] * scale;

    acc = 0.0f;
    m = -INFINITY;
    l = 0.0f;

    int pos = pos_begin;
    for (; pos + DECODE_PER_ITER <= pos_end; pos += DECODE_PER_ITER) {
        float qk[DECODE_PER_ITER];
        #pragma unroll
        for (int r = 0; r < DECODE_PER_ITER; r++)
            qk[r] = q_val * k_base[(size_t)(pos + r) * D + tid];

        #pragma unroll
        for (int r = 0; r < DECODE_PER_ITER; r++) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                qk[r] += __shfl_down_sync(0xFFFFFFFF, qk[r], off);
        }

        // Guards the previous round's readers against this round's writes.
        __syncthreads();
        if (lane_id == 0) {
            #pragma unroll
            for (int r = 0; r < DECODE_PER_ITER; r++)
                smem_qk[r][warp_id] = qk[r];
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < DECODE_PER_ITER; r++) {
            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < NW; w++)
                dot += smem_qk[r][w];

            float v_val = v_base[(size_t)(pos + r) * D + tid];
            float m_new = fmaxf(m, dot);
            float exp_old = expf(m - m_new);
            float exp_new = expf(dot - m_new);

            acc = acc * exp_old + v_val * exp_new;
            l = l * exp_old + exp_new;
            m = m_new;
        }
    }

    for (; pos < pos_end; pos++) {
        float qk = q_val * k_base[(size_t)pos * D + tid];

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            qk += __shfl_down_sync(0xFFFFFFFF, qk, off);

        __syncthreads();
        if (lane_id == 0) smem_qk[0][warp_id] = qk;
        __syncthreads();

        float dot = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; w++)
            dot += smem_qk[0][w];

        float v_val = v_base[(size_t)pos * D + tid];
        float m_new = fmaxf(m, dot);
        float exp_old = expf(m - m_new);
        float exp_new = expf(dot - m_new);

        acc = acc * exp_old + v_val * exp_new;
        l = l * exp_old + exp_new;
        m = m_new;
    }
}

// ============================================================================
// Whole-sequence kernel: one block per (batch, head)
// ============================================================================

template<int D>
__device__ __forceinline__ void decode_attention_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
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

    const float* q_row = Q + (size_t)(b * num_heads + h) * D;
    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;

    // Sliding window keeps the last `window_size` keys; `0` disables it.
    const int pos_start = (window_size > 0) ? max(0, seq_len_k - window_size) : 0;

    float acc, m, l;
    decode_attention_core<D>(q_row, k_base, v_base, pos_start, seq_len_k, scale, acc, m, l);

    O[(size_t)(b * num_heads + h) * D + tid] = (l > 0.0f) ? acc / l : 0.0f;
    if (tid == 0)
        LSE[b * num_heads + h] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Split-KV kernel: grid (batch * num_heads, num_splits)
// ============================================================================

// `partial_o` is [B * num_heads, num_splits, D] and `partial_ml` is
// [B * num_heads, num_splits, 2] holding `(m, l)` per slice. An empty slice
// writes `m = -inf, l = 0`, which the combine pass drops without contributing
// to the global maximum.
template<int D>
__device__ __forceinline__ void decode_attention_split_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
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

    const float* q_row = Q + (size_t)(b * num_heads + h) * D;
    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * D;

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
        decode_attention_core<D>(q_row, k_base, v_base, begin, end, scale, acc, m, l);

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
template<int D>
__device__ __forceinline__ void decode_attention_combine_impl(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_ml,
    float* __restrict__ O,
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

    O[(size_t)bh * D + tid] = (l_total > 0.0f) ? acc / l_total : 0.0f;
    if (tid == 0)
        LSE[bh] = (l_total > 0.0f) ? (m_max + logf(l_total)) : -INFINITY;
}

// ============================================================================
// Non-graph entry points: seq_len_k as plain int (zero overhead)
// ============================================================================

extern "C" __global__ void decode_attention_128_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    // Non-graph decode is only dispatched for window_size == 0 (see
    // ops/cuda/attention/flash.rs); windowed non-graph decode goes to flash_v2.
    decode_attention_impl<128>(Q, K, V, O, LSE, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale, 0);
}

extern "C" __global__ void decode_attention_64_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    decode_attention_impl<64>(Q, K, V, O, LSE, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale, 0);
}

extern "C" __global__ void decode_attention_128_fp32_split(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale, int num_splits
) {
    decode_attention_split_impl<128>(Q, K, V, partial_o, partial_ml, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale, 0, num_splits);
}

extern "C" __global__ void decode_attention_64_fp32_split(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale, int num_splits
) {
    decode_attention_split_impl<64>(Q, K, V, partial_o, partial_ml, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale, 0, num_splits);
}

extern "C" __global__ void decode_attention_128_fp32_combine(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_ml,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_splits
) {
    decode_attention_combine_impl<128>(partial_o, partial_ml, O, LSE, num_splits);
}

extern "C" __global__ void decode_attention_64_fp32_combine(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_ml,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_splits
) {
    decode_attention_combine_impl<64>(partial_o, partial_ml, O, LSE, num_splits);
}

// ============================================================================
// Graph-mode entry points: seq_len_k from device pointer, separate stride
// ============================================================================

extern "C" __global__ void decode_attention_128_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    const int* seq_len_k_ptr,
    int kv_seq_stride,
    float scale,
    int window_size
) {
    decode_attention_impl<128>(Q, K, V, O, LSE, num_heads, num_kv_heads, *seq_len_k_ptr, kv_seq_stride, scale, window_size);
}

extern "C" __global__ void decode_attention_64_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int num_heads, int num_kv_heads,
    const int* seq_len_k_ptr,
    int kv_seq_stride,
    float scale,
    int window_size
) {
    decode_attention_impl<64>(Q, K, V, O, LSE, num_heads, num_kv_heads, *seq_len_k_ptr, kv_seq_stride, scale, window_size);
}
