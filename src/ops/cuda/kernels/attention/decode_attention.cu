// Decode attention kernel — optimized for S_q=1 (autoregressive decoding)
//
// Inspired by llama.cpp fattn-vec and vLLM's paged_attention.
// For contiguous KV cache (no paging). One block per (batch, head) pair.
//
// Algorithm: load Q into registers, iterate over all K positions,
// compute Q·K via warp-shuffle reduction, online softmax, accumulate V.
//
// Layout: Q [B, num_heads, 1, D], K/V [B, num_kv_heads, seq_k, D]
// Output: O [B, num_heads, 1, D]
//
// Two variants per head_dim:
//   - Non-graph: seq_len_k passed as plain int kernel arg (zero overhead)
//   - Graph-mode (_graph suffix): seq_len_k_ptr is a device pointer to i32,
//     kv_seq_stride is the memory stride (capacity >= seq_len_k)

// ============================================================================
// Shared device function for the attention loop
// ============================================================================

__device__ __forceinline__ void decode_attention_128_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const float* q_row = Q + (size_t)(b * num_heads + h) * 128;
    const float q_val = q_row[tid] * scale;

    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * 128;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * 128;

    float acc = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    __shared__ float smem_qk[4];

    for (int pos = 0; pos < seq_len_k; pos++) {
        float qk = q_val * k_base[pos * 128 + tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            qk += __shfl_down_sync(0xFFFFFFFF, qk, offset);

        if (lane_id == 0) smem_qk[warp_id] = qk;
        __syncthreads();

        float dot = smem_qk[0] + smem_qk[1] + smem_qk[2] + smem_qk[3];
        __syncthreads();

        float m_new = fmaxf(m, dot);
        float exp_old = expf(m - m_new);
        float exp_new = expf(dot - m_new);

        acc = acc * exp_old + v_base[pos * 128 + tid] * exp_new;
        l = l * exp_old + exp_new;
        m = m_new;
    }

    float* o_row = O + (size_t)(b * num_heads + h) * 128;
    o_row[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

__device__ __forceinline__ void decode_attention_64_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const float* q_row = Q + (size_t)(b * num_heads + h) * 64;
    const float q_val = q_row[tid] * scale;

    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * 64;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * kv_seq_stride * 64;

    float acc = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    __shared__ float smem_qk[2];

    for (int pos = 0; pos < seq_len_k; pos++) {
        float qk = q_val * k_base[pos * 64 + tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            qk += __shfl_down_sync(0xFFFFFFFF, qk, offset);

        if (lane_id == 0) smem_qk[warp_id] = qk;
        __syncthreads();

        float dot = smem_qk[0] + smem_qk[1];
        __syncthreads();

        float m_new = fmaxf(m, dot);
        float exp_old = expf(m - m_new);
        float exp_new = expf(dot - m_new);

        acc = acc * exp_old + v_base[pos * 64 + tid] * exp_new;
        l = l * exp_old + exp_new;
        m = m_new;
    }

    float* o_row = O + (size_t)(b * num_heads + h) * 64;
    o_row[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ============================================================================
// Non-graph entry points: seq_len_k as plain int (zero overhead)
// ============================================================================

extern "C" __global__ void decode_attention_128_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    decode_attention_128_impl(Q, K, V, O, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale);
}

extern "C" __global__ void decode_attention_64_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, int kv_seq_stride,
    float scale
) {
    decode_attention_64_impl(Q, K, V, O, num_heads, num_kv_heads, seq_len_k, kv_seq_stride, scale);
}

// ============================================================================
// Graph-mode entry points: seq_len_k from device pointer, separate stride
// ============================================================================

extern "C" __global__ void decode_attention_128_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    const int* seq_len_k_ptr,
    int kv_seq_stride,
    float scale
) {
    decode_attention_128_impl(Q, K, V, O, num_heads, num_kv_heads, *seq_len_k_ptr, kv_seq_stride, scale);
}

extern "C" __global__ void decode_attention_64_fp32_graph(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    const int* seq_len_k_ptr,
    int kv_seq_stride,
    float scale
) {
    decode_attention_64_impl(Q, K, V, O, num_heads, num_kv_heads, *seq_len_k_ptr, kv_seq_stride, scale);
}
