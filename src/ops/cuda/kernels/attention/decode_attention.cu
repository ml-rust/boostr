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

// ============================================================================
// head_dim = 128, 128 threads (4 warps), F32
// ============================================================================

extern "C" __global__ void decode_attention_128_fp32(
    const float* __restrict__ Q,    // [B, num_heads, 1, 128]
    const float* __restrict__ K,    // [B, num_kv_heads, seq_k, 128]
    const float* __restrict__ V,    // [B, num_kv_heads, seq_k, 128]
    float* __restrict__ O,          // [B, num_heads, 1, 128]
    int num_heads, int num_kv_heads,
    int seq_len_k, float scale
) {
    // One block per (batch, head) pair
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);

    const int tid = threadIdx.x;  // 0..127 = head_dim element
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Load Q[b, h, 0, :] into register (one element per thread), pre-scaled
    const float* q_row = Q + (size_t)(b * num_heads + h) * 128;
    const float q_val = q_row[tid] * scale;

    // K/V base for this KV head
    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * seq_len_k * 128;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * seq_len_k * 128;

    // Online softmax state (per-thread V accumulator, shared softmax scalars)
    float acc = 0.0f;  // V accumulator for this thread's head_dim element
    float m = -1e30f;   // running max of Q·K
    float l = 0.0f;     // running sum of exp(Q·K - m)

    // Shared memory for cross-warp Q·K reduction (4 warps)
    __shared__ float smem_qk[4];

    for (int pos = 0; pos < seq_len_k; pos++) {
        // Q·K dot product: each thread has one element
        float qk = q_val * k_base[pos * 128 + tid];

        // Warp-level reduction (within 32 lanes)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            qk += __shfl_down_sync(0xFFFFFFFF, qk, offset);

        // Cross-warp reduction via shared memory
        if (lane_id == 0) smem_qk[warp_id] = qk;
        __syncthreads();

        // Sum the 4 warp partials (all threads read, avoid divergence)
        float dot = smem_qk[0] + smem_qk[1] + smem_qk[2] + smem_qk[3];
        __syncthreads();

        // Online softmax update
        float m_new = fmaxf(m, dot);
        float exp_old = expf(m - m_new);
        float exp_new = expf(dot - m_new);

        // Rescale running accumulator and add new V contribution
        acc = acc * exp_old + v_base[pos * 128 + tid] * exp_new;
        l = l * exp_old + exp_new;
        m = m_new;
    }

    // Write output: O[b, h, 0, tid] = acc / l
    float* o_row = O + (size_t)(b * num_heads + h) * 128;
    o_row[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ============================================================================
// head_dim = 64, 64 threads (2 warps), F32
// ============================================================================

extern "C" __global__ void decode_attention_64_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, float scale
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

    const float* k_base = K + (size_t)(b * num_kv_heads + kv_h) * seq_len_k * 64;
    const float* v_base = V + (size_t)(b * num_kv_heads + kv_h) * seq_len_k * 64;

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
