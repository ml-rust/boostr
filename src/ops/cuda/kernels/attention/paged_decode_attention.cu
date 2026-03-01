// Paged decode attention kernel — S_q=1 specialized with block_table lookup
//
// Same algorithm as decode_attention.cu but reads K/V from paged blocks
// via block_table indirection. GQA-aware: kv_h = h / (num_heads / num_kv_heads).
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Q layout: [B, num_heads, 1, D]
// Output: [B, num_heads, 1, D]
//
// One block per (batch, Q_head), head_dim threads.

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// head_dim = 128, 128 threads (4 warps), F32
// ============================================================================

extern "C" __global__ void paged_decode_attention_128_fp32(
    const float* __restrict__ Q,            // [B, num_heads, 1, 128]
    const float* __restrict__ K_blocks,     // [num_blocks, block_size, num_kv_heads, 128]
    const float* __restrict__ V_blocks,     // [num_blocks, block_size, num_kv_heads, 128]
    const int* __restrict__ block_table,    // [B, max_num_blocks]
    float* __restrict__ O,                  // [B, num_heads, 1, 128]
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale
) {
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh % num_heads;
    const int kv_h = h / (num_heads / num_kv_heads);

    const int tid = threadIdx.x;  // 0..127 = head_dim element
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Load Q[b, h, 0, :] into register, pre-scaled
    const float* q_row = Q + (size_t)(b * num_heads + h) * 128;
    const float q_val = q_row[tid] * scale;

    // Block table for this batch
    const int* bt = block_table + b * max_num_blocks;

    // Online softmax state
    float acc = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    __shared__ float smem_qk[4];

    // Iterate over all K/V positions using block table
    const int num_kv_blocks = (seq_len_k + block_size - 1) / block_size;
    for (int blk = 0; blk < num_kv_blocks; blk++) {
        const int physical_block = bt[blk];
        const int tokens_in_block = min(block_size, seq_len_k - blk * block_size);

        // Base offset for this physical block + KV head
        // Layout: [num_blocks, block_size, num_kv_heads, head_dim]
        const size_t block_base = (size_t)physical_block * block_size * num_kv_heads * 128
                                + (size_t)kv_h * 128;
        const int kv_stride = num_kv_heads * 128;  // stride between tokens within a block

        for (int off = 0; off < tokens_in_block; off++) {
            // K[physical_block, off, kv_h, tid]
            float k_val = K_blocks[block_base + (size_t)off * kv_stride + tid];
            float qk = q_val * k_val;

            // Warp-level reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                qk += __shfl_down_sync(0xFFFFFFFF, qk, offset);

            if (lane_id == 0) smem_qk[warp_id] = qk;
            __syncthreads();

            float dot = smem_qk[0] + smem_qk[1] + smem_qk[2] + smem_qk[3];
            __syncthreads();

            // Online softmax update
            float m_new = fmaxf(m, dot);
            float exp_old = expf(m - m_new);
            float exp_new = expf(dot - m_new);

            float v_val = V_blocks[block_base + (size_t)off * kv_stride + tid];
            acc = acc * exp_old + v_val * exp_new;
            l = l * exp_old + exp_new;
            m = m_new;
        }
    }

    // Write output
    float* o_row = O + (size_t)(b * num_heads + h) * 128;
    o_row[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ============================================================================
// head_dim = 64, 64 threads (2 warps), F32
// ============================================================================

extern "C" __global__ void paged_decode_attention_64_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K_blocks,
    const float* __restrict__ V_blocks,
    const int* __restrict__ block_table,
    float* __restrict__ O,
    int num_heads, int num_kv_heads,
    int seq_len_k, int max_num_blocks,
    int block_size, float scale
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

    const int* bt = block_table + b * max_num_blocks;

    float acc = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    __shared__ float smem_qk[2];

    const int num_kv_blocks = (seq_len_k + block_size - 1) / block_size;
    for (int blk = 0; blk < num_kv_blocks; blk++) {
        const int physical_block = bt[blk];
        const int tokens_in_block = min(block_size, seq_len_k - blk * block_size);

        const size_t block_base = (size_t)physical_block * block_size * num_kv_heads * 64
                                + (size_t)kv_h * 64;
        const int kv_stride = num_kv_heads * 64;

        for (int off = 0; off < tokens_in_block; off++) {
            float k_val = K_blocks[block_base + (size_t)off * kv_stride + tid];
            float qk = q_val * k_val;

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

            float v_val = V_blocks[block_base + (size_t)off * kv_stride + tid];
            acc = acc * exp_old + v_val * exp_new;
            l = l * exp_old + exp_new;
            m = m_new;
        }
    }

    float* o_row = O + (size_t)(b * num_heads + h) * 64;
    o_row[tid] = (l > 0.0f) ? acc / l : 0.0f;
}
