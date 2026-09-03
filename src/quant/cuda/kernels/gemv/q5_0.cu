// Q5_0 GEMV kernel — F32 activation only (no dp4a, simple 32-element blocks)
//
// Q5_0 block: 32 elements, 22 bytes
// Layout: [d:f16(2), qh:u32(4), qs:16B]
// 5-bit values: 4-bit low nibble + 1-bit from qh

#include "common.cuh"

#define Q5_0_BLOCK_BYTES 22

// ============================================================================
// Q5_0 GEMV (F32 activation) — warp-per-column
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q5_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned int row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q5_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        // The block stride is 22 bytes, so `block + 2` is only 2-byte aligned on
        // odd blocks: a 4-byte load through `unsigned int*` traps with
        // CUDA_ERROR_MISALIGNED_ADDRESS and poisons the context for every later
        // launch. memcpy is the portable unaligned load.
        unsigned int qh;
        memcpy(&qh, block + 2, sizeof(unsigned int));
        const unsigned char* qs = block + 6;

        // Each lane handles one of 32 elements. Split-half nibble order: lane
        // `l` takes the LOW nibble of qs[l] for l < 16, the HIGH nibble of
        // qs[l - 16] for l >= 16 (llama.cpp dequantize_row_q5_0). The fifth bit
        // is `qh` bit `l`, which already matches the element index.
        int low4 = gguf_split_half_nibble(qs, (int)lane_id);
        int high1 = (qh >> lane_id) & 1;
        int val = (low4 | (high1 << 4)) - 16;
        acc += act_row[b * 32 + lane_id] * ((float)val * d);
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}
