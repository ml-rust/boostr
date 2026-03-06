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
        unsigned int qh = *reinterpret_cast<const unsigned int*>(block + 2);
        const unsigned char* qs = block + 6;

        // Each lane handles one of 32 elements
        int byte_idx = lane_id >> 1;
        int is_high = lane_id & 1;
        unsigned char byte = qs[byte_idx];
        int low4 = is_high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        int high1 = (qh >> lane_id) & 1;
        int val = (low4 | (high1 << 4)) - 16;
        acc += act_row[b * 32 + lane_id] * ((float)val * d);
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}
