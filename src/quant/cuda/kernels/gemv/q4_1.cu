// Q4_1 GEMV kernel — F32 activation only (simple 32-element blocks)
//
// Q4_1 block: 32 elements, 20 bytes
// Layout: [d:f16(2), m:f16(2), qs:16B]
// 4-bit unsigned values with min: dequant = d * nibble + m

#include "common.cuh"

#define Q4_1_BLOCK_BYTES 20

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q4_1_f32(
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
    const unsigned int row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q4_1_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float mn = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const unsigned char* qs = block + 4;

        int byte_idx = lane_id >> 1;
        int is_high = lane_id & 1;
        unsigned char byte = qs[byte_idx];
        int nibble = is_high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        float val = d * (float)nibble + mn;
        acc += act_row[b * 32 + lane_id] * val;
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}
