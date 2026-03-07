// Q8_1 GEMV kernel — F32 activation only (simple 32-element blocks)
//
// Q8_1 block: 32 elements, 36 bytes
// Layout: [d:f16(2), s:f16(2), qs:32B signed]
// dequant = d * qs[i] + s

#include "common.cuh"

#define Q8_1_BLOCK_BYTES 36

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q8_1_f32(
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
    const unsigned int row_bytes = blocks_per_row * Q8_1_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q8_1_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float s = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const signed char* qs = reinterpret_cast<const signed char*>(block + 4);

        float val = d * (float)qs[lane_id] + s;
        acc += act_row[b * 32 + lane_id] * val;
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}
