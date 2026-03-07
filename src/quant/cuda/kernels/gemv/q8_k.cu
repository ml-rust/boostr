// Q8_K GEMV kernel — F32 activation only
//
// Q8_K block: 256 elements, 292 bytes
// Layout: [d:f32(4), qs:256B signed, bsums:32B(ignored)]
// dequant = d * qs[i]

#include "common.cuh"

#define Q8_K_BLOCK_BYTES 292
#define Q8_K_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q8_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    unsigned int row = blockIdx.y;
    if (col >= N || row >= M) return;

    unsigned int blocks_per_row = K / Q8_K_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * Q8_K_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * Q8_K_BLOCK_BYTES;
        float d;
        memcpy(&d, block, sizeof(float));
        const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
        unsigned int base = b * Q8_K_BLOCK_SIZE;

        for (int i = 0; i < Q8_K_BLOCK_SIZE; i++) {
            sum += act_row[base + i] * (d * (float)qs[i]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
