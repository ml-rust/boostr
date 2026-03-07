// TQ2_0 GEMV kernel — F32 activation path
//
// TQ2_0 block: 256 elements, 66 bytes
// Layout: [d:f16(2), qs:64B]
// Each byte has 4 x 2-bit values: (byte >> (2*j)) & 3 - 1, result * d

#include "common.cuh"

#define TQ2_0_BLOCK_BYTES 66
#define TQ2_0_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_tq2_0_f32(
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

    unsigned int blocks_per_row = K / TQ2_0_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * TQ2_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * TQ2_0_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * TQ2_0_BLOCK_SIZE;

        for (int i = 0; i < 64; i++) {
            unsigned char byte = qs[i];
            for (int j = 0; j < 4; j++) {
                int val = ((byte >> (2 * j)) & 0x03) - 1;
                sum += act_row[base + i * 4 + j] * (d * (float)val);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
