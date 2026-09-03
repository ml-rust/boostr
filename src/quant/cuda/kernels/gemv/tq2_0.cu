// TQ2_0 GEMV kernel — F32 activation path
//
// TQ2_0 block: 256 elements, 66 bytes
// Layout: [qs:64B, d:f16(2)] — the scale is at the END. Element order and
// the trit unpack both live in decode.cuh, included via common.cuh.

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
        memcpy(&d_half, block + GGUF_TQ2_0_D_OFFSET, sizeof(__half));
        float d = __half2float(d_half);
        unsigned int base = b * TQ2_0_BLOCK_SIZE;

        for (int i = 0; i < TQ2_0_BLOCK_SIZE; i++) {
            sum += act_row[base + i] * (d * (float)gguf_tq2_0_trit(block, i));
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
