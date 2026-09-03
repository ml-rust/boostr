// TQ1_0 GEMV kernel — F32 activation path
//
// TQ1_0 block: 256 elements, 54 bytes
// Layout: [qs:48B, qh:4B, d:f16(2)] — the scale is at the END.
// Base-3 encoding: five trits per byte, recovered by a WRAPPING 8-bit multiply
// against a power of three, not by repeated division. Element order and the
// unpack itself live in decode.cuh, included via common.cuh.

#include "common.cuh"

#define TQ1_0_BLOCK_BYTES 54
#define TQ1_0_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_tq1_0_f32(
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

    unsigned int blocks_per_row = K / TQ1_0_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * TQ1_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * TQ1_0_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block + GGUF_TQ1_0_D_OFFSET, sizeof(__half));
        float d = __half2float(d_half);
        unsigned int base = b * TQ1_0_BLOCK_SIZE;

        for (int i = 0; i < TQ1_0_BLOCK_SIZE; i++) {
            sum += act_row[base + i] * (d * (float)gguf_tq1_0_trit(block, i));
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
