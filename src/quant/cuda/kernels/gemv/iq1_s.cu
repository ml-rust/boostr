// IQ1_S GEMV kernel — F32 activation path
//
// IQ1_S block: 256 elements, 50 bytes
// Layout: [d:f16(2), qs:32B, qh:16B]
// 16 sub-blocks of 16 elements each.
// qs_val = 16-bit from qs, grid_idx = qs_val & 0xFFF
// sign_bits = qh[sb]
// Base-3 decode: val = (grid_val % 3) - 1, grid_val /= 3

#include "common.cuh"

#define IQ1_S_BLOCK_BYTES 50
#define IQ1_S_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq1_s_f32(
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

    unsigned int blocks_per_row = K / IQ1_S_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ1_S_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * IQ1_S_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        const unsigned char* qh = block + 34;
        unsigned int base = b * IQ1_S_BLOCK_SIZE;

        for (int sb = 0; sb < 16; sb++) {
            unsigned int qs_val = (unsigned int)qs[sb * 2] | ((unsigned int)qs[sb * 2 + 1] << 8);
            unsigned int grid_idx = qs_val & 0x0FFF;
            unsigned char sign_bits = qh[sb];

            unsigned int grid_val = grid_idx;
            for (int k = 0; k < 16; k++) {
                int t = (int)(grid_val % 3) - 1;
                float sign = ((sign_bits >> (k % 8)) & 1) ? -1.0f : 1.0f;
                sum += act_row[base + sb * 16 + k] * (d * (float)t * sign);
                grid_val /= 3;
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
