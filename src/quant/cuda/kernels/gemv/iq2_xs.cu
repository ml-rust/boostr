// IQ2_XS GEMV kernel — F32 activation path
//
// IQ2_XS block: 256 elements, 74 bytes
// Layout: [d:f16(2), scales:16B, qs:56B]
// 16 sub-blocks of 16 values

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq2_xs_f32(
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

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 74;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * 74;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* sc = block + 2;
        const unsigned char* qs = block + 18;
        unsigned int base = b * 256;

        for (int sb = 0; sb < 16; sb++) {
            float scale = d * ((float)((signed char)sc[sb]) + 0.5f);
            unsigned int q_offset = sb * 2;
            unsigned int q_val = (unsigned int)qs[q_offset] | ((unsigned int)qs[q_offset + 1] << 8);

            for (int k = 0; k < 16; k++) {
                int val_2bit = (q_val >> (k % 8 * 2)) & 0x03;
                float magnitude = (float)val_2bit + 0.5f;
                float sign = ((q_val >> (8 + k)) & 1) ? -1.0f : 1.0f;
                sum += act_row[base + sb * 16 + k] * scale * magnitude * sign;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        output[row * N + col] = sum;
}
