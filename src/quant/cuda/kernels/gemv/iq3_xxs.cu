// IQ3_XXS GEMV kernel — F32 activation path
//
// IQ3_XXS block: 256 elements, 98 bytes
// Layout: [d:f16(2), qs:96B]
// 8 groups of 12 bytes. gdata+8 has signs u32,
// sub_scale_bits = (signs>>28)&0xF, sub_scale = d*(1+sub_scale_bits)
// Grid: 16-bit index, 2-bit values + 1, sign from sign bits

#include "common.cuh"

#define IQ3_XXS_BLOCK_BYTES 98
#define IQ3_XXS_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq3_xxs_f32(
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

    unsigned int blocks_per_row = K / IQ3_XXS_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ3_XXS_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * IQ3_XXS_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * IQ3_XXS_BLOCK_SIZE;

        for (int group = 0; group < 8; group++) {
            const unsigned char* gdata = qs + group * 12;
            unsigned int signs;
            memcpy(&signs, gdata + 8, sizeof(unsigned int));
            unsigned int sub_scale_bits = (signs >> 28) & 0x0F;
            float sub_scale = d * (1.0f + (float)sub_scale_bits);

            for (int sub = 0; sub < 4; sub++) {
                unsigned int grid_idx = (unsigned int)gdata[sub * 2] |
                                        ((unsigned int)gdata[sub * 2 + 1] << 8);
                for (int k = 0; k < 8; k++) {
                    float val = (float)((grid_idx >> (k * 2)) & 0x03) + 1.0f;
                    unsigned int sign_bit = (signs >> (sub * 8 + k)) & 1;
                    float sign = sign_bit ? -1.0f : 1.0f;
                    sum += act_row[base + group * 32 + sub * 8 + k] * (sub_scale * val * sign);
                }
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
