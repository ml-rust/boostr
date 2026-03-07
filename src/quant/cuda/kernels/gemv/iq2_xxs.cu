// IQ2_XXS GEMV kernel — F32 activation path
//
// IQ2_XXS block: 256 elements, 66 bytes
// Layout: [d:f16(2), qs:64B]
// 8 groups of 8 bytes each. q64 = 8-byte load,
// grid_indices = low32, signs_and_scales = high32,
// sub_scale_bits = (signs>>28)&0xF, sub_scale = d*(0.5+sub_scale_bits)
// 4 sub-groups of 8 elements each

#include "common.cuh"

#define IQ2_XXS_BLOCK_BYTES 66
#define IQ2_XXS_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq2_xxs_f32(
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

    unsigned int blocks_per_row = K / IQ2_XXS_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ2_XXS_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * IQ2_XXS_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * IQ2_XXS_BLOCK_SIZE;

        for (int group = 0; group < 8; group++) {
            const unsigned char* gdata = qs + group * 8;
            unsigned long long q64;
            memcpy(&q64, gdata, sizeof(unsigned long long));
            unsigned int grid_indices = (unsigned int)q64;
            unsigned int signs_and_scales = (unsigned int)(q64 >> 32);
            unsigned int sub_scale_bits = (signs_and_scales >> 28) & 0x0F;
            float sub_scale = d * (0.5f + (float)sub_scale_bits);

            for (int sub = 0; sub < 4; sub++) {
                unsigned int grid_idx = (grid_indices >> (8 * sub)) & 0xFF;
                unsigned char sign_bits = (unsigned char)((signs_and_scales >> (7 * sub)) & 0x7F);
                for (int k = 0; k < 8; k++) {
                    int shift = k * 2;
                    int bits;
                    if (shift < 8) bits = (grid_idx >> shift) & 0x03;
                    else           bits = ((grid_idx >> (shift - 8)) ^ (grid_idx >> 1)) & 0x03;
                    float grid_val = (float)bits;
                    float sign = ((sign_bits >> k) & 1) ? -1.0f : 1.0f;
                    sum += act_row[base + group * 32 + sub * 8 + k] * (sub_scale * grid_val * sign);
                }
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
