// IQ4_XS GEMV kernel — F32 activation path
//
// IQ4_XS block: 256 elements, 136 bytes
// Layout: [d:f16(2), scales_h:1B, scales_l:4B, pad:1B, qs:128B]
// 8 sub-blocks of 32 elements, 6-bit scales, KVALUES_IQ4NL codebook

#include <cuda_fp16.h>

__constant__ signed char KVALUES_IQ4NL_XS[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq4_xs_f32(
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
    unsigned int row_bytes = blocks_per_row * 136;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * 136;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        unsigned char scales_h = block[2];
        const unsigned char* scales_l = block + 3;
        const unsigned char* qs = block + 8;
        unsigned int base = b * 256;

        for (int sb = 0; sb < 8; sb++) {
            unsigned char sl = (sb % 2 == 0) ? (scales_l[sb / 2] & 0x0F) : ((scales_l[sb / 2] >> 4) & 0x0F);
            unsigned char sh = (sb < 4) ? ((scales_h >> (2 * sb)) & 0x03) : 0;
            int scale_6bit = (int)(sl | (sh << 4));
            float sub_scale = d * (float)(scale_6bit - 32);

            const unsigned char* sub_qs = qs + sb * 16;
            for (int i = 0; i < 16; i++) {
                unsigned char byte = sub_qs[i];
                sum += act_row[base + sb * 32 + i * 2]     * sub_scale * (float)KVALUES_IQ4NL_XS[byte & 0x0F];
                sum += act_row[base + sb * 32 + i * 2 + 1] * sub_scale * (float)KVALUES_IQ4NL_XS[(byte >> 4) & 0x0F];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        output[row * N + col] = sum;
}
