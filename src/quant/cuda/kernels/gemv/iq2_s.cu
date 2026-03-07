// IQ2_S GEMV kernel — F32 activation path
//
// IQ2_S block: 256 elements, 82 bytes
// Layout: [d:f16(2), qs:36B, signs:16B, scales:28B]
// sub_scale = d*(signed scale + 0.5), 2-bit grid values with signs

#include "common.cuh"

#define IQ2_S_BLOCK_BYTES 82
#define IQ2_S_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq2_s_f32(
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

    unsigned int blocks_per_row = K / IQ2_S_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ2_S_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * IQ2_S_BLOCK_BYTES;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        const unsigned char* signs_data = block + 38;
        const unsigned char* scales = block + 54;
        unsigned int base = b * IQ2_S_BLOCK_SIZE;

        for (int sb = 0; sb < 16; sb++) {
            unsigned char scale_byte = (sb < 28) ? scales[sb] : 0;
            float sub_scale = d * ((float)((signed char)scale_byte) + 0.5f);

            for (int k = 0; k < 16; k++) {
                int byte_idx = sb * 2 + k / 8;
                unsigned char grid_byte = (byte_idx < 32) ? qs[byte_idx] : 0;
                int bit_pos = k % 8;
                unsigned char sign_byte = (sb < 16) ? signs_data[sb] : 0;
                float sign = ((sign_byte >> bit_pos) & 1) ? -1.0f : 1.0f;
                float val = (float)((grid_byte >> ((bit_pos % 4) * 2)) & 0x03);
                sum += act_row[base + sb * 16 + k] * (sub_scale * val * sign);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
