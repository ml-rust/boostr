// IQ2_S tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ2_S block: 256 elements, 82 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_iq2_s_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 82;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 82;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        const unsigned char* signs_data = block + 38;
        const unsigned char* scales = block + 54;
        unsigned int base = b * 256;

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
    output[row * N + col] = sum;
}
