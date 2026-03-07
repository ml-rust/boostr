// IQ3_XXS tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ3_XXS block: 256 elements, 98 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_iq3_xxs_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 98;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 98;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 256;

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
    output[row * N + col] = sum;
}
