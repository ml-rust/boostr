// IQ2_XXS tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ2_XXS block: 256 elements, 66 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_iq2_xxs_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 66;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 66;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 256;

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
    output[row * N + col] = sum;
}
