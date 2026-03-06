// IQ3_S tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ3_S block: 256 elements, 110 bytes
// Layout: [d:f16(2), qs:32B, qh:4B, signs:32B, scales:8B]
// 8 sub-blocks of 32, 3-bit grid values with sign bits

#include <cuda_fp16.h>

#define TILE 16

extern "C" __global__ void quant_matmul_iq3_s_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * TILE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 110;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 110;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        const unsigned char* qh = block + 34;
        const unsigned char* signs = block + 38;
        const unsigned char* scales = block + 70;
        unsigned int base = b * 256;

        for (int sb = 0; sb < 8; sb++) {
            float sub_scale = d * (1.0f + (float)(scales[sb] & 0x0F));

            for (int k = 0; k < 32; k++) {
                int byte_idx = sb * 4 + k / 8;
                int q3 = (byte_idx < 32) ? ((qs[byte_idx] >> ((k % 8) / 2 * 2)) & 0x03) : 0;
                int qh_byte_idx = (sb * 32 + k) / 8;
                int qh_bit = (qh_byte_idx < 4) ? ((qh[qh_byte_idx] >> ((sb * 32 + k) % 8)) & 1) : 0;
                float val = (float)q3 + (float)qh_bit * 4.0f + 1.0f;

                int sign_byte_idx = sb * 4 + k / 8;
                int sign_bit = (sign_byte_idx < 32) ? ((signs[sign_byte_idx] >> (k % 8)) & 1) : 0;
                float sign = sign_bit ? -1.0f : 1.0f;

                sum += act_row[base + sb * 32 + k] * sub_scale * val * sign;
            }
        }
    }
    output[row * N + col] = sum;
}
