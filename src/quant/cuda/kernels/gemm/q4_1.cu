// Q4_1 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q4_1 block: 32 elements, 20 bytes

#include "common.cuh"

extern "C" __global__ void quant_matmul_q4_1_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 20;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 20;
        __half d_half, m_half;
        memcpy(&d_half, block, sizeof(__half));
        memcpy(&m_half, block + 2, sizeof(__half));
        float d = __half2float(d_half);
        float m = __half2float(m_half);
        const unsigned char* qs = block + 4;
        unsigned int base = b * 32;

        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            sum += act_row[base + i * 2]     * (d * (float)(byte & 0x0F) + m);
            sum += act_row[base + i * 2 + 1] * (d * (float)((byte >> 4) & 0x0F) + m);
        }
    }
    output[row * N + col] = sum;
}
