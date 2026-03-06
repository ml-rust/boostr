// IQ4_NL tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ4_NL block: 32 elements, 18 bytes
// Non-linear codebook: value = KVALUES_IQ4NL[nibble]

#include <cuda_fp16.h>

__constant__ signed char KVALUES_IQ4NL_GEMM[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

extern "C" __global__ void quant_matmul_iq4_nl_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 18;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 18;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 32;

        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            sum += act_row[base + i * 2]     * d * (float)KVALUES_IQ4NL_GEMM[byte & 0x0F];
            sum += act_row[base + i * 2 + 1] * d * (float)KVALUES_IQ4NL_GEMM[(byte >> 4) & 0x0F];
        }
    }
    output[row * N + col] = sum;
}
