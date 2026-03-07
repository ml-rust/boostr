// Q8_1 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q8_1 block: 32 elements, 36 bytes

#include "common.cuh"

extern "C" __global__ void quant_matmul_q8_1_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 36;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 36;
        __half d_half, s_half;
        memcpy(&d_half, block, sizeof(__half));
        memcpy(&s_half, block + 2, sizeof(__half));
        float d = __half2float(d_half);
        float s = __half2float(s_half);
        const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
        unsigned int base = b * 32;

        for (int i = 0; i < 32; i++) {
            sum += act_row[base + i] * ((float)qs[i] * d + s);
        }
    }
    output[row * N + col] = sum;
}
