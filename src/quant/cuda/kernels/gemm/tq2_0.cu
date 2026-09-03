// TQ2_0 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// TQ2_0 block: 256 elements, 66 bytes

#include <cuda_fp16.h>

#include "../decode.cuh"

extern "C" __global__ void quant_matmul_tq2_0_f32(
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
        memcpy(&d_half, block + GGUF_TQ2_0_D_OFFSET, sizeof(__half));
        float d = __half2float(d_half);
        unsigned int base = b * 256;

        for (int i = 0; i < 256; i++) {
            sum += act_row[base + i] * (d * (float)gguf_tq2_0_trit(block, i));
        }
    }
    output[row * N + col] = sum;
}
