// Q8_K tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q8_K block: 256 elements, 292 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_q8_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 292;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 292;
        float d;
        memcpy(&d, block, sizeof(float));
        const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
        unsigned int base = b * 256;

        for (int i = 0; i < 256; i++) {
            sum += act_row[base + i] * ((float)qs[i] * d);
        }
    }
    output[row * N + col] = sum;
}
