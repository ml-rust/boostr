// TQ1_0 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// TQ1_0 block: 256 elements, 54 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_tq1_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 54;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 54;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 256;

        int idx = 0;
        for (int i = 0; i < 52; i++) {
            unsigned int val = (unsigned int)qs[i];
            for (int j = 0; j < 5; j++) {
                if (idx >= 256) break;
                int t = (int)(val % 3) - 1;
                sum += act_row[base + idx] * (d * (float)t);
                val /= 3;
                idx++;
            }
        }
    }
    output[row * N + col] = sum;
}
