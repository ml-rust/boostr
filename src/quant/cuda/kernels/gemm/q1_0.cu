// Q1_0 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q1_0 block: 128 elements, 18 bytes
// Layout: [d:f16(2), qs:16B] — one sign bit per element, low bit first
// Value: bit set -> +d, clear -> -d
// The decode helpers and offsets come from `../lowbit_dequant.cuh`; this
// file restates neither.

#include "common.cuh"
#include "../lowbit_dequant.cuh"

#define Q1_0_BLOCK_ELEMS 128
#define Q1_0_BLOCK_BYTES 18

extern "C" __global__ void quant_matmul_q1_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / Q1_0_BLOCK_ELEMS;
    unsigned int row_bytes = blocks_per_row * Q1_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q1_0_BLOCK_BYTES;
        const float d = lowbit_load_d(block);
        const unsigned char* qs = block + GGUF_LOWBIT_QS_OFFSET;
        const float* act_blk = act_row + b * Q1_0_BLOCK_ELEMS;

        for (int e = 0; e < Q1_0_BLOCK_ELEMS; e++) {
            sum += act_blk[e] * ((float)gguf_sign_bit(qs, e) * d);
        }
    }
    output[row * N + col] = sum;
}
