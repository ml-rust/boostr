// PQ2_0 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// PQ2_0 block: 128 elements, 34 bytes
// Layout: [d:f16(2), qs:32B] — one 2-bit code per element, low bits first
// Value: (code - 1) * d, so {-1, 0, 1, 2} * d
// The decode helpers and offsets come from `../prism_dequant.cuh`; this
// file restates neither.

#include "common.cuh"
#include "../prism_dequant.cuh"

#define PQ2_0_BLOCK_ELEMS 128
#define PQ2_0_BLOCK_BYTES 34

extern "C" __global__ void quant_matmul_pq2_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / PQ2_0_BLOCK_ELEMS;
    unsigned int row_bytes = blocks_per_row * PQ2_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * PQ2_0_BLOCK_BYTES;
        const float d = prism_load_d(block);
        const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
        const float* act_blk = act_row + b * PQ2_0_BLOCK_ELEMS;

        for (int e = 0; e < PQ2_0_BLOCK_ELEMS; e++) {
            sum += act_blk[e] * ((float)gguf_code2_minus_1(qs, e) * d);
        }
    }
    output[row * N + col] = sum;
}
