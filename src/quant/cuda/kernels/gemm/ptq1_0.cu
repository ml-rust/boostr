// PTQ1_0 tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// PTQ1_0 block: 128 elements, 28 bytes.
// Layout: qs[0..24], qh[24..26], d:f16[26..28] — the scale is at the END.
// The decode helpers and offsets come from `../lowbit_dequant.cuh`; this
// file restates neither.

#include "../lowbit_dequant.cuh"

#define PTQ1_0_BLOCK_ELEMS 128
#define PTQ1_0_BLOCK_BYTES 28

extern "C" __global__ void quant_matmul_ptq1_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / PTQ1_0_BLOCK_ELEMS;
    unsigned int row_bytes = blocks_per_row * PTQ1_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * PTQ1_0_BLOCK_BYTES;
        const float d = lowbit_load_d(block + GGUF_PTQ1_0_D_OFFSET);
        unsigned int base = b * PTQ1_0_BLOCK_ELEMS;

        for (int i = 0; i < PTQ1_0_BLOCK_ELEMS; i++) {
            sum += act_row[base + i] * (d * (float)gguf_ptq1_0_trit(block, i));
        }
    }
    output[row * N + col] = sum;
}
