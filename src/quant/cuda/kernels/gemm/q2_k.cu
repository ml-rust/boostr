// Q2_K tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q2_K block: 256 elements, 84 bytes

#include "common.cuh"

extern "C" __global__ void quant_matmul_q2_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 84;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 84;
        const unsigned char* sc = block;
        const unsigned char* qs = block + 16;
        float d = load_f16_as_f32_gemm(block + 80);
        float dmin = load_f16_as_f32_gemm(block + 82);
        unsigned int base = b * 256;

        int y = 0, is = 0;
        for (int n_half = 0; n_half < 2; n_half++) {
            const unsigned char* q = qs + n_half * 32;
            for (int shift = 0; shift < 8; shift += 2) {
                float dl = d * (float)(sc[is] & 0x0F);
                float ml = dmin * (float)(sc[is] >> 4);
                is++;
                for (int l = 0; l < 16; l++) {
                    sum += act_row[base + y] * (dl * (float)((q[l] >> shift) & 3) - ml);
                    y++;
                }
                dl = d * (float)(sc[is] & 0x0F);
                ml = dmin * (float)(sc[is] >> 4);
                is++;
                for (int l = 0; l < 16; l++) {
                    sum += act_row[base + y] * (dl * (float)((q[16 + l] >> shift) & 3) - ml);
                    y++;
                }
            }
        }
    }
    output[row * N + col] = sum;
}
