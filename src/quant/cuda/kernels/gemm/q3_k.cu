// Q3_K tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q3_K block: 256 elements, 110 bytes

#include "common.cuh"

extern "C" __global__ void quant_matmul_q3_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 110;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 110;
        const unsigned char* hmask = block;
        const unsigned char* qs = block + 32;
        const unsigned char* sc_raw = block + 96;
        float d = load_f16_as_f32_gemm(block + 108);
        unsigned int base = b * 256;

        signed char scales[16];
        unpack_q3k_scales_gemm(sc_raw, scales);

        int y = 0, is = 0;
        unsigned char mask = 1;
        for (int n_half = 0; n_half < 2; n_half++) {
            const unsigned char* q = qs + n_half * 32;
            for (int shift = 0; shift < 8; shift += 2) {
                float dl = d * (float)scales[is++];
                for (int l = 0; l < 16; l++) {
                    int low2 = (q[l] >> shift) & 3;
                    int hsub = (hmask[l] & mask) ? 0 : 4;
                    sum += act_row[base + y] * dl * (float)(low2 - hsub);
                    y++;
                }
                dl = d * (float)scales[is++];
                for (int l = 0; l < 16; l++) {
                    int low2 = (q[16 + l] >> shift) & 3;
                    int hsub = (hmask[16 + l] & mask) ? 0 : 4;
                    sum += act_row[base + y] * dl * (float)(low2 - hsub);
                    y++;
                }
                mask <<= 1;
            }
        }
    }
    output[row * N + col] = sum;
}
