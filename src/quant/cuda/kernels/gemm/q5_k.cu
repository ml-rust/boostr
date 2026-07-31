// Q5_K tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// Q5_K block: 256 elements, 176 bytes

#include "common.cuh"

extern "C" __global__ void quant_matmul_q5_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 176;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 176;
        float d = load_f16_as_f32_gemm(block);
        float dmin = load_f16_as_f32_gemm(block + 2);
        const unsigned char* sc = block + 4;
        const unsigned char* qh = block + 16;
        const unsigned char* qs = block + 48;
        unsigned int base = b * 256;

        unsigned char scales[8], mins[8];
        unpack_q4k_q5k_scales_gemm(sc, scales, mins);

        for (int j = 0; j < 8; j++) {
            float dl = d * (float)scales[j];
            float ml = dmin * (float)mins[j];
            // Sub-block PAIRS share one 32-byte run of qs (even sub-block takes
            // the low nibbles, odd the high nibbles of the SAME bytes), and qh
            // is indexed by ELEMENT with the BIT selected by sub-block index.
            // Must match the CPU `dequant_q5k` / llama.cpp exactly.
            int qs_base = (j / 2) * 32;
            int is_high_nibble = j % 2;
            for (int l = 0; l < 32; l++) {
                int low4 = is_high_nibble ? ((qs[qs_base + l] >> 4) & 0x0F)
                                          : (qs[qs_base + l] & 0x0F);
                int high1 = (qh[l] >> j) & 0x01;
                float q = (float)(low4 | (high1 << 4));
                sum += act_row[base + j * 32 + l] * (dl * q - ml);
            }
        }
    }
    output[row * N + col] = sum;
}
