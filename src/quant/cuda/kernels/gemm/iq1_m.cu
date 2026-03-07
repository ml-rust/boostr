// IQ1_M tiled GEMM — activation [M,K] × weight [N,K]^T → output [M,N]
// IQ1_M block: 256 elements, 56 bytes

#include <cuda_fp16.h>

extern "C" __global__ void quant_matmul_iq1_m_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 56;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 56;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* scales_data = block + 2;
        const unsigned char* qs = block + 8;
        const unsigned char* qh = block + 40;
        unsigned int base = b * 256;

        for (int sb = 0; sb < 16; sb++) {
            // Extract 3-bit scale
            int scale_bit_offset = sb * 3;
            int byte_idx = scale_bit_offset / 8;
            int bit_offset = scale_bit_offset % 8;
            unsigned int raw;
            if (byte_idx + 1 < 6) {
                unsigned int lo = (unsigned int)scales_data[byte_idx];
                unsigned int hi = (unsigned int)scales_data[byte_idx + 1];
                raw = ((lo | (hi << 8)) >> bit_offset) & 0x07;
            } else if (byte_idx < 6) {
                raw = ((unsigned int)(scales_data[byte_idx] >> bit_offset)) & 0x07;
            } else {
                raw = 0;
            }
            float sub_scale = d * ((float)raw + 0.5f);

            unsigned int qs_val = (unsigned int)qs[sb * 2] | ((unsigned int)qs[sb * 2 + 1] << 8);
            unsigned int grid_idx = qs_val & 0x0FFF;
            unsigned char sign_bits = qh[sb];

            unsigned int grid_val = grid_idx;
            for (int k = 0; k < 16; k++) {
                int t = (int)(grid_val % 3) - 1;
                float sign = ((sign_bits >> (k % 8)) & 1) ? -1.0f : 1.0f;
                sum += act_row[base + sb * 16 + k] * (sub_scale * (float)t * sign);
                grid_val /= 3;
            }
        }
    }
    output[row * N + col] = sum;
}
