// IQ4_NL GEMV kernel — F32 activation path
//
// IQ4_NL block: 32 elements, 18 bytes
// Layout: [d:f16(2), qs:16B]
// Non-linear codebook: value = KVALUES_IQ4NL[nibble]
// dequant(i) = d * KVALUES_IQ4NL[nibble]

#include <cuda_fp16.h>

__constant__ signed char KVALUES_IQ4NL_GEMV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq4_nl_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    unsigned int row = blockIdx.y;
    if (col >= N || row >= M) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 18;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * 18;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 32;

        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            sum += act_row[base + i * 2]     * d * (float)KVALUES_IQ4NL_GEMV[byte & 0x0F];
            sum += act_row[base + i * 2 + 1] * d * (float)KVALUES_IQ4NL_GEMV[(byte >> 4) & 0x0F];
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        output[row * N + col] = sum;
}
