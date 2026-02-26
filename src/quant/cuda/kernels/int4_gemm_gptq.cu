// GPTQ INT4 GEMM with g_idx permutation
// GPTQ INT4 GEMM: input[M,K] x dequant(qweight[K/8,N]) -> output[M,N]
// Sequential 4-bit packing, g_idx permutation
// Dequant: w = (q - qzero) * scale

#include <stdint.h>

__device__ __forceinline__ int unpack_int4_seq(uint32_t packed, int idx) {
    return (packed >> (idx * 4)) & 0xF;
}

// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void int4_gemm_gptq_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const uint32_t* __restrict__ qzeros,  // [num_groups, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const int* __restrict__ g_idx,        // [K]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    uint32_t n_packed_zeros = N / 8;

    float acc = 0.0f;
    for (uint32_t ki = 0; ki < K; ki++) {
        float a = input[row * K + ki];

        uint32_t pack_row = ki / 8;
        uint32_t sub_idx = ki % 8;
        uint32_t packed = qweight[pack_row * N + col];
        int q = unpack_int4_seq(packed, sub_idx);

        int group = g_idx[ki];
        uint32_t zero_pack = qzeros[group * n_packed_zeros + col / 8];
        int qzero = unpack_int4_seq(zero_pack, col % 8);

        float scale = scales[group * N + col];
        float w = ((float)q - (float)qzero) * scale;
        acc += a * w;
    }
    output[row * N + col] = acc;
}
