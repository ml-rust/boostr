// AWQ INT4 GEMM: input[M,K] x dequant(qweight[K,N/8]) -> output[M,N]
// AWQ packing: 8 INT4 per u32, shifts [0,16,4,20,8,24,12,28]
// Dequant: w = (q - zero) * scale

#include <stdint.h>

__constant__ int AWQ_SHIFTS[8] = {0, 16, 4, 20, 8, 24, 12, 28};

__device__ __forceinline__ int unpack_int4_awq(uint32_t packed, int idx) {
    return (packed >> AWQ_SHIFTS[idx]) & 0xF;
}

// Simple tiled GEMM: each thread computes one output element
// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void int4_gemm_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const float* __restrict__ zeros,      // [num_groups, N]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t group_size
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    uint32_t n_packed = N / 8;
    uint32_t pack_col = col / 8;
    uint32_t sub = col % 8;

    float acc = 0.0f;
    for (uint32_t ki = 0; ki < K; ki++) {
        float a = input[row * K + ki];
        uint32_t packed = qweight[ki * n_packed + pack_col];
        int q = unpack_int4_awq(packed, sub);

        uint32_t group = ki / group_size;
        float scale = scales[group * N + col];
        float zero = zeros[group * N + col];
        float w = ((float)q - zero) * scale;
        acc += a * w;
    }
    output[row * N + col] = acc;
}
