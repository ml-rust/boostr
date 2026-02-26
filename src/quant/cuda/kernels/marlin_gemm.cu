// Marlin-format INT4 GEMM: sequential 4-bit packing
// Dequant: w = (q - 8) * scale + zero

#include <stdint.h>

__device__ __forceinline__ int unpack_int4_seq(uint32_t packed, int idx) {
    return (packed >> (idx * 4)) & 0xF;
}

// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void marlin_gemm_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ weight,  // [K/8, N]
    const float* __restrict__ scales,     // [num_groups, N]
    const float* __restrict__ zeros,      // [num_groups, N]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t group_size
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    uint32_t k_packed = K / 8;

    float acc = 0.0f;
    for (uint32_t pack_ki = 0; pack_ki < k_packed; pack_ki++) {
        uint32_t packed = weight[pack_ki * N + col];

        for (int sub = 0; sub < 8; sub++) {
            uint32_t ki = pack_ki * 8 + sub;
            float a = input[row * K + ki];
            int q = unpack_int4_seq(packed, sub);

            uint32_t group = ki / group_size;
            float scale = scales[group * N + col];
            float zero = zeros[group * N + col];
            float w = ((float)q - 8.0f) * scale + zero;
            acc += a * w;
        }
    }
    output[row * N + col] = acc;
}
