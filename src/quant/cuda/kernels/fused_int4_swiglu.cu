// Fused INT4 dual-GEMM + SwiGLU: silu(input @ gate_w) * (input @ up_w)
// Both weights in AWQ INT4 format

#include <stdint.h>
#include <math.h>

__constant__ int SWIGLU_AWQ_SHIFTS[8] = {0, 16, 4, 20, 8, 24, 12, 28};

__device__ __forceinline__ int unpack_int4_awq(uint32_t packed, int idx) {
    return (packed >> SWIGLU_AWQ_SHIFTS[idx]) & 0xF;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void fused_int4_swiglu_f32(
    const float* __restrict__ input,           // [M, K]
    const uint32_t* __restrict__ gate_qweight, // [K, N/8]
    const float* __restrict__ gate_scales,     // [num_groups, N]
    const float* __restrict__ gate_zeros,      // [num_groups, N]
    const uint32_t* __restrict__ up_qweight,   // [K, N/8]
    const float* __restrict__ up_scales,       // [num_groups, N]
    const float* __restrict__ up_zeros,        // [num_groups, N]
    float* __restrict__ output,                // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t group_size
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    uint32_t n_packed = N / 8;
    uint32_t pack_col = col / 8;
    uint32_t sub = col % 8;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint32_t ki = 0; ki < K; ki++) {
        float a = input[row * K + ki];
        uint32_t group = ki / group_size;
        float gs = gate_scales[group * N + col];
        float gz = gate_zeros[group * N + col];
        float us = up_scales[group * N + col];
        float uz = up_zeros[group * N + col];

        uint32_t gate_packed = gate_qweight[ki * n_packed + pack_col];
        uint32_t up_packed = up_qweight[ki * n_packed + pack_col];

        int gq = unpack_int4_awq(gate_packed, sub);
        int uq = unpack_int4_awq(up_packed, sub);

        gate_acc += a * ((float)gq - gz) * gs;
        up_acc += a * ((float)uq - uz) * us;
    }

    output[row * N + col] = silu(gate_acc) * up_acc;
}
