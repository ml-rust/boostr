// NF4 (Normal Float 4-bit) dequantization and fused GEMV
// Codebook: 16 values from normal distribution quantiles

#include <stdint.h>

__constant__ float NF4_CODEBOOK[16] = {
    0.0f, -1.0f, -0.6961928f, -0.5250730f,
    -0.3949739f, -0.2844144f, -0.1848489f, -0.0911179f,
    0.0796013f, 0.1609302f, 0.2461123f, 0.3379120f,
    0.4407173f, 0.5626170f, 0.7229568f, 1.0f
};

// NF4 dequantization: nf4_data[N/2] u8 + absmax[N/blocksize] -> output[N] f32
// Grid: (ceil(N/2/256)), Block: 256
// Each thread processes one byte (2 elements)
extern "C" __global__ void nf4_dequant_f32(
    const uint8_t* __restrict__ nf4_data,  // [N/2]
    const float* __restrict__ absmax,      // [num_blocks]
    float* __restrict__ output,            // [N]
    uint32_t num_bytes, uint32_t blocksize
) {
    uint32_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= num_bytes) return;

    uint8_t byte_val = nf4_data[byte_idx];
    int idx_lo = byte_val & 0x0F;
    int idx_hi = (byte_val >> 4) & 0x0F;

    uint32_t elem_lo = byte_idx * 2;
    uint32_t elem_hi = byte_idx * 2 + 1;

    float abs_lo = absmax[elem_lo / blocksize];
    float abs_hi = absmax[elem_hi / blocksize];

    output[elem_lo] = NF4_CODEBOOK[idx_lo] * abs_lo;
    output[elem_hi] = NF4_CODEBOOK[idx_hi] * abs_hi;
}

// NF4 fused GEMM: input[M,K] x nf4_weight[N,K] -> output[M,N]
// weight stored as nf4_data[N*K/2] u8, row-major [N,K]
// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void nf4_gemm_f32(
    const float* __restrict__ input,       // [M, K]
    const uint8_t* __restrict__ nf4_weight,// [N*K/2]
    const float* __restrict__ absmax,      // [N*K/blocksize]
    float* __restrict__ output,            // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t blocksize
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    uint32_t k_packed = K / 2;
    uint32_t weight_row_start = col * k_packed;
    uint32_t absmax_row_start = col * (K / blocksize);

    float acc = 0.0f;
    for (uint32_t bi = 0; bi < k_packed; bi++) {
        uint8_t byte_val = nf4_weight[weight_row_start + bi];
        int idx_lo = byte_val & 0x0F;
        int idx_hi = (byte_val >> 4) & 0x0F;

        uint32_t elem_lo = bi * 2;
        uint32_t elem_hi = bi * 2 + 1;

        float w_lo = NF4_CODEBOOK[idx_lo] * absmax[absmax_row_start + elem_lo / blocksize];
        float w_hi = NF4_CODEBOOK[idx_hi] * absmax[absmax_row_start + elem_hi / blocksize];

        acc += input[row * K + elem_lo] * w_lo + input[row * K + elem_hi] * w_hi;
    }
    output[row * N + col] = acc;
}
