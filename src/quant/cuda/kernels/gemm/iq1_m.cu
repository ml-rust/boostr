// IQ1_M tiled GEMM — activation [M,K] x weight [N,K]^T -> output [M,N]
//
// IQ1_M block: 256 elements, 56 bytes.
//
// IQ1_M is a codebook quantization: `qs` holds INDICES into a grid of
// precomputed points, not magnitudes. The layout and the grid tables live once
// in ../iq_dequant.cuh, shared with the dequant and GEMV paths and gated
// against llama.cpp by tests/gguf_conformance_llama_cpp.rs. The kernel
// decodes each block, then takes its dot product with the activation.

#include <cuda_fp16.h>

#include "../iq_dequant.cuh"

#define IQ1_M_BLOCK_BYTES 56
#define IQ1_M_BLOCK_SIZE 256

extern "C" __global__ void quant_matmul_iq1_m_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / IQ1_M_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ1_M_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float w[IQ1_M_BLOCK_SIZE];
    float sum = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        iq1_m_dequant_block(w_row + (unsigned long long)b * IQ1_M_BLOCK_BYTES, w);
        const float* act = act_row + (unsigned long long)b * IQ1_M_BLOCK_SIZE;
        for (int k = 0; k < IQ1_M_BLOCK_SIZE; k++)
            sum += act[k] * w[k];
    }
    output[row * N + col] = sum;
}
