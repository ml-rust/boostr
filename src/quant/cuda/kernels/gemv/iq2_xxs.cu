// IQ2_XXS GEMV kernel — F32 activation path
//
// IQ2_XXS block: 256 elements, 66 bytes.
//
// IQ2_XXS is a codebook quantization: `qs` holds INDICES into a grid of
// precomputed points, not magnitudes. The layout and the grid tables live once
// in ../iq_dequant.cuh, shared with the dequant and GEMM paths and gated
// against llama.cpp by tests/gguf_conformance_llama_cpp.rs. The kernel
// decodes each block, then takes its dot product with the activation.

#include "common.cuh"

#include "../iq_dequant.cuh"

#define IQ2_XXS_BLOCK_BYTES 66
#define IQ2_XXS_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq2_xxs_f32(
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

    unsigned int blocks_per_row = K / IQ2_XXS_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ2_XXS_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float w[IQ2_XXS_BLOCK_SIZE];
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        iq2_xxs_dequant_block(w_row + (unsigned long long)b * IQ2_XXS_BLOCK_BYTES, w);
        const float* act = act_row + (unsigned long long)b * IQ2_XXS_BLOCK_SIZE;
        for (int k = 0; k < IQ2_XXS_BLOCK_SIZE; k++)
            sum += act[k] * w[k];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
