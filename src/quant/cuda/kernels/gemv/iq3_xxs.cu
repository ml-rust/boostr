// IQ3_XXS GEMV kernel — F32 activation path
//
// IQ3_XXS block: 256 elements, 98 bytes.
//
// IQ3_XXS is a codebook quantization: `qs` holds INDICES into a grid of
// precomputed points, not magnitudes. The layout and the grid tables live once
// in ../iq_dequant.cuh, shared with the dequant and GEMM paths and gated
// against llama.cpp by tests/gguf_conformance_llama_cpp.rs. The kernel
// decodes each block, then takes its dot product with the activation.

// The shared token-batched body and decode policies for the six grid-indexed
// IQ formats. It pulls in `common.cuh` — WARP_SIZE, WARPS_PER_BLOCK,
// `load_int_ua`, `dp4a`, `warp_reduce_sum` and the MWR reductions — and
// `../iq_dequant.cuh`, whose grid tables and per-block decoders the F32 kernel
// below reads. One include for both paths, so neither can drift onto a private
// copy of a table.
#include "iq_grid_ntok.cuh"

#define IQ3_XXS_BLOCK_BYTES 98
#define IQ3_XXS_BLOCK_SIZE 256

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq3_xxs_f32(
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

    unsigned int blocks_per_row = K / IQ3_XXS_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * IQ3_XXS_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float w[IQ3_XXS_BLOCK_SIZE];
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        iq3_xxs_dequant_block(w_row + (unsigned long long)b * IQ3_XXS_BLOCK_BYTES, w);
        const float* act = act_row + (unsigned long long)b * IQ3_XXS_BLOCK_SIZE;
        for (int k = 0; k < IQ3_XXS_BLOCK_SIZE; k++)
            sum += act[k] * w[k];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}

// ============================================================================
// Token-batched IQ3_XXS GEMV with dp4a (IQ3_XXS weight x Q8_1 activation)
//
// One block covers two token columns and decodes each 8-element sub-group —
// TWO four-component grid reads, one sign-table read and one scale nibble —
// ONCE for both, instead of repeating that per token as the F32 kernel above
// does. Body and decode policy live in `iq_grid_ntok.cuh`, shared with the
// other five grid-indexed IQ formats; see the header for the lane map, the
// ragged-tail rule and the alignment constraint that puts this format's 4-byte
// `aux` read through `load_int_ua`.
//
// K MULTIPLE. `dispatch_gemv` gates this format on `k % 256 == 0` rather than
// the dp4a branch's usual 32: a sub-group's byte offset is resolved through its
// 256-element super-block, and a row whose last super-block were partial has no
// on-disk representation.
//
// There is no single-token sibling: at m = 1 the tile's spare column is pure
// overhead and the F32 kernel above serves that shape.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_iq3_xxs_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_iq_grid_q8_1_mwr_ntok<IqGridIq3Xxs, 2>(q8_act, weight, output, M, K, N);
}
