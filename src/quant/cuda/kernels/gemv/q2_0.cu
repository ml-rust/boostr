// Q2_0 GEMV kernels — F32 activation (m = 1) and token-batched dp4a
//
// Q2_0 block: 64 elements, 18 bytes
// Layout: [d:f16(2), qs:16B] — one 2-bit code per element, low bits first
// Value: (code - 1) * d, so {-1, 0, 1, 2} * d
// The decode helpers and offsets come from `../prism_dequant.cuh`; this
// file restates neither.

#include "legacy_ntok.cuh"
#include "prism_ntok.cuh"

// ============================================================================
// Q2_0 GEMV (F32 activation) — warp-per-column
// ============================================================================
//
// One warp per output column, one block of 64 elements per loop step.
// Lane `l` decodes elements `l + 32 * c` for each of the block's 2
// 32-element chunks, so each step reads 64 consecutive activation
// floats in 2 coalesced sweeps.

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q2_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N) return;

    const unsigned int blocks_per_row = K / 64;
    const unsigned int row_bytes = blocks_per_row * PrismQ20::BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * PrismQ20::BLOCK_BYTES;
        const float d = prism_load_d(block);
        const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
        const float* act_blk = act_row + b * 64;

        #pragma unroll
        for (int c = 0; c < PrismQ20::CHUNKS_PER_BLOCK; c++) {
            const int e = c * 32 + (int)lane_id;
            acc += act_blk[e] * ((float)gguf_code2_minus_1(qs, e) * d);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Token-batched Q2_0 GEMV with dp4a (Q2_0 weight × Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each weight
// chunk once for all of them, instead of re-reading the whole weight matrix
// per token as the F32 kernel above does. Both the `_n2` and `_n4` tile
// widths exist; `dispatch_gemv` picks the narrowest one that covers M. The
// body lives in `legacy_ntok.cuh` and the decode in `prism_ntok.cuh`; see
// those headers for the lane map, the chunk-per-block rule, the ragged-tail
// rule and the alignment constraint.
//
// There is no single-token sibling: at m = 1 the tile's spare column is pure
// overhead and the F32 kernel above serves that shape.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<PrismQ20, 2>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<PrismQ20, 4>(q8_act, weight, output, M, K, N);
}
