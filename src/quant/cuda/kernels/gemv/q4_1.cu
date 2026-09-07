// Q4_1 GEMV kernel — F32 activation only (simple 32-element blocks)
//
// Q4_1 block: 32 elements, 20 bytes
// Layout: [d:f16(2), m:f16(2), qs:16B]
// 4-bit unsigned values with min: dequant = d * nibble + m

#include "legacy_ntok.cuh"

#define Q4_1_BLOCK_BYTES 20

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q4_1_f32(
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

    const unsigned int blocks_per_row = K / 32;
    const unsigned int row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q4_1_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float mn = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const unsigned char* qs = block + 4;

        // Split-half nibble order: lane `l` takes the LOW nibble of qs[l] for
        // l < 16, the HIGH nibble of qs[l - 16] for l >= 16 (llama.cpp
        // dequantize_row_q4_1). See gguf_split_half_nibble in decode.cuh.
        int nibble = gguf_split_half_nibble(qs, (int)lane_id);
        float val = d * (float)nibble + mn;
        acc += act_row[b * 32 + lane_id] * val;
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Token-batched Q4_1 GEMV with dp4a (Q4_1 weight × Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each weight
// block once for all of them, instead of re-reading the whole weight matrix
// per token as the F32 kernel above does. Both the `_n2` and `_n4` tile
// widths exist; `dispatch_gemv` picks the narrowest one that covers M. Body
// and decode live in `legacy_ntok.cuh`, shared with the other legacy
// 32-element formats; see the header for the lane map, the ragged-tail rule
// and the alignment constraint.
//
// There is no single-token sibling: at m = 1 the tile's spare column is pure
// overhead and the F32 kernel above serves that shape.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_q4_1_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LegacyQ41, 2>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_q4_1_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LegacyQ41, 4>(q8_act, weight, output, M, K, N);
}
