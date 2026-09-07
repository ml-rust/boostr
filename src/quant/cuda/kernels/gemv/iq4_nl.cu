// IQ4_NL GEMV kernel — F32 activation path
//
// IQ4_NL block: 32 elements, 18 bytes
// Layout: [d:f16(2), qs:16B]
// Non-linear codebook: value = KVALUES_IQ4NL[nibble]
// dequant(i) = d * KVALUES_IQ4NL[nibble]

// The shared token-batched body and decode policies for the IQ4 codebook
// formats. It pulls in `common.cuh` — WARP_SIZE, WARPS_PER_BLOCK,
// `load_int_ua`, `dp4a`, `warp_reduce_sum` and the MWR reductions — which in
// turn includes `decode.cuh`, whose `KVALUES_IQ4NL` is the one codebook every
// CUDA kernel reads. The F32 kernel below uses it too, so this file carries no
// private copy: a second copy is a second thing to keep in step.
#include "iq4_ntok.cuh"

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq4_nl_f32(
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

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 18;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * 18;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 32;

        // Split-half nibble order: qs[j] holds element j in its low nibble and
        // element j + 16 in its high nibble (llama.cpp dequantize_row_iq4_nl).
        for (int j = 0; j < 16; j++) {
            unsigned char byte = qs[j];
            sum += act_row[base + j]      * d * (float)KVALUES_IQ4NL[byte & 0x0F];
            sum += act_row[base + j + 16] * d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        output[row * N + col] = sum;
}

// ============================================================================
// Token-batched IQ4_NL GEMV with dp4a (IQ4_NL weight x Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each weight
// block once for all of them, instead of re-reading the whole weight matrix
// per token as the F32 kernel above does. Both the `_n2` and `_n4` tile
// widths exist; `dispatch_gemv` picks the narrowest one that covers M. Body
// and decode live in `iq4_ntok.cuh`, shared with IQ4_XS; see the header for
// the lane map, the ragged-tail rule and the alignment constraint.
//
// There is no single-token sibling: at m = 1 the tile's spare column is pure
// overhead and the F32 kernel above serves that shape.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_iq4_nl_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_iq4_q8_1_mwr_ntok<Iq4Nl, 2>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_iq4_nl_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_iq4_q8_1_mwr_ntok<Iq4Nl, 4>(q8_act, weight, output, M, K, N);
}
