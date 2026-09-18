// Q1_0 GEMV kernels — F32 activation fallback, single-token and token-batched dp4a
//
// Q1_0 block: 128 elements, 18 bytes
// Layout: [d:f16(2), qs:16B] — one sign bit per element, low bit first
// Value: bit set -> +d, clear -> -d
// The decode helpers and offsets come from `../prism_dequant.cuh`; this
// file restates neither.

#include "legacy_ntok.cuh"
#include "prism_ntok.cuh"

// ============================================================================
// Q1_0 GEMV (F32 activation) — warp-per-column
// ============================================================================
//
// Fallback for a K that is not a multiple of the block size, the same
// role `quant_gemv_q8_0_f32` keeps beside its dp4a kernel. `dispatch_gemv`
// routes every aligned K, m = 1 included, to the dp4a kernels below.
//
// One warp per output column, one block of 128 elements per loop step.
// Lane `l` decodes elements `l + 32 * c` for each of the block's 4
// 32-element chunks, so each step reads 128 consecutive activation
// floats in 4 coalesced sweeps.

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q1_0_f32(
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

    const unsigned int blocks_per_row = K / 128;
    const unsigned int row_bytes = blocks_per_row * PrismQ10::BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * PrismQ10::BLOCK_BYTES;
        const float d = prism_load_d(block);
        const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
        const float* act_blk = act_row + b * 128;

        #pragma unroll
        for (int c = 0; c < PrismQ10::CHUNKS_PER_BLOCK; c++) {
            const int e = c * 32 + (int)lane_id;
            acc += act_blk[e] * ((float)gguf_sign_bit(qs, e) * d);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q1_0 GEMV with dp4a (Q1_0 weight × Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each weight
// chunk once for all of them, instead of re-reading the whole weight matrix
// per token as the F32 kernel above does. The body lives in
// `legacy_ntok.cuh` and the decode in `prism_ntok.cuh`; see those headers
// for the lane map, the chunk-per-block rule, the ragged-tail rule and the
// alignment constraint.
//
// Three tile widths exist and `dispatch_gemv` picks the narrowest one that
// covers M. The unsuffixed kernel is the NTOK = 1 instance of the same
// body: four warps per block, one output column per block, each warp
// striding over 8-chunk groups of K with a shared-memory reduction at the
// end. That is the geometry of `quant_gemv_q8_0_q8_1_mwr`, and it serves
// m = 1 with int8 dot products and one `d * d8` FMA per 32-element chunk
// where the F32 kernel above spends one FMA per element.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(1) * WARP_SIZE, 1) void quant_gemv_q1_0_q8_1_mwr(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<PrismQ10, 1>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_q1_0_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<PrismQ10, 2>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_q1_0_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<PrismQ10, 4>(q8_act, weight, output, M, K, N);
}
