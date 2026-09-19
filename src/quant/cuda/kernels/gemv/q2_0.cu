// Q2_0 GEMV kernels — F32 activation fallback, single-token and token-batched dp4a
//
// Q2_0 block: 64 elements, 18 bytes
// Layout: [d:f16(2), qs:16B] — one 2-bit code per element, low bits first
// Value: (code - 1) * d, so {-1, 0, 1, 2} * d
// The decode helpers and offsets come from `../lowbit_dequant.cuh`; this
// file restates neither.

#include "legacy_ntok.cuh"
#include "lowbit_ntok.cuh"

// ============================================================================
// Q2_0 GEMV (F32 activation) — warp-per-column
// ============================================================================
//
// Fallback for a K that is not a multiple of the block size, the same
// role `quant_gemv_q8_0_f32` keeps beside its dp4a kernel. `dispatch_gemv`
// routes every aligned K, m = 1 included, to the dp4a kernels below.
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
    const unsigned int row_bytes = blocks_per_row * LowbitQ20::BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * LowbitQ20::BLOCK_BYTES;
        const float d = lowbit_load_d(block);
        const unsigned char* qs = block + GGUF_LOWBIT_QS_OFFSET;
        const float* act_blk = act_row + b * 64;

        #pragma unroll
        for (int c = 0; c < LowbitQ20::CHUNKS_PER_BLOCK; c++) {
            const int e = c * 32 + (int)lane_id;
            acc += act_blk[e] * ((float)gguf_code2_minus_1(qs, e) * d);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q2_0 GEMV with dp4a (Q2_0 weight × Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each weight
// chunk once for all of them, instead of re-reading the whole weight matrix
// per token as the F32 kernel above does. The body lives in
// `legacy_ntok_body.cuh` and the decode in `lowbit_ntok.cuh`; see those
// headers for the lane map, the chunk-per-block rule, the ragged-tail rules
// and the alignment constraint.
//
// Three tile widths exist and `dispatch_gemv` picks the narrowest one that
// covers M. The unsuffixed kernel is the NTOK = 1 instance of the same
// body: four warps per block, one output column per block, each warp
// striding over 8-chunk groups of K with a shared-memory reduction at the
// end. That is the geometry of `quant_gemv_q8_0_q8_1_mwr`, and it serves
// m = 1 with int8 dot products and one `d * d8` FMA per 32-element chunk
// where the F32 kernel above spends one FMA per element.
//
// `_r4` and `_r8` are the same NTOK = 1 body with ROWS = 4 and 8 output
// columns per block. One block per column re-reads the whole Q8_1
// activation row N times per launch, ~30 MB at K = N = 5120, against a
// 7 MB weight; a block that owns ROWS columns divides that by ROWS. The
// dispatcher picks one via `LOWBIT_GEMV_ROWS` in
// `quant_matmul/format_dispatch/gemv_rows.rs`
// and launches grid x as `ceil(N / ROWS)`. Same warp count and
// `__launch_bounds__` as the unsuffixed kernel: ROWS accumulators and
// decoded words per lane raise register use, and the min-blocks-per-SM
// bound of 1 leaves the compiler free to spend them. A row's bits are the
// same at every ROWS (see the body header).
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(1) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LowbitQ20, 1>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(1) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_r4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LowbitQ20, 1, 4>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(1) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_r8(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LowbitQ20, 1, 8>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LowbitQ20, 2>(q8_act, weight, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_q2_0_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_legacy_q8_1_mwr_ntok<LowbitQ20, 4>(q8_act, weight, output, M, K, N);
}
