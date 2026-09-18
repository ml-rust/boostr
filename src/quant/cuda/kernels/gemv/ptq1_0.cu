// PTQ1_0 GEMV kernel — F32 activation path
//
// PTQ1_0 block: 128 elements, 28 bytes.
// Layout: qs[0..24], qh[24..26], d:f16[26..28] — the scale is at the END.
// Base-3 encoding: TQ1_0's trit packing at group 128. `gguf_ptq1_0_trit` is
// TQ1_0's `gguf_base3_trit` applied to this block's three qs/qh runs — see
// `prism_dequant.cuh` for the run boundaries. That header, not this file,
// owns the layout and the unpack; `common.cuh` includes `decode.cuh` only,
// which does not have `gguf_ptq1_0_trit`, so this file includes
// `prism_dequant.cuh` directly.

#include "common.cuh"
#include "../prism_dequant.cuh"

#define PTQ1_0_BLOCK_BYTES 28
#define PTQ1_0_BLOCK_SIZE 128

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_ptq1_0_f32(
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

    unsigned int blocks_per_row = K / PTQ1_0_BLOCK_SIZE;
    unsigned int row_bytes = blocks_per_row * PTQ1_0_BLOCK_BYTES;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * PTQ1_0_BLOCK_BYTES;
        const float d = prism_load_d(block + GGUF_PTQ1_0_D_OFFSET);
        unsigned int base = b * PTQ1_0_BLOCK_SIZE;

        for (int i = 0; i < PTQ1_0_BLOCK_SIZE; i++) {
            sum += act_row[base + i] * (d * (float)gguf_ptq1_0_trit(block, i));
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        output[row * N + col] = sum;
}
