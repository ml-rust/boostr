// IQ4_XS GEMV kernel — F32 activation path
//
// IQ4_XS block: 256 elements, 136 bytes
// Layout matches llama.cpp `block_iq4_xs`:
//   { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
// scales_h is TWO bytes at offset 2, scales_l starts at 4 (no pad byte), and
// scales_h carries high scale bits for all EIGHT sub-blocks (16 bits = 8 x 2).
// 8 sub-blocks of 32 elements, 6-bit scales, KVALUES_IQ4NL codebook

// The shared token-batched body and decode policies for the IQ4 codebook
// formats. It pulls in `common.cuh` — WARP_SIZE, WARPS_PER_BLOCK,
// `load_int_ua`, `dp4a`, `warp_reduce_sum` and the MWR reductions — which in
// turn includes `decode.cuh`, whose `KVALUES_IQ4NL` is the one codebook every
// CUDA kernel reads. The F32 kernel below uses it too, so this file carries no
// private copy: a second copy is a second thing to keep in step.
#include "iq4_ntok.cuh"

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_iq4_xs_f32(
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

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 136;
    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const unsigned char* block = w_row + b * 136;
        __half d_half;
        memcpy(&d_half, block, sizeof(__half));
        float d = __half2float(d_half);
        unsigned short scales_h;
        memcpy(&scales_h, block + 2, sizeof(unsigned short));
        const unsigned char* scales_l = block + 4;
        const unsigned char* qs = block + 8;
        unsigned int base = b * 256;

        for (int sb = 0; sb < 8; sb++) {
            int sl = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0x0F;
            int sh = ((unsigned int)scales_h >> (2 * sb)) & 0x03;
            int scale_6bit = sl | (sh << 4);
            float sub_scale = d * (float)(scale_6bit - 32);

            const unsigned char* sub_qs = qs + sb * 16;
            // Split-half nibble order within each sub-block.
            for (int j = 0; j < 16; j++) {
                unsigned char byte = sub_qs[j];
                sum += act_row[base + sb * 32 + j]      * sub_scale * (float)KVALUES_IQ4NL[byte & 0x0F];
                sum += act_row[base + sb * 32 + j + 16] * sub_scale * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        output[row * N + col] = sum;
}

// ============================================================================
// Token-batched IQ4_XS GEMV with dp4a (IQ4_XS weight x Q8_1 activation)
//
// One block covers two token columns and decodes each 32-element run once for
// both, instead of re-reading the whole weight matrix per token as the F32
// kernel above does. Body and decode live in `iq4_ntok.cuh`, shared with
// IQ4_NL; see the header for the lane map, the ragged-tail rule and the
// alignment constraint.
//
// K MULTIPLE. The body walks 32-element runs, but a run's byte offset is
// resolved through its 256-element super-block, so `dispatch_gemv` gates this
// format on `k % 256 == 0` rather than the dp4a branch's usual `k % 32 == 0`.
// A row whose last super-block were partial has no on-disk representation.
//
// There is no single-token sibling: at m = 1 the tile's spare column is pure
// overhead and the F32 kernel above serves that shape.
// ============================================================================

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_iq4_xs_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    quant_gemv_iq4_q8_1_mwr_ntok<Iq4Xs, 2>(q8_act, weight, output, M, K, N);
}
