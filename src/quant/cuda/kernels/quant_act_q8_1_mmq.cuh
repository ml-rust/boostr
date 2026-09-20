// One 32-value block of the feature-major MMQ activation record.
//
// The record layout and the reason for it are documented at
// `quantize_f32_q8_1_mmq` in quant_act.cu. This header holds the per-block
// quantization that every producer of the record runs: the plain producer
// in quant_act.cu and the Hadamard-fused one in fwht_quant_act.cu include
// it, so both form a block's bytes from the same float with the same
// instructions, and a record from either is byte for byte the record from
// the other.
//
// Called by a full warp. Lane `lane` holds `xi`, the block's element at
// that lane. `live` is warp-uniform and says whether the block exists: a
// padded token slot or a padded k-block is written as zeros (`d = 0`,
// `sum = 0`, every `q = 0`). `rec` points at the block's record (36 ints)
// and `sub` is the block's index within the record's four.
//
// `d` is rounded through `__half` before it scales anything, so the scale
// the matmul reads is the scale the block was quantized with. `sum` is the
// integer sum of the clamped int8 quants: at most 32 * 128 in magnitude, so
// exact in `int` and exact in the int16 it is stored as.

#ifndef BOOSTR_QUANT_ACT_Q8_1_MMQ_CUH
#define BOOSTR_QUANT_ACT_Q8_1_MMQ_CUH

#include <cuda_fp16.h>

__device__ __forceinline__ void q8_1_mmq_write_block(
    float xi,
    bool live,
    int* __restrict__ rec,
    unsigned int sub,
    unsigned int lane
) {
    float d = 0.0f;
    signed char q = 0;
    if (live) {
        float amax = fabsf(xi);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        }

        d = __half2float(__float2half(amax / 127.0f));
        const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
        const int qi = (int)roundf(xi * id);
        q = (signed char)min(max(qi, -128), 127);
    }

    int sum = (int)q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    ((signed char*)(rec + 4))[sub * 32 + lane] = q;
    if (lane == 0) {
        // Header word assembled from its two 16-bit fields: bits 0..15 the
        // `half` scale, bits 16..31 the int16 sum. `live` false zeroes `q`,
        // `d` and `sum`, so a padded slot's word is (d = 0, sum = 0).
        const unsigned int lo = (unsigned int)__half_as_ushort(__float2half(d));
        const unsigned int hi = (unsigned int)(unsigned short)(short)sum;
        rec[sub] = (int)(lo | (hi << 16));
    }
}

#endif // BOOSTR_QUANT_ACT_Q8_1_MMQ_CUH
