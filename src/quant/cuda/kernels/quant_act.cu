// F32 → Q8_1 activation quantization kernel
//
// Q8_1 block layout (36 bytes per 32 elements):
//   __half d       — scale factor (2 bytes)
//   __half s       — d * sum(qs) (2 bytes), used for min compensation
//   int8_t qs[32]  — quantized values (32 bytes)
//
// Each warp processes one Q8_1 block (32 elements).
// Lane i handles element i within the block.
// Warp reductions compute amax and sum across all 32 lanes.
//
// Grid:  (num_blocks, M, 1)  where num_blocks = K / 32
// Block: (32, 1, 1)          one warp per block

#include <cuda_fp16.h>

extern "C" __global__ void quantize_f32_q8_1(
    const float* __restrict__ input,     // [M, K]
    unsigned char* __restrict__ output,   // [M, num_blocks * 36] Q8_1 blocks
    unsigned int M,
    unsigned int K
) {
    const unsigned int block_idx = blockIdx.x;   // which Q8_1 block in the row
    const unsigned int m = blockIdx.y;            // which row
    const unsigned int lane_id = threadIdx.x;     // 0..31

    const unsigned int num_blocks = K / 32;
    if (block_idx >= num_blocks) return;

    // Load one float value
    const float xi = input[m * K + block_idx * 32 + lane_id];

    // Warp reduction: find max absolute value
    float amax = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, amax, offset);
        amax = fmaxf(amax, other);
    }

    // Warp reduction: compute sum
    float sum = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // Compute scale
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;

    // Quantize
    const int qi = (int)roundf(xi * id);
    const signed char q = (signed char)min(max(qi, -128), 127);

    // Write to output
    // Q8_1 block layout: [d (half), s (half), qs[32]]
    unsigned char* out_block = output + (m * num_blocks + block_idx) * 36;

    // All lanes write their quantized value
    ((signed char*)(out_block + 4))[lane_id] = q;

    // Lane 0 writes the header
    if (lane_id == 0) {
        __half* header = (__half*)out_block;
        header[0] = __float2half(d);
        header[1] = __float2half(d * sum);
    }
}

// F32 -> repacked Q8_1, for the feature-major MMQ kernels only. The per-token
// layout above is unchanged and still feeds the dp4a, `quant_mmq_q8_0_q8_1_mma`,
// and K-quant kernels.
//
// Format-neutral: every MMQ path quantizes activations to Q8_1, so a future
// format reuses this producer rather than adding its own.
//
// Quantization is identical to `quantize_f32_q8_1`: same `d`, same `id`, same
// rounding and clamp. Only the layout differs, and it differs so that a token
// tile is CONTIGUOUS: records are indexed k-group-major, token-minor, which
// lets the matmul stage its activation tile with a flat copy instead of a
// per-element gather.
//
// Record = 144 bytes covering 128 k-values of one token:
//   int    ds[4]    — one header word per 32-value block, at byte 0
//   int8_t qs[128]  — four 32-value blocks, at byte 16
// Record index = kgroup * ntok + token.
//
// Header word bit layout, per 32-value sub-block:
//   bits  0..15  __half  d       — the block scale
//   bits 16..31  int16   sum     — sum of that block's 32 clamped int8 quants
//
// The low half is a `half` and the high half is a raw two's-complement integer.
// Nothing may read the high half as a float. `MmqfQ80` reads only the low half
// (`__low2float`), which this layout leaves untouched.
//
// The sum is an integer, not `d * sum` as a `half`, because it must be EXACT.
// 32 int8 quants bound the sum by 32 * 128 = 4096, so every attainable value
// fits int16 with no rounding at all. A `half` carries an 11-bit significand,
// so `d * sum` rounded to it perturbs each sub-block's min-correction term by
// that much, and the feature-major Q4_K GEMM then drifts from the GEMV path
// past the parity bound the backend tests hold both to. The consumer rebuilds
// the correction as `(-dmin * m) * d * (float)sum`, and `d` is exact here too.
//
// `d` is rounded through `__half` before being widened, so the value the matmul
// consumes is exactly the `half` scale the per-token layout stores. `sum` is the
// warp-reduced sum of the clamped int8 `q` values, not of the input floats, and
// the reduction runs in `int` so no intermediate rounds.
//
// Grid:  (kgroups * 4, ntok, 1)   Block: (32, 1, 1) — one warp per 32-value block
extern "C" __global__ void quantize_f32_q8_1_mmq(
    const float* __restrict__ input,  // [M, K]
    int* __restrict__ output,         // kgroups * ntok * 36 ints
    unsigned int M,
    unsigned int K,
    unsigned int ntok                 // token slots per k-group; a multiple of the token tile
) {
    const unsigned int b = blockIdx.x;  // 32-value block within the row
    const unsigned int j = blockIdx.y;  // token slot
    const unsigned int lane = threadIdx.x;

    const unsigned int g = b / 4;    // k-group of 128 values
    const unsigned int sub = b % 4;  // 32-value block within that group
    const unsigned int rec = (g * ntok + j) * 36;

    // Padded token slots and padded k-blocks are zeroed here rather than left
    // undefined. The matmul stages them, but the k-step count keeps the padded
    // blocks out of the sum and the masked write-back drops the padded tokens.
    const bool live = j < M && b < K / 32;
    float d = 0.0f;
    signed char q = 0;
    if (live) {
        const float xi = input[(unsigned long long)j * K + b * 32 + lane];

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

    // Sum of the clamped int8 values, not of the input floats. The reduction is
    // INTEGER: |sum| <= 32 * 128 = 4096, so it is exact in `int` and then exact
    // in the int16 it is stored as. A float reduction would round once the
    // partial sums left the exactly-representable range for the widths the
    // consumer cares about, and the min correction needs the exact value.
    int sum = (int)q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    ((signed char*)(output + rec + 4))[sub * 32 + lane] = q;
    if (lane == 0) {
        // Header word assembled explicitly from its two 16-bit fields rather
        // than through a struct, because the high half is an integer and the
        // low half is a `half`:
        //   bits  0..15  __half d
        //   bits 16..31  int16  sum
        // `live` false zeros both `q` (above) and `d`/`sum` here, so a padded
        // or dead slot's word is (d = 0, sum = 0).
        const unsigned int lo = (unsigned int)__half_as_ushort(__float2half(d));
        const unsigned int hi = (unsigned int)(unsigned short)(short)sum;
        output[rec + sub] = (int)(lo | (hi << 16));
    }
}
