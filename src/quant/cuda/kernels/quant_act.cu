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
