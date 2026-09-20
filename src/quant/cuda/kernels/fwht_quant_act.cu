// Hadamard rotation fused into the feature-major MMQ activation quantization,
// for one token.
//
// A rotated linear at decode runs numr's `fwht` (fwht.cu: sign-multiply,
// `1/sqrt(block_size)` scale, Sylvester butterfly, one block per segment)
// and then `quantize_f32_q8_1_mmq` (quant_act.cu) over the rotated row.
// Both are latency-bound at one token, and the rotated f32 row only exists
// to be quantized. This kernel keeps the segment in shared memory between
// the two: it is numr's kernel up to the store, then each warp quantizes
// the segment's 32-value blocks from shared memory with the same body the
// plain producer runs (`q8_1_mmq_write_block`, quant_act_q8_1_mmq.cuh).
//
// BIT IDENTITY with the two launches. The rotated row numr stores is f32,
// so the global round trip rounds nothing; the fused record equals the
// two-launch record when every float operation is the same operation in
// the same order. The transform is copied from fwht.cu stage for stage:
// `v = x * scale`, then `v = v * sign`, then per stage `s[i] = u + v`,
// `s[j] = u - v` with `h` doubling from 1. The arithmetic is written as
// PTX so it compiles as numr compiles it and not as this crate's flags
// would compile C: numr builds fwht.cu with `--ftz=false`, this crate
// builds with `--use_fast_math` alone (denormals flushed), so a C-level
// `u + v` here would be `add.ftz.f32` against numr's `add.f32`. `mul.rn`,
// `add.rn` and `sub.rn` without `.ftz` are the instructions numr's PTX
// holds (no `.ftz`, and no contraction: the products go through shared
// memory and a barrier before any add, so `.rn` changes nothing). The
// scale is `rsqrt.approx.f32` of the block size converted `cvt.rn`, which
// is what numr's `1.0f / sqrtf(n)` becomes under `--use_fast_math`. The
// quantization is the shared header, compiled in this crate with the same
// flags as quant_act.cu.
//
// Coverage. One token only: `blockIdx.y` is the token slot, slot 0 is the
// token and every other slot is padding written as zeros, as the plain
// producer writes it. `block_size` is a multiple of 128 so every k-group
// of the record lies inside one segment, and `K` is a multiple of
// `block_size`; the launcher checks both and falls back to the two
// launches otherwise.
//
// Grid:  (K / block_size, ntok, 1)   one segment per block
// Block: (min(block_size, 256), 1, 1)
// Shared memory: block_size * 4 bytes, dynamic.

#include <cuda_fp16.h>
#include "quant_act_q8_1_mmq.cuh"

extern __shared__ __align__(16) float boostr_fwht_qact_smem[];

__device__ __forceinline__ float fwht_qact_mul(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fwht_qact_add(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fwht_qact_sub(float a, float b) {
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fwht_qact_rsqrt(float n) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(n));
    return r;
}

extern "C" __global__ void fwht_quantize_f32_q8_1_mmq(
    const float* __restrict__ input,  // [1, K]
    const float* __restrict__ signs,  // [K], or null
    int* __restrict__ output,         // (K / 128) * ntok * 36 ints
    unsigned int K,
    unsigned int block_size,
    unsigned int ntok                 // token slots per k-group
) {
    float* s = boostr_fwht_qact_smem;

    const unsigned int seg = blockIdx.x;
    const unsigned int j = blockIdx.y;
    const unsigned int col0 = seg * block_size;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int warps = blockDim.x / 32;
    // 32-value blocks of this segment, and the row index of its first.
    const unsigned int blocks32 = block_size / 32;
    const unsigned int b0 = col0 / 32;

    if (j != 0) {
        // A padded token slot: no transform, zero records.
        for (unsigned int c = warp; c < blocks32; c += warps) {
            const unsigned int b = b0 + c;
            const unsigned int rec = ((b / 4) * ntok + j) * 36;
            q8_1_mmq_write_block(0.0f, false, output + rec, b % 4, lane);
        }
        return;
    }

    // Load, folding in the scale and the sign, in numr's order.
    const float scale = fwht_qact_rsqrt((float)block_size);
    for (unsigned int i = threadIdx.x; i < block_size; i += blockDim.x) {
        float v = fwht_qact_mul(input[col0 + i], scale);
        if (signs != nullptr) {
            v = fwht_qact_mul(v, signs[col0 + i]);
        }
        s[i] = v;
    }
    __syncthreads();

    // Sylvester butterfly: the low element of a pair takes u + v, the high
    // one u - v. Pair `p` at stage `h` sits at `i = (p / h) * 2h + (p % h)`.
    const unsigned int half = block_size / 2;
    for (unsigned int h = 1; h < block_size; h <<= 1) {
        for (unsigned int p = threadIdx.x; p < half; p += blockDim.x) {
            const unsigned int i = (p / h) * 2 * h + (p % h);
            const unsigned int jj = i + h;
            const float u = s[i];
            const float v = s[jj];
            s[i] = fwht_qact_add(u, v);
            s[jj] = fwht_qact_sub(u, v);
        }
        __syncthreads();
    }

    // Quantize from shared memory: one warp per 32-value block, the block's
    // record and sub-index as the plain producer computes them for token 0.
    for (unsigned int c = warp; c < blocks32; c += warps) {
        const unsigned int b = b0 + c;
        const unsigned int rec = ((b / 4) * ntok + j) * 36;
        q8_1_mmq_write_block(s[c * 32 + lane], true, output + rec, b % 4, lane);
    }
}
