// GGUF PrismML-fork block decoders — the ONE place these layouts are written
// for CUDA.
//
// Every CUDA kernel that decodes a Q1_0, Q2_0, PQ2_0 or PTQ1_0 block includes
// this instead of restating the layout. `src/quant/cpu/kernels/dequant_prism.rs`
// is the CPU mirror; keep the two in step.
//
// Q1_0, Q2_0 and PQ2_0 keep `d` at the START and order elements byte-major,
// low bits first. PTQ1_0 is TQ1_0's base-3 trit packing at group 128 with `d`
// at the END; `gguf_base3_trit` in decode.cuh is shared with TQ1_0.
//
//   Q1_0   (18B): d[0..2],  qs[2..18]              1 bit  each, 128 elements
//   Q2_0   (18B): d[0..2],  qs[2..18]              2 bits each,  64 elements
//   PQ2_0  (34B): d[0..2],  qs[2..34]              2 bits each, 128 elements
//   PTQ1_0 (28B): qs[0..24], qh[24..26], d[26..28]
//
// PTQ1_0 runs, in order:
//   [  0,  80)  qs[0..16]   x 5 levels, 16 per level
//   [ 80, 120)  qs[16..24]  x 5 levels,  8 per level
//   [120, 128)  qh[0..2]    x 4 levels,  2 per level

#pragma once

#include <cuda_fp16.h>

#include "decode.cuh"

#define GGUF_PRISM_QS_OFFSET   2
#define GGUF_PTQ1_0_D_OFFSET   26

static __device__ __forceinline__ float prism_load_d(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

// Returns {-1, 1} of element `elem` of a Q1_0 `qs` run: bit set is +1.
static __device__ __forceinline__ int gguf_sign_bit(
    const unsigned char* qs, int elem
) {
    return ((qs[elem >> 3] >> (elem & 7)) & 1) ? 1 : -1;
}

// Returns {-1, 0, 1, 2} of element `elem` of a Q2_0 or PQ2_0 `qs` run:
// code `q` maps to `q - 1` (00=-1, 01=0, 10=+1, 11=+2).
static __device__ __forceinline__ int gguf_code2_minus_1(
    const unsigned char* qs, int elem
) {
    return (int)((qs[elem >> 2] >> ((elem & 3) * 2)) & 0x03) - 1;
}

// Returns the ternary value {-1, 0, 1} of element `elem` (0..128) of a PTQ1_0
// block. `block` points at the start of the 28-byte block.
static __device__ __forceinline__ int gguf_ptq1_0_trit(
    const unsigned char* block, int elem
) {
    unsigned char byte;
    int level;
    if (elem < 80) {
        level = elem >> 4;
        byte = block[elem & 15];
    } else if (elem < 120) {
        const int r = elem - 80;
        level = r >> 3;
        byte = block[16 + (r & 7)];
    } else {
        const int r = elem - 120;
        level = r >> 1;
        byte = block[24 + (r & 1)];
    }
    return gguf_base3_trit(byte, level);
}

// ── Whole-block decoders ────────────────────────────────────────────────

static __device__ __forceinline__ void q1_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = prism_load_d(block);
    const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_sign_bit(qs, i);
}

static __device__ __forceinline__ void q2_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = prism_load_d(block);
    const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
    for (int i = 0; i < 64; i++) out[i] = d * (float)gguf_code2_minus_1(qs, i);
}

static __device__ __forceinline__ void pq2_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = prism_load_d(block);
    const unsigned char* qs = block + GGUF_PRISM_QS_OFFSET;
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_code2_minus_1(qs, i);
}

static __device__ __forceinline__ void ptq1_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = prism_load_d(block + GGUF_PTQ1_0_D_OFFSET);
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_ptq1_0_trit(block, i);
}
