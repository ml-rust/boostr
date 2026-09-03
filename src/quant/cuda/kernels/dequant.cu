// Dequantization CUDA kernels for boostr
// Supports: Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_XS → f32
//
// Each kernel processes one block per thread (or group of threads).
// Block formats match llama.cpp bit-for-bit.

#include <cuda_fp16.h>

#include "iq_dequant.cuh"

extern "C" {

// ============================================================================
// Q4_0 Dequantization
// Block: 32 elements, 18 bytes (2-byte f16 scale + 16 bytes nibbles)
// Formula: x = (nibble - 8) * scale
// One thread per block of 32 elements
// ============================================================================

__global__ void dequant_q4_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 18;
    float* out = output + bid * 32;

    // Read f16 scale
    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);

    const unsigned char* qs = block + 2;

    // Split-half nibble order (llama.cpp `dequantize_row_q4_0`): element j takes
    // the LOW nibble of qs[j], element j+16 the HIGH nibble of the SAME byte.
    // They are 16 apart, not adjacent — see decode.cuh.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low = (int)(byte & 0x0F) - 8;
        int high = (int)((byte >> 4) & 0x0F) - 8;
        out[j] = (float)low * d;
        out[j + 16] = (float)high * d;
    }
}

// ============================================================================
// Q5_0 Dequantization
// Block: 32 elements, 22 bytes (2-byte f16 scale + 4-byte qh + 16 bytes nibbles)
// Formula: x = ((low4 | (high1 << 4)) - 16) * scale
// ============================================================================

__global__ void dequant_q5_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 22;
    float* out = output + bid * 32;

    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);
    // The block stride is 22 bytes, so `block + 2` is only 2-byte aligned for
    // odd blocks: a 4-byte load through a `unsigned int*` traps with
    // CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
    // launch. memcpy is the portable unaligned load.
    unsigned int qh;
    memcpy(&qh, block + 2, sizeof(unsigned int));
    const unsigned char* qs = block + 6;

    // Split-half nibble order (llama.cpp `dequantize_row_q5_0`): element j takes
    // the LOW nibble of qs[j] and fifth bit `qh` bit j; element j+16 takes the
    // HIGH nibble of the same byte and `qh` bit j+16. See decode.cuh.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low  = (byte & 0x0F) | (((qh >> j) & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = (float)(low - 16) * d;
        out[j + 16] = (float)(high - 16) * d;
    }
}

// ============================================================================
// Q8_0 Dequantization
// Block: 32 elements, 34 bytes (2-byte f16 scale + 32 bytes i8 values)
// Formula: x = qs[i] * scale
// ============================================================================

__global__ void dequant_q8_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 34;
    float* out = output + bid * 32;

    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);

    const signed char* qs = reinterpret_cast<const signed char*>(block + 2);

    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d;
    }
}

// ============================================================================
// Q4_K Dequantization
// Block: 256 elements, 144 bytes
// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 128-byte qs
// 8 sub-blocks of 32 elements with 6-bit scales/mins
// ============================================================================

__global__ void dequant_q4_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 144;
    float* out = output + bid * 256;

    __half d_half = *reinterpret_cast<const __half*>(block);
    __half dmin_half = *reinterpret_cast<const __half*>(block + 2);
    float d = __half2float(d_half);
    float dmin = __half2float(dmin_half);

    const unsigned char* sc = block + 4;   // 12-byte scales
    const unsigned char* qs = block + 16;  // 128-byte quantized values

    // Unpack 6-bit scales and mins (matches llama.cpp get_scale_min_k4)
    unsigned char scales[8];
    unsigned char mins[8];

    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i] = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
    }

    // 8 sub-blocks of 32 elements
    for (int j = 0; j < 8; j++) {
        float dl = d * (float)scales[j];
        float ml = dmin * (float)mins[j];

        int chunk = j / 2;
        int is_high = j % 2;
        int qs_base = chunk * 32;

        for (int l = 0; l < 32; l++) {
            float q;
            if (is_high) {
                q = (float)((qs[qs_base + l] >> 4) & 0x0F);
            } else {
                q = (float)(qs[qs_base + l] & 0x0F);
            }
            out[j * 32 + l] = dl * q - ml;
        }
    }
}

// ============================================================================
// Q6_K Dequantization
// Block: 256 elements, 210 bytes
// Layout: 128-byte ql, 64-byte qh, 16-byte scales (i8), 2-byte d
// ============================================================================

__global__ void dequant_q6_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 210;
    float* out = output + bid * 256;

    const unsigned char* ql = block;
    const unsigned char* qh = block + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(block + 192);
    __half d_half = *reinterpret_cast<const __half*>(block + 208);
    float d = __half2float(d_half);

    // Process in two halves of 128 elements
    for (int n = 0; n < 2; n++) {
        int y_base = n * 128;
        int ql_base = n * 64;
        int qh_base = n * 32;
        int sc_base = n * 8;

        for (int l = 0; l < 32; l++) {
            int is = l / 16;

            int q1 = (int)((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) - 32;
            int q2 = (int)((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4)) - 32;
            int q3 = (int)((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) - 32;
            int q4 = (int)((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4)) - 32;

            out[y_base + l]      = d * (float)sc[sc_base + is]     * (float)q1;
            out[y_base + l + 32] = d * (float)sc[sc_base + is + 2] * (float)q2;
            out[y_base + l + 64] = d * (float)sc[sc_base + is + 4] * (float)q3;
            out[y_base + l + 96] = d * (float)sc[sc_base + is + 6] * (float)q4;
        }
    }
}

// ============================================================================
// Q2_K Dequantization
// Block: 256 elements, 84 bytes
// Layout: 16-byte sc, 64-byte qs, 2-byte d, 2-byte dmin
// 16 sub-blocks of 16 elements, 2-bit values
// ============================================================================

__global__ void dequant_q2_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 84;
    float* out = output + bid * 256;

    const unsigned char* sc = block;
    const unsigned char* qs = block + 16;
    __half d_half = *reinterpret_cast<const __half*>(block + 80);
    __half dmin_half = *reinterpret_cast<const __half*>(block + 82);
    float d = __half2float(d_half);
    float dmin = __half2float(dmin_half);

    int y = 0, is = 0;
    for (int n = 0; n < 2; n++) {
        const unsigned char* q = qs + n * 32;
        for (int shift = 0; shift < 8; shift += 2) {
            float dl = d * (float)(sc[is] & 0x0F);
            float ml = dmin * (float)(sc[is] >> 4);
            is++;
            for (int l = 0; l < 16; l++)
                out[y++] = dl * (float)((q[l] >> shift) & 3) - ml;
            dl = d * (float)(sc[is] & 0x0F);
            ml = dmin * (float)(sc[is] >> 4);
            is++;
            for (int l = 0; l < 16; l++)
                out[y++] = dl * (float)((q[16 + l] >> shift) & 3) - ml;
        }
    }
}

// ============================================================================
// Q3_K Dequantization
// Block: 256 elements, 110 bytes
// Layout: 32-byte hmask, 64-byte qs, 12-byte scales, 2-byte d
// ============================================================================

__global__ void dequant_q3_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 110;
    float* out = output + bid * 256;

    const unsigned char* hmask = block;
    const unsigned char* qs = block + 32;
    const unsigned char* sc_raw = block + 96;
    __half d_half = *reinterpret_cast<const __half*>(block + 108);
    float d = __half2float(d_half);

    // Unpack 16 6-bit scales from 12 bytes
    unsigned int aux[4];
    unsigned char aux_bytes[12];
    for (int i = 0; i < 12; i++) aux_bytes[i] = sc_raw[i];
    memcpy(&aux[0], aux_bytes, 4);
    memcpy(&aux[1], aux_bytes + 4, 4);
    memcpy(&aux[2], aux_bytes + 8, 4);

    unsigned int tmp = aux[2];
    const unsigned int KMASK1 = 0x03030303u;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    unsigned int a0 = aux[0], a1 = aux[1];
    aux[0] = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    aux[2] = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    signed char scales[16];
    memcpy(&scales[0],  &aux[0], 4);
    memcpy(&scales[4],  &aux[1], 4);
    memcpy(&scales[8],  &aux[2], 4);
    memcpy(&scales[12], &aux[3], 4);
    for (int i = 0; i < 16; i++)
        scales[i] = (signed char)((unsigned char)scales[i] - 32);

    int y = 0, is = 0;
    unsigned char m = 1;
    for (int n = 0; n < 2; n++) {
        const unsigned char* q = qs + n * 32;
        for (int shift = 0; shift < 8; shift += 2) {
            float dl = d * (float)scales[is++];
            for (int l = 0; l < 16; l++) {
                int low2 = (q[l] >> shift) & 3;
                int hsub = (hmask[l] & m) ? 0 : 4;
                out[y++] = dl * (float)(low2 - hsub);
            }
            dl = d * (float)scales[is++];
            for (int l = 0; l < 16; l++) {
                int low2 = (q[16 + l] >> shift) & 3;
                int hsub = (hmask[16 + l] & m) ? 0 : 4;
                out[y++] = dl * (float)(low2 - hsub);
            }
            m <<= 1;
        }
    }
}

// ============================================================================
// Q5_K Dequantization
// Block: 256 elements, 176 bytes
// Layout: 2-byte d, 2-byte dmin, 12-byte sc, 32-byte qh, 128-byte qs
// 8 sub-blocks of 32 elements, 5-bit values (4-bit low + 1-bit high)
// ============================================================================

__global__ void dequant_q5_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 176;
    float* out = output + bid * 256;

    __half d_half = *reinterpret_cast<const __half*>(block);
    __half dmin_half = *reinterpret_cast<const __half*>(block + 2);
    float d = __half2float(d_half);
    float dmin = __half2float(dmin_half);
    const unsigned char* sc = block + 4;
    const unsigned char* qh = block + 16;
    const unsigned char* qs = block + 48;

    // Unpack 6-bit scales and mins (same as Q4_K)
    unsigned char scales[8];
    unsigned char mins[8];
    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i] = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
    }

    for (int j = 0; j < 8; j++) {
        float dl = d * (float)scales[j];
        float ml = dmin * (float)mins[j];

        // llama.cpp `dequantize_row_q5_K`: sub-block PAIRS share one 32-byte run
        // of `qs` — the even sub-block takes the low nibbles, the odd one the
        // high nibbles of the SAME bytes (identical to Q4_K above). It is NOT a
        // per-sub-block 16-byte run with interleaved nibbles.
        int qs_base = (j / 2) * 32;
        int is_high_nibble = j % 2;

        for (int l = 0; l < 32; l++) {
            int low4 = is_high_nibble ? ((qs[qs_base + l] >> 4) & 0x0F)
                                      : (qs[qs_base + l] & 0x0F);
            // 5th bit: `qh` is indexed by ELEMENT within the sub-block and the
            // BIT is the sub-block index — one qh byte per element carries that
            // element's high bit for all 8 sub-blocks. Not a flat bitstream.
            int high1 = (qh[l] >> j) & 0x01;
            float q = (float)(low4 | (high1 << 4));
            out[j * 32 + l] = dl * q - ml;
        }
    }
}

// ============================================================================
// IQ4_NL Dequantization
// Block: 32 elements, 18 bytes (f16 scale + 16 bytes nibbles)
// Non-linear codebook: x = scale * KVALUES_IQ4NL[nibble]
// ============================================================================

__constant__ signed char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__global__ void dequant_iq4_nl_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 18;
    float* out = output + bid * 32;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    const unsigned char* qs = block + 2;

    // Split-half nibble order (llama.cpp `dequantize_row_iq4_nl`): `y[j]` takes
    // the low nibble, `y[j + QK4_NL/2]` the high nibble of the SAME byte.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        out[j]      = d * (float)KVALUES_IQ4NL[byte & 0x0F];
        out[j + 16] = d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
    }
}

// ============================================================================
// IQ4_XS Dequantization
// Block: 256 elements, 136 bytes
// Layout matches llama.cpp `block_iq4_xs` exactly:
//   { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
// so scales_h is a TWO-byte field at 2..4 and scales_l occupies 4..8. There is
// no pad byte, and scales_h carries high scale bits for all EIGHT sub-blocks
// (16 bits = 8 x 2).
// 8 sub-blocks of 32 elements, 6-bit scales, KVALUES_IQ4NL codebook
// ============================================================================

__global__ void dequant_iq4_xs_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 136;
    float* out = output + bid * 256;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    unsigned short scales_h;
    memcpy(&scales_h, block + 2, sizeof(unsigned short));
    const unsigned char* scales_l = block + 4;
    const unsigned char* qs = block + 8;

    for (int sb = 0; sb < 8; sb++) {
        // 4 low bits from scales_l (one nibble per sub-block), 2 high bits from
        // scales_h (2 bits per sub-block across all 8).
        int sl = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0x0F;
        int sh = ((unsigned int)scales_h >> (2 * sb)) & 0x03;
        int scale_6bit = sl | (sh << 4);
        float sub_scale = d * (float)(scale_6bit - 32);

        const unsigned char* sub_qs = qs + sb * 16;
        float* sub_out = out + sb * 32;
        // Split-half nibble order within each sub-block.
        for (int j = 0; j < 16; j++) {
            unsigned char byte = sub_qs[j];
            sub_out[j]      = sub_scale * (float)KVALUES_IQ4NL[byte & 0x0F];
            sub_out[j + 16] = sub_scale * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
        }
    }
}

// ============================================================================
// IQ3_S and IQ2_XS Dequantization
//
// Both are codebook quantizations. The block layouts and the grid tables live
// once in iq_dequant.cuh, shared with dequant_generic.cu, the quant-matmul
// path and the GEMV/GEMM kernels, and gated against llama.cpp by
// tests/gguf_conformance_llama_cpp.rs.
// ============================================================================

__global__ void dequant_iq3_s_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    iq3_s_dequant_block(input + (unsigned long long)bid * 110, output + (unsigned long long)bid * 256);
}

__global__ void dequant_iq2_xs_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    iq2_xs_dequant_block(input + (unsigned long long)bid * 74, output + (unsigned long long)bid * 256);
}

} // extern "C"
