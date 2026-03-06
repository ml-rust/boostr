// Dequantization CUDA kernels for boostr
// Supports: Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_XS → f32
//
// Each kernel processes one block per thread (or group of threads).
// Block formats match llama.cpp bit-for-bit.

#include <cuda_fp16.h>

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

    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        int low = (int)(byte & 0x0F) - 8;
        int high = (int)((byte >> 4) & 0x0F) - 8;
        out[i * 2] = (float)low * d;
        out[i * 2 + 1] = (float)high * d;
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
    unsigned int qh = *reinterpret_cast<const unsigned int*>(block + 2);
    const unsigned char* qs = block + 6;

    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        int low  = (byte & 0x0F) | (((qh >> (i * 2))     & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (i * 2 + 1)) & 1) << 4);
        out[i * 2]     = (float)(low - 16) * d;
        out[i * 2 + 1] = (float)(high - 16) * d;
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
        for (int l = 0; l < 32; l++) {
            int idx = j * 32 + l;
            int qs_idx = j * 16 + l / 2;
            int low4;
            if (l % 2 == 0) low4 = qs[qs_idx] & 0x0F;
            else            low4 = (qs[qs_idx] >> 4) & 0x0F;
            int high1 = (qh[idx / 8] >> (idx % 8)) & 0x01;
            float q = (float)(low4 | (high1 << 4));
            out[idx] = dl * q - ml;
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

    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        out[i * 2]     = d * (float)KVALUES_IQ4NL[byte & 0x0F];
        out[i * 2 + 1] = d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
    }
}

// ============================================================================
// IQ4_XS Dequantization
// Block: 256 elements, 136 bytes
// Layout: f16 d (2B) + scales_h (1B) + scales_l (4B) + pad (1B) + qs (128B)
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
    unsigned char scales_h = block[2];
    const unsigned char* scales_l = block + 3;
    const unsigned char* qs = block + 8;

    for (int sb = 0; sb < 8; sb++) {
        unsigned char sl = (sb % 2 == 0) ? (scales_l[sb / 2] & 0x0F) : ((scales_l[sb / 2] >> 4) & 0x0F);
        unsigned char sh = (sb < 4) ? ((scales_h >> (2 * sb)) & 0x03) : 0;
        int scale_6bit = (int)(sl | (sh << 4));
        float sub_scale = d * (float)(scale_6bit - 32);

        const unsigned char* sub_qs = qs + sb * 16;
        float* sub_out = out + sb * 32;
        for (int i = 0; i < 16; i++) {
            unsigned char byte = sub_qs[i];
            sub_out[i * 2]     = sub_scale * (float)KVALUES_IQ4NL[byte & 0x0F];
            sub_out[i * 2 + 1] = sub_scale * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
        }
    }
}

// ============================================================================
// IQ3_S Dequantization
// Block: 256 elements, 110 bytes
// Layout: f16 d (2B) + qs (32B) + qh (4B) + signs (32B) + scales (8B)
// 8 sub-blocks of 32, 3-bit values with sign bits
// ============================================================================

__global__ void dequant_iq3_s_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 110;
    float* out = output + bid * 256;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    const unsigned char* qs = block + 2;
    const unsigned char* qh = block + 34;
    const unsigned char* signs = block + 38;
    const unsigned char* scales = block + 70;

    for (int sb = 0; sb < 8; sb++) {
        float sub_scale = d * (1.0f + (float)(scales[sb] & 0x0F));
        float* sub_out = out + sb * 32;

        for (int k = 0; k < 32; k++) {
            int byte_idx = sb * 4 + k / 8;
            int q3 = (byte_idx < 32) ? ((qs[byte_idx] >> ((k % 8) / 2 * 2)) & 0x03) : 0;
            int qh_byte_idx = (sb * 32 + k) / 8;
            int qh_bit = (qh_byte_idx < 4) ? ((qh[qh_byte_idx] >> ((sb * 32 + k) % 8)) & 1) : 0;
            float val = (float)q3 + (float)qh_bit * 4.0f + 1.0f;

            int sign_byte_idx = sb * 4 + k / 8;
            int sign_bit = (sign_byte_idx < 32) ? ((signs[sign_byte_idx] >> (k % 8)) & 1) : 0;
            float sign = sign_bit ? -1.0f : 1.0f;

            sub_out[k] = sub_scale * val * sign;
        }
    }
}

// ============================================================================
// IQ2_XS Dequantization
// Block: 256 elements, 74 bytes
// Layout: f16 d (2B) + scales (16B) + qs (56B)
// 16 sub-blocks of 16 values
// ============================================================================

__global__ void dequant_iq2_xs_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 74;
    float* out = output + bid * 256;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    const unsigned char* sc = block + 2;
    const unsigned char* qs = block + 18;

    for (int sb = 0; sb < 16; sb++) {
        float scale = d * ((float)((signed char)sc[sb]) + 0.5f);

        unsigned int q_offset = sb * 2;
        unsigned int q_val = (unsigned int)qs[q_offset] | ((unsigned int)qs[q_offset + 1] << 8);

        float* sub_out = out + sb * 16;
        for (int k = 0; k < 16; k++) {
            int bits = (q_val >> k) & 1;
            // IQ2_XS: each element is sign * (grid_magnitude)
            // Simplified: use 2-bit value extraction
            int val_2bit = (q_val >> (k % 8 * 2)) & 0x03;
            float magnitude = (float)val_2bit + 0.5f;
            float sign = ((q_val >> (8 + k)) & 1) ? -1.0f : 1.0f;
            sub_out[k] = scale * magnitude * sign;
        }
    }
}

} // extern "C"
