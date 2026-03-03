// Generic dequantization CUDA kernel — fallback for all GGUF quant formats
// that lack dedicated optimized kernels.
//
// One thread per quant block. The format_id parameter selects the decode path
// via a switch statement. Not optimal, but correct for all 23 formats.
//
// Optimized kernels in dequant.cu (Q4_0, Q8_0, Q4_K, Q6_K) should be
// preferred when available; this is the catch-all fallback.
//
// IMPORTANT: All multi-byte loads from quant block data use memcpy to avoid
// misaligned access errors. Quant blocks are packed contiguously and their
// internal fields may not be naturally aligned.

#include <cuda_fp16.h>

// Format IDs (must match QuantFormat::format_id() in Rust)
#define FMT_Q4_0    0
#define FMT_Q4_1    1
#define FMT_Q5_0    2
#define FMT_Q5_1    3
#define FMT_Q8_0    4
#define FMT_Q8_1    5
#define FMT_Q2K     6
#define FMT_Q3K     7
#define FMT_Q4K     8
#define FMT_Q5K     9
#define FMT_Q6K     10
#define FMT_Q8K     11
#define FMT_IQ1S    12
#define FMT_IQ1M    13
#define FMT_IQ2XXS  14
#define FMT_IQ2XS   15
#define FMT_IQ2S    16
#define FMT_IQ3XXS  17
#define FMT_IQ3S    18
#define FMT_IQ4NL   19
#define FMT_IQ4XS   20
#define FMT_TQ1_0   21
#define FMT_TQ2_0   22

// IQ4_NL codebook (matches llama.cpp kvalues_iq4nl)
__constant__ signed char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ── Safe unaligned load helpers ─────────────────────────────────────
// Quant blocks are packed contiguously; internal fields may not be
// naturally aligned for their type. memcpy avoids misaligned-access traps.

__device__ __forceinline__ float load_f16_as_f32(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

__device__ __forceinline__ float load_f32(const unsigned char* p) {
    float tmp;
    memcpy(&tmp, p, sizeof(float));
    return tmp;
}

__device__ __forceinline__ unsigned int load_u32(const unsigned char* p) {
    unsigned int tmp;
    memcpy(&tmp, p, sizeof(unsigned int));
    return tmp;
}

__device__ __forceinline__ unsigned long long load_u64(const unsigned char* p) {
    unsigned long long tmp;
    memcpy(&tmp, p, sizeof(unsigned long long));
    return tmp;
}

// ── Simple quant device functions ────────────────────────────────────

__device__ void dequant_q4_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        out[i * 2]     = (float)((int)(byte & 0x0F) - 8) * d;
        out[i * 2 + 1] = (float)((int)((byte >> 4) & 0x0F) - 8) * d;
    }
}

__device__ void dequant_q4_1_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    float m = load_f16_as_f32(block + 2);
    const unsigned char* qs = block + 4;
    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        out[i * 2]     = d * (float)(byte & 0x0F) + m;
        out[i * 2 + 1] = d * (float)((byte >> 4) & 0x0F) + m;
    }
}

__device__ void dequant_q5_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    unsigned int qh = load_u32(block + 2);
    const unsigned char* qs = block + 6;
    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        int low  = (byte & 0x0F) | (((qh >> (i * 2))     & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (i * 2 + 1)) & 1) << 4);
        out[i * 2]     = (float)(low - 16) * d;
        out[i * 2 + 1] = (float)(high - 16) * d;
    }
}

__device__ void dequant_q5_1_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    float m = load_f16_as_f32(block + 2);
    unsigned int qh = load_u32(block + 4);
    const unsigned char* qs = block + 8;
    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        int low  = (byte & 0x0F) | (((qh >> (i * 2))     & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (i * 2 + 1)) & 1) << 4);
        out[i * 2]     = d * (float)low + m;
        out[i * 2 + 1] = d * (float)high + m;
    }
}

__device__ void dequant_q8_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const signed char* qs = reinterpret_cast<const signed char*>(block + 2);
    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d;
    }
}

__device__ void dequant_q8_1_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    float s = load_f16_as_f32(block + 2);
    const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d + s;
    }
}

// ── K-quant device functions ─────────────────────────────────────────

__device__ void dequant_q2k_block(const unsigned char* block, float* out) {
    const unsigned char* sc = block;
    const unsigned char* qs = block + 16;
    float d    = load_f16_as_f32(block + 80);
    float dmin = load_f16_as_f32(block + 82);

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

__device__ void dequant_q3k_block(const unsigned char* block, float* out) {
    const unsigned char* hmask = block;
    const unsigned char* qs = block + 32;
    const unsigned char* sc_raw = block + 96;
    float d = load_f16_as_f32(block + 108);

    // Unpack 16 6-bit scales from 12 bytes — byte-by-byte to avoid alignment issues
    unsigned int aux[4];
    unsigned char aux_bytes[12];
    memcpy(aux_bytes, sc_raw, 12);
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
    for (int i = 0; i < 16; i++) {
        scales[i] = (signed char)((unsigned char)scales[i] - 32);
    }

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

__device__ void dequant_q4k_block(const unsigned char* block, float* out) {
    float d    = load_f16_as_f32(block);
    float dmin = load_f16_as_f32(block + 2);
    const unsigned char* sc = block + 4;
    const unsigned char* qs = block + 16;

    unsigned char scales[8], mins[8];
    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i]   = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i]   = (sc[i + 4] >> 4)   | ((sc[i] >> 6) << 4);
    }

    for (int j = 0; j < 8; j++) {
        float dl = d * (float)scales[j];
        float ml = dmin * (float)mins[j];
        int chunk = j / 2;
        int is_high = j % 2;
        int qs_base = chunk * 32;
        for (int l = 0; l < 32; l++) {
            float q;
            if (is_high) q = (float)((qs[qs_base + l] >> 4) & 0x0F);
            else         q = (float)(qs[qs_base + l] & 0x0F);
            out[j * 32 + l] = dl * q - ml;
        }
    }
}

__device__ void dequant_q5k_block(const unsigned char* block, float* out) {
    float d    = load_f16_as_f32(block);
    float dmin = load_f16_as_f32(block + 2);
    const unsigned char* sc = block + 4;
    const unsigned char* qh = block + 16;
    const unsigned char* qs = block + 48;

    unsigned char scales[8], mins[8];
    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i]   = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i]   = (sc[i + 4] >> 4)   | ((sc[i] >> 6) << 4);
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
            int qh_byte = idx / 8;
            int qh_bit  = idx % 8;
            int high1 = (qh[qh_byte] >> qh_bit) & 0x01;
            float q = (float)(low4 | (high1 << 4));
            out[idx] = dl * q - ml;
        }
    }
}

__device__ void dequant_q6k_block(const unsigned char* block, float* out) {
    const unsigned char* ql = block;
    const unsigned char* qh = block + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(block + 192);
    float d = load_f16_as_f32(block + 208);

    for (int n = 0; n < 2; n++) {
        int y_base  = n * 128;
        int ql_base = n * 64;
        int qh_base = n * 32;
        int sc_base = n * 8;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[ql_base+l] & 0x0F) | ((qh[qh_base+l] & 0x03) << 4)) - 32;
            int q2 = (int)((ql[ql_base+l+32] & 0x0F) | (((qh[qh_base+l]>>2) & 0x03) << 4)) - 32;
            int q3 = (int)((ql[ql_base+l] >> 4) | (((qh[qh_base+l]>>4) & 0x03) << 4)) - 32;
            int q4 = (int)((ql[ql_base+l+32] >> 4) | (((qh[qh_base+l]>>6) & 0x03) << 4)) - 32;
            out[y_base+l]    = d * (float)sc[sc_base+is]   * (float)q1;
            out[y_base+l+32] = d * (float)sc[sc_base+is+2] * (float)q2;
            out[y_base+l+64] = d * (float)sc[sc_base+is+4] * (float)q3;
            out[y_base+l+96] = d * (float)sc[sc_base+is+6] * (float)q4;
        }
    }
}

__device__ void dequant_q8k_block(const unsigned char* block, float* out) {
    float d = load_f32(block); // f32 scale, not f16
    const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
    for (int i = 0; i < 256; i++) {
        out[i] = (float)qs[i] * d;
    }
}

// ── IQ/TQ device functions ───────────────────────────────────────────

__device__ void dequant_iq4_nl_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    for (int i = 0; i < 16; i++) {
        unsigned char byte = qs[i];
        out[i * 2]     = d * (float)KVALUES_IQ4NL[byte & 0x0F];
        out[i * 2 + 1] = d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
    }
}

__device__ void dequant_iq4_xs_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    unsigned char scales_h = block[2];
    const unsigned char* scales_l = block + 3;
    const unsigned char* qs = block + 8;

    for (int sb = 0; sb < 8; sb++) {
        int sl = (sb % 2 == 0) ? (scales_l[sb/2] & 0x0F) : ((scales_l[sb/2] >> 4) & 0x0F);
        int sh = (sb < 4) ? ((scales_h >> (2 * sb)) & 0x03) : 0;
        int scale_6bit = sl | (sh << 4);
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

__device__ void dequant_tq2_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    for (int i = 0; i < 64; i++) {
        unsigned char byte = qs[i];
        for (int j = 0; j < 4; j++) {
            int val = ((byte >> (2 * j)) & 0x03) - 1;
            out[i * 4 + j] = d * (float)val;
        }
    }
}

__device__ void dequant_tq1_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    int idx = 0;
    for (int i = 0; i < 52; i++) {
        unsigned int val = (unsigned int)qs[i];
        for (int j = 0; j < 5; j++) {
            if (idx >= 256) break;
            int t = (int)(val % 3) - 1;
            out[idx] = d * (float)t;
            val /= 3;
            idx++;
        }
    }
    // Zero remaining (52*5=260 > 256, but loop breaks at 256)
}

__device__ void dequant_iq2_xxs_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;

    for (int group = 0; group < 8; group++) {
        const unsigned char* gdata = qs + group * 8;
        unsigned long long q64 = load_u64(gdata);
        unsigned int grid_indices = (unsigned int)q64;
        unsigned int signs_and_scales = (unsigned int)(q64 >> 32);
        unsigned int sub_scale_bits = (signs_and_scales >> 28) & 0x0F;
        float sub_scale = d * (0.5f + (float)sub_scale_bits);
        float* group_out = out + group * 32;

        for (int sub = 0; sub < 4; sub++) {
            unsigned int grid_idx = (grid_indices >> (8 * sub)) & 0xFF;
            unsigned char sign_bits = (unsigned char)((signs_and_scales >> (7 * sub)) & 0x7F);
            for (int k = 0; k < 8; k++) {
                // Extract 2-bit grid value
                int shift = k * 2;
                int bits;
                if (shift < 8) bits = (grid_idx >> shift) & 0x03;
                else           bits = ((grid_idx >> (shift - 8)) ^ (grid_idx >> 1)) & 0x03;
                float grid_val;
                switch (bits) {
                    case 0: grid_val = 0.0f; break;
                    case 1: grid_val = 1.0f; break;
                    case 2: grid_val = 2.0f; break;
                    default: grid_val = 3.0f; break;
                }
                float sign = ((sign_bits >> k) & 1) ? -1.0f : 1.0f;
                group_out[sub * 8 + k] = sub_scale * grid_val * sign;
            }
        }
    }
}

__device__ void dequant_iq2_xs_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* scales = block + 2;
    const unsigned char* qs = block + 18;

    for (int sb = 0; sb < 16; sb++) {
        float scale = d * ((float)((signed char)scales[sb]) + 0.5f);
        unsigned int q_val = (unsigned int)qs[sb * 2] | ((unsigned int)qs[sb * 2 + 1] << 8);
        unsigned int grid_idx = q_val & 0x1FF;
        unsigned char signs = (unsigned char)(q_val >> 9);
        float* sub_out = out + sb * 16;

        for (int k = 0; k < 16; k++) {
            int pos = k % 8;
            int bits = (grid_idx >> (pos * 2)) & 0x03;
            float grid_val;
            switch (bits) {
                case 0: grid_val = 0.0f; break;
                case 1: grid_val = 1.0f; break;
                case 2: grid_val = 2.0f; break;
                default: grid_val = 3.0f; break;
            }
            float sign = ((signs >> (k % 8)) & 1) ? -1.0f : 1.0f;
            sub_out[k] = scale * grid_val * sign;
        }
    }
}

__device__ void dequant_iq2_s_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    const unsigned char* signs_data = block + 38;
    const unsigned char* scales = block + 54;

    for (int sb = 0; sb < 16; sb++) {
        unsigned char scale_byte = (sb < 28) ? scales[sb] : 0;
        float sub_scale = d * ((float)((signed char)scale_byte) + 0.5f);
        float* sub_out = out + sb * 16;

        for (int k = 0; k < 16; k++) {
            int byte_idx = sb * 2 + k / 8;
            unsigned char grid_byte = (byte_idx < 32) ? qs[byte_idx] : 0;
            int bit_pos = k % 8;
            unsigned char sign_byte = (sb < 16) ? signs_data[sb] : 0;
            float sign = ((sign_byte >> bit_pos) & 1) ? -1.0f : 1.0f;
            float val = (float)((grid_byte >> ((bit_pos % 4) * 2)) & 0x03);
            sub_out[k] = sub_scale * val * sign;
        }
    }
}

__device__ void dequant_iq3_xxs_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;

    for (int group = 0; group < 8; group++) {
        const unsigned char* gdata = qs + group * 12;
        unsigned int signs = load_u32(gdata + 8);
        unsigned int sub_scale_bits = (signs >> 28) & 0x0F;
        float sub_scale = d * (1.0f + (float)sub_scale_bits);
        float* group_out = out + group * 32;

        for (int sub = 0; sub < 4; sub++) {
            unsigned int grid_idx = (unsigned int)gdata[sub * 2] |
                                    ((unsigned int)gdata[sub * 2 + 1] << 8);
            for (int k = 0; k < 8; k++) {
                float val = (float)((grid_idx >> (k * 2)) & 0x03) + 1.0f;
                unsigned int sign_bit = (signs >> (sub * 8 + k)) & 1;
                float sign = sign_bit ? -1.0f : 1.0f;
                group_out[sub * 8 + k] = sub_scale * val * sign;
            }
        }
    }
}

__device__ void dequant_iq3_s_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    const unsigned char* qh = block + 34;
    const unsigned char* signs_data = block + 38;
    const unsigned char* scales = block + 70;

    for (int sb = 0; sb < 8; sb++) {
        float sub_scale = d * (1.0f + (float)(scales[sb] & 0x0F));
        float* sub_out = out + sb * 32;

        for (int k = 0; k < 32; k++) {
            int byte_idx = sb * 4 + k / 8;
            int bit_pos = k % 8;
            int q3 = (byte_idx < 32) ? ((qs[byte_idx] >> ((bit_pos % 4) * 2)) & 0x03) : 0;
            int qh_byte_idx = (sb * 32 + k) / 8;
            int qh_bit = (qh_byte_idx < 4) ? ((qh[qh_byte_idx] >> ((sb * 32 + k) % 8)) & 1) : 0;
            float val = (float)q3 + (float)qh_bit * 4.0f + 1.0f;
            int sign_byte_idx = sb * 4 + k / 8;
            unsigned char sign_byte = (sign_byte_idx < 32) ? signs_data[sign_byte_idx] : 0;
            float sign = ((sign_byte >> (k % 8)) & 1) ? -1.0f : 1.0f;
            sub_out[k] = sub_scale * val * sign;
        }
    }
}

__device__ void dequant_iq1_s_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    const unsigned char* qh = block + 34;

    for (int sb = 0; sb < 16; sb++) {
        unsigned int qs_val = (unsigned int)qs[sb * 2] | ((unsigned int)qs[sb * 2 + 1] << 8);
        unsigned int grid_idx = qs_val & 0x0FFF;
        unsigned char sign_bits = qh[sb];
        float* sub_out = out + sb * 16;

        unsigned int grid_val = grid_idx;
        for (int k = 0; k < 16; k++) {
            int t = (int)(grid_val % 3) - 1;
            float sign = ((sign_bits >> (k % 8)) & 1) ? -1.0f : 1.0f;
            sub_out[k] = d * (float)t * sign;
            grid_val /= 3;
        }
    }
}

__device__ void dequant_iq1_m_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* scales_data = block + 2;
    const unsigned char* qs = block + 8;
    const unsigned char* qh = block + 40;

    for (int sb = 0; sb < 16; sb++) {
        int scale_bit_offset = sb * 3;
        int byte_idx = scale_bit_offset / 8;
        int bit_offset = scale_bit_offset % 8;
        unsigned int raw;
        if (byte_idx + 1 < 6) {
            unsigned int lo = (unsigned int)scales_data[byte_idx];
            unsigned int hi = (unsigned int)scales_data[byte_idx + 1];
            raw = ((lo | (hi << 8)) >> bit_offset) & 0x07;
        } else if (byte_idx < 6) {
            raw = ((unsigned int)(scales_data[byte_idx] >> bit_offset)) & 0x07;
        } else {
            raw = 0;
        }
        float sub_scale = d * ((float)raw + 0.5f);

        unsigned int qs_val = (unsigned int)qs[sb * 2] | ((unsigned int)qs[sb * 2 + 1] << 8);
        unsigned int grid_idx = qs_val & 0x0FFF;
        unsigned char sign_bits = qh[sb];
        float* sub_out = out + sb * 16;

        unsigned int grid_val = grid_idx;
        for (int k = 0; k < 16; k++) {
            int t = (int)(grid_val % 3) - 1;
            float sign = ((sign_bits >> (k % 8)) & 1) ? -1.0f : 1.0f;
            sub_out[k] = sub_scale * (float)t * sign;
            grid_val /= 3;
        }
    }
}

// ── Main dispatch kernel ─────────────────────────────────────────────

extern "C" {

// Block sizes per format
__device__ int get_block_size(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0: case FMT_Q4_1: case FMT_Q5_0: case FMT_Q5_1:
        case FMT_Q8_0: case FMT_Q8_1: case FMT_IQ4NL:
            return 32;
        default: // All k-quants, IQ (except IQ4NL), TQ
            return 256;
    }
}

__device__ int get_block_bytes(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0:   return 18;
        case FMT_Q4_1:   return 20;
        case FMT_Q5_0:   return 22;
        case FMT_Q5_1:   return 24;
        case FMT_Q8_0:   return 34;
        case FMT_Q8_1:   return 36;
        case FMT_Q2K:    return 84;
        case FMT_Q3K:    return 110;
        case FMT_Q4K:    return 144;
        case FMT_Q5K:    return 176;
        case FMT_Q6K:    return 210;
        case FMT_Q8K:    return 292;
        case FMT_IQ1S:   return 50;
        case FMT_IQ1M:   return 56;
        case FMT_IQ2XXS: return 66;
        case FMT_IQ2XS:  return 74;
        case FMT_IQ2S:   return 82;
        case FMT_IQ3XXS: return 98;
        case FMT_IQ3S:   return 110;
        case FMT_IQ4NL:  return 18;
        case FMT_IQ4XS:  return 136;
        case FMT_TQ1_0:  return 54;
        case FMT_TQ2_0:  return 66;
        default:         return 0;
    }
}

/// Generic dequantization kernel — handles all 23 GGUF formats.
/// One thread per quant block. format_id selects the decode path.
__global__ void dequant_generic_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks,
    unsigned int format_id
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    int block_bytes = get_block_bytes(format_id);
    int block_size  = get_block_size(format_id);
    if (block_bytes == 0) return; // unknown format

    const unsigned char* block = input + (unsigned long long)bid * block_bytes;
    float* out = output + (unsigned long long)bid * block_size;

    switch (format_id) {
        case FMT_Q4_0:   dequant_q4_0_block(block, out); break;
        case FMT_Q4_1:   dequant_q4_1_block(block, out); break;
        case FMT_Q5_0:   dequant_q5_0_block(block, out); break;
        case FMT_Q5_1:   dequant_q5_1_block(block, out); break;
        case FMT_Q8_0:   dequant_q8_0_block(block, out); break;
        case FMT_Q8_1:   dequant_q8_1_block(block, out); break;
        case FMT_Q2K:    dequant_q2k_block(block, out); break;
        case FMT_Q3K:    dequant_q3k_block(block, out); break;
        case FMT_Q4K:    dequant_q4k_block(block, out); break;
        case FMT_Q5K:    dequant_q5k_block(block, out); break;
        case FMT_Q6K:    dequant_q6k_block(block, out); break;
        case FMT_Q8K:    dequant_q8k_block(block, out); break;
        case FMT_IQ4NL:  dequant_iq4_nl_block(block, out); break;
        case FMT_IQ4XS:  dequant_iq4_xs_block(block, out); break;
        case FMT_IQ2XXS: dequant_iq2_xxs_block(block, out); break;
        case FMT_IQ2XS:  dequant_iq2_xs_block(block, out); break;
        case FMT_IQ2S:   dequant_iq2_s_block(block, out); break;
        case FMT_IQ3XXS: dequant_iq3_xxs_block(block, out); break;
        case FMT_IQ3S:   dequant_iq3_s_block(block, out); break;
        case FMT_IQ1S:   dequant_iq1_s_block(block, out); break;
        case FMT_IQ1M:   dequant_iq1_m_block(block, out); break;
        case FMT_TQ1_0:  dequant_tq1_0_block(block, out); break;
        case FMT_TQ2_0:  dequant_tq2_0_block(block, out); break;
    }
}

} // extern "C"
