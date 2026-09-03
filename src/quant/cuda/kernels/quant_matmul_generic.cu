// Generic fused dequant+matmul CUDA kernel — fallback for all GGUF quant formats
//
// Dequantizes weight blocks on-the-fly in registers (never materializes full f32
// weight), then accumulates dot product with f32 activation. One warp per output
// element, cooperating on the K-dimension reduction via warp shuffle.
//
// Not optimized (no dp4a, no shared memory tiling), but correct for all 23 formats
// and memory-efficient. Callers must prefer the dedicated kernels in
// quant_gemv.cu / quant_matmul.cu for formats that have them.
//
// IMPORTANT: All multi-byte loads from quant block data use memcpy to avoid
// misaligned access errors. Quant blocks are packed contiguously and their
// internal fields are not always naturally aligned.

#include <cuda_fp16.h>

#include "decode.cuh"
#include "iq_dequant.cuh"

#define WARP_SIZE 32

// Format IDs — must match QuantFormat::format_id() and dequant_generic.cu
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

__constant__ signed char KVALUES_IQ4NL_GM[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ── Safe unaligned load helpers ─────────────────────────────────────

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

__device__ __forceinline__ unsigned short load_u16(const unsigned char* p) {
    unsigned short tmp;
    memcpy(&tmp, p, sizeof(unsigned short));
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

// ── Per-block dequant into a local f32 buffer ────────────────────────

__device__ void dq_q4_0(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    // Split-half nibble order: element j takes the low nibble, element j+16 the
    // high nibble of the SAME byte (llama.cpp dequantize_row_q4_0).
    for (int j = 0; j < 16; j++) {
        unsigned char v = b[2 + j];
        out[j]      = (float)((int)(v & 0x0F) - 8) * d;
        out[j + 16] = (float)((int)((v>>4) & 0x0F) - 8) * d;
    }
}

__device__ void dq_q4_1(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    float m = load_f16_as_f32(b+2);
    for (int j = 0; j < 16; j++) {
        unsigned char v = b[4+j];
        out[j]      = d * (float)(v & 0x0F) + m;
        out[j + 16] = d * (float)((v>>4) & 0x0F) + m;
    }
}

__device__ void dq_q5_0(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    unsigned int qh = load_u32(b+2);
    for (int j = 0; j < 16; j++) {
        unsigned char v = b[6+j];
        // 5th bit: bit j for the first half, bit j+16 for the second.
        int lo  = (v & 0x0F) | (((qh >> j) & 1) << 4);
        int hi  = ((v>>4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = (float)(lo - 16) * d;
        out[j + 16] = (float)(hi - 16) * d;
    }
}

__device__ void dq_q5_1(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    float m = load_f16_as_f32(b+2);
    unsigned int qh = load_u32(b+4);
    for (int j = 0; j < 16; j++) {
        unsigned char v = b[8+j];
        // 5th bit: bit j for the first half, bit j+16 for the second.
        int lo  = (v & 0x0F) | (((qh >> j) & 1) << 4);
        int hi  = ((v>>4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = d * (float)lo + m;
        out[j + 16] = d * (float)hi + m;
    }
}

__device__ void dq_q8_0(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    const signed char* qs = (const signed char*)(b+2);
    for (int i = 0; i < 32; i++) out[i] = (float)qs[i] * d;
}

__device__ void dq_q8_1(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    // b[2..4] is `s`, llama.cpp's precomputed dot-product sum in `block_q8_1`,
    // NOT a min. Dequant is q * d and must ignore it.
    const signed char* qs = (const signed char*)(b+4);
    for (int i = 0; i < 32; i++) out[i] = (float)qs[i] * d;
}

__device__ void dq_q2k(const unsigned char* b, float* out) {
    const unsigned char* sc = b;
    const unsigned char* qs = b + 16;
    float d    = load_f16_as_f32(b+80);
    float dmin = load_f16_as_f32(b+82);
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
                out[y++] = dl * (float)((q[16+l] >> shift) & 3) - ml;
        }
    }
}

__device__ void dq_q3k(const unsigned char* b, float* out) {
    const unsigned char* hmask = b;
    const unsigned char* qs = b + 32;
    const unsigned char* sc_raw = b + 96;
    float d = load_f16_as_f32(b+108);

    unsigned int aux[4];
    unsigned char aux_bytes[12];
    memcpy(aux_bytes, sc_raw, 12);
    memcpy(&aux[0], aux_bytes, 4);
    memcpy(&aux[1], aux_bytes + 4, 4);
    memcpy(&aux[2], aux_bytes + 8, 4);

    unsigned int tmp = aux[2];
    const unsigned int M1 = 0x03030303u, M2 = 0x0f0f0f0fu;
    unsigned int a0 = aux[0], a1 = aux[1];
    aux[0] = (a0 & M2) | ((tmp & M1) << 4);
    aux[1] = (a1 & M2) | (((tmp>>2) & M1) << 4);
    aux[2] = ((a0>>4) & M2) | (((tmp>>4) & M1) << 4);
    aux[3] = ((a1>>4) & M2) | (((tmp>>6) & M1) << 4);
    signed char scales[16];
    memcpy(&scales[0],  &aux[0], 4);
    memcpy(&scales[4],  &aux[1], 4);
    memcpy(&scales[8],  &aux[2], 4);
    memcpy(&scales[12], &aux[3], 4);
    for (int i = 0; i < 16; i++) scales[i] = (signed char)((unsigned char)scales[i] - 32);

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
                int low2 = (q[16+l] >> shift) & 3;
                int hsub = (hmask[16+l] & m) ? 0 : 4;
                out[y++] = dl * (float)(low2 - hsub);
            }
            m <<= 1;
        }
    }
}

__device__ void dq_q4k(const unsigned char* b, float* out) {
    float d    = load_f16_as_f32(b);
    float dmin = load_f16_as_f32(b+2);
    const unsigned char* sc = b + 4;
    const unsigned char* qs = b + 16;
    unsigned char scales[8], mins[8];
    for (int i = 0; i < 4; i++) { scales[i] = sc[i] & 0x3F; mins[i] = sc[i+4] & 0x3F; }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i+4] & 0x0F) | ((sc[i-4] >> 6) << 4);
        mins[i]   = (sc[i+4] >> 4)   | ((sc[i] >> 6) << 4);
    }
    for (int j = 0; j < 8; j++) {
        float dl = d * (float)scales[j], ml = dmin * (float)mins[j];
        int chunk = j/2, is_high = j%2, qs_base = chunk * 32;
        for (int l = 0; l < 32; l++) {
            float q = is_high ? (float)((qs[qs_base+l]>>4) & 0x0F) : (float)(qs[qs_base+l] & 0x0F);
            out[j*32+l] = dl * q - ml;
        }
    }
}

__device__ void dq_q5k(const unsigned char* b, float* out) {
    float d    = load_f16_as_f32(b);
    float dmin = load_f16_as_f32(b+2);
    const unsigned char* sc = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* qs = b + 48;
    unsigned char scales[8], mins[8];
    for (int i = 0; i < 4; i++) { scales[i] = sc[i] & 0x3F; mins[i] = sc[i+4] & 0x3F; }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i+4] & 0x0F) | ((sc[i-4] >> 6) << 4);
        mins[i]   = (sc[i+4] >> 4)   | ((sc[i] >> 6) << 4);
    }
    for (int j = 0; j < 8; j++) {
        float dl = d * (float)scales[j], ml = dmin * (float)mins[j];
        // Sub-block PAIRS share one 32-byte run of qs; qh is indexed by ELEMENT
        // with the BIT selected by sub-block index. Must match the CPU
        // `dequant_q5k` / llama.cpp exactly.
        int qs_base = (j / 2) * 32;
        int is_high_nibble = j % 2;
        for (int l = 0; l < 32; l++) {
            int low4 = is_high_nibble ? ((qs[qs_base + l] >> 4) & 0x0F)
                                      : (qs[qs_base + l] & 0x0F);
            int high1 = (qh[l] >> j) & 0x01;
            out[j*32 + l] = dl * (float)(low4 | (high1<<4)) - ml;
        }
    }
}

__device__ void dq_q6k(const unsigned char* b, float* out) {
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char* sc = (const signed char*)(b + 192);
    float d = load_f16_as_f32(b+208);
    for (int n = 0; n < 2; n++) {
        int yb = n*128, qlb = n*64, qhb = n*32, scb = n*8;
        for (int l = 0; l < 32; l++) {
            int is = l/16;
            int q1 = (int)((ql[qlb+l] & 0x0F) | ((qh[qhb+l] & 0x03) << 4)) - 32;
            int q2 = (int)((ql[qlb+l+32] & 0x0F) | (((qh[qhb+l]>>2) & 0x03) << 4)) - 32;
            int q3 = (int)((ql[qlb+l] >> 4) | (((qh[qhb+l]>>4) & 0x03) << 4)) - 32;
            int q4 = (int)((ql[qlb+l+32] >> 4) | (((qh[qhb+l]>>6) & 0x03) << 4)) - 32;
            out[yb+l]    = d * (float)sc[scb+is]   * (float)q1;
            out[yb+l+32] = d * (float)sc[scb+is+2] * (float)q2;
            out[yb+l+64] = d * (float)sc[scb+is+4] * (float)q3;
            out[yb+l+96] = d * (float)sc[scb+is+6] * (float)q4;
        }
    }
}

__device__ void dq_q8k(const unsigned char* b, float* out) {
    float d = load_f32(b);
    const signed char* qs = (const signed char*)(b+4);
    for (int i = 0; i < 256; i++) out[i] = (float)qs[i] * d;
}

__device__ void dq_iq4_nl(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b);
    // Split-half nibble order (llama.cpp dequantize_row_iq4_nl).
    for (int j = 0; j < 16; j++) {
        unsigned char v = b[2+j];
        out[j]      = d * (float)KVALUES_IQ4NL_GM[v & 0x0F];
        out[j + 16] = d * (float)KVALUES_IQ4NL_GM[(v>>4) & 0x0F];
    }
}

__device__ void dq_iq4_xs(const unsigned char* b, float* out) {
    // block_iq4_xs = { half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
    // scales_h is TWO bytes at offset 2, so scales_l starts at 4 (no pad byte),
    // and scales_h supplies high scale bits for all EIGHT sub-blocks.
    float d = load_f16_as_f32(b);
    unsigned int scales_h = (unsigned int)load_u16(b + 2);
    const unsigned char* scales_l = b + 4;
    const unsigned char* qs = b + 8;
    for (int sb = 0; sb < 8; sb++) {
        int sl = (scales_l[sb/2] >> (4*(sb%2))) & 0x0F;
        int sh = (scales_h >> (2*sb)) & 0x03;
        float sub_scale = d * (float)((sl | (sh<<4)) - 32);
        for (int j = 0; j < 16; j++) {
            unsigned char v = qs[sb*16+j];
            out[sb*32 + j]      = sub_scale * (float)KVALUES_IQ4NL_GM[v & 0x0F];
            out[sb*32 + j + 16] = sub_scale * (float)KVALUES_IQ4NL_GM[(v>>4) & 0x0F];
        }
    }
}

__device__ void dq_tq2_0(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b + GGUF_TQ2_0_D_OFFSET);
    for (int i = 0; i < 256; i++)
        out[i] = d * (float)gguf_tq2_0_trit(b, i);
}

__device__ void dq_tq1_0(const unsigned char* b, float* out) {
    float d = load_f16_as_f32(b + GGUF_TQ1_0_D_OFFSET);
    for (int i = 0; i < 256; i++)
        out[i] = d * (float)gguf_tq1_0_trit(b, i);
}

// The seven IQ formats are codebook quantizations; their block layouts live
// once in iq_dequant.cuh, shared with the dequant and GEMV/GEMM paths.

__device__ void dq_iq2_xxs(const unsigned char* b, float* o) { iq2_xxs_dequant_block(b, o); }
__device__ void dq_iq2_xs (const unsigned char* b, float* o) { iq2_xs_dequant_block(b, o);  }
__device__ void dq_iq2_s  (const unsigned char* b, float* o) { iq2_s_dequant_block(b, o);   }
__device__ void dq_iq3_xxs(const unsigned char* b, float* o) { iq3_xxs_dequant_block(b, o); }
__device__ void dq_iq3_s  (const unsigned char* b, float* o) { iq3_s_dequant_block(b, o);   }
__device__ void dq_iq1_s  (const unsigned char* b, float* o) { iq1_s_dequant_block(b, o);   }
__device__ void dq_iq1_m  (const unsigned char* b, float* o) { iq1_m_dequant_block(b, o);   }

// ── Dequant one block into buffer, dispatching by format ─────────────

__device__ int get_bs(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0: case FMT_Q4_1: case FMT_Q5_0: case FMT_Q5_1:
        case FMT_Q8_0: case FMT_Q8_1: case FMT_IQ4NL:
            return 32;
        default: return 256;
    }
}

__device__ int get_bb(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0:   return 18;  case FMT_Q4_1:   return 20;
        case FMT_Q5_0:   return 22;  case FMT_Q5_1:   return 24;
        case FMT_Q8_0:   return 34;  case FMT_Q8_1:   return 36;
        case FMT_Q2K:    return 84;  case FMT_Q3K:    return 110;
        case FMT_Q4K:    return 144; case FMT_Q5K:    return 176;
        case FMT_Q6K:    return 210; case FMT_Q8K:    return 292;
        case FMT_IQ1S:   return 50;  case FMT_IQ1M:   return 56;
        case FMT_IQ2XXS: return 66;  case FMT_IQ2XS:  return 74;
        case FMT_IQ2S:   return 82;  case FMT_IQ3XXS: return 98;
        case FMT_IQ3S:   return 110; case FMT_IQ4NL:  return 18;
        case FMT_IQ4XS:  return 136; case FMT_TQ1_0:  return 54;
        case FMT_TQ2_0:  return 66;  default:         return 0;
    }
}

__device__ void dequant_block(const unsigned char* b, float* out, unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0:   dq_q4_0(b, out); break;
        case FMT_Q4_1:   dq_q4_1(b, out); break;
        case FMT_Q5_0:   dq_q5_0(b, out); break;
        case FMT_Q5_1:   dq_q5_1(b, out); break;
        case FMT_Q8_0:   dq_q8_0(b, out); break;
        case FMT_Q8_1:   dq_q8_1(b, out); break;
        case FMT_Q2K:    dq_q2k(b, out); break;
        case FMT_Q3K:    dq_q3k(b, out); break;
        case FMT_Q4K:    dq_q4k(b, out); break;
        case FMT_Q5K:    dq_q5k(b, out); break;
        case FMT_Q6K:    dq_q6k(b, out); break;
        case FMT_Q8K:    dq_q8k(b, out); break;
        case FMT_IQ4NL:  dq_iq4_nl(b, out); break;
        case FMT_IQ4XS:  dq_iq4_xs(b, out); break;
        case FMT_IQ2XXS: dq_iq2_xxs(b, out); break;
        case FMT_IQ2XS:  dq_iq2_xs(b, out); break;
        case FMT_IQ2S:   dq_iq2_s(b, out); break;
        case FMT_IQ3XXS: dq_iq3_xxs(b, out); break;
        case FMT_IQ3S:   dq_iq3_s(b, out); break;
        case FMT_IQ1S:   dq_iq1_s(b, out); break;
        case FMT_IQ1M:   dq_iq1_m(b, out); break;
        case FMT_TQ1_0:  dq_tq1_0(b, out); break;
        case FMT_TQ2_0:  dq_tq2_0(b, out); break;
    }
}

// ── Fused dequant+dot functions (no buffer) ─────────────────────────
// These compute dot(dequant(block), activation) without materializing
// the full dequantized output, eliminating the 256-float register spill.

__device__ float dq_dot_q2k(const unsigned char* b, const float* act) {
    const unsigned char* sc = b;
    const unsigned char* qs = b + 16;
    float d    = load_f16_as_f32(b+80);
    float dmin = load_f16_as_f32(b+82);
    float sum = 0.0f;
    int y = 0, is = 0;
    for (int n = 0; n < 2; n++) {
        const unsigned char* q = qs + n * 32;
        for (int shift = 0; shift < 8; shift += 2) {
            float dl = d * (float)(sc[is] & 0x0F);
            float ml = dmin * (float)(sc[is] >> 4);
            is++;
            for (int l = 0; l < 16; l++)
                sum += act[y++] * (dl * (float)((q[l] >> shift) & 3) - ml);
            dl = d * (float)(sc[is] & 0x0F);
            ml = dmin * (float)(sc[is] >> 4);
            is++;
            for (int l = 0; l < 16; l++)
                sum += act[y++] * (dl * (float)((q[16+l] >> shift) & 3) - ml);
        }
    }
    return sum;
}

__device__ float dq_dot_q3k(const unsigned char* b, const float* act) {
    const unsigned char* hmask = b;
    const unsigned char* qs = b + 32;
    const unsigned char* sc_raw = b + 96;
    float d = load_f16_as_f32(b+108);

    unsigned int aux[4];
    unsigned char aux_bytes[12];
    memcpy(aux_bytes, sc_raw, 12);
    memcpy(&aux[0], aux_bytes, 4);
    memcpy(&aux[1], aux_bytes + 4, 4);
    memcpy(&aux[2], aux_bytes + 8, 4);
    unsigned int tmp = aux[2];
    const unsigned int M1 = 0x03030303u, M2 = 0x0f0f0f0fu;
    unsigned int a0 = aux[0], a1 = aux[1];
    aux[0] = (a0 & M2) | ((tmp & M1) << 4);
    aux[1] = (a1 & M2) | (((tmp>>2) & M1) << 4);
    aux[2] = ((a0>>4) & M2) | (((tmp>>4) & M1) << 4);
    aux[3] = ((a1>>4) & M2) | (((tmp>>6) & M1) << 4);
    signed char scales[16];
    memcpy(&scales[0],  &aux[0], 4);
    memcpy(&scales[4],  &aux[1], 4);
    memcpy(&scales[8],  &aux[2], 4);
    memcpy(&scales[12], &aux[3], 4);
    for (int i = 0; i < 16; i++) scales[i] = (signed char)((unsigned char)scales[i] - 32);

    float sum = 0.0f;
    int y = 0, is_idx = 0;
    unsigned char m = 1;
    for (int n = 0; n < 2; n++) {
        const unsigned char* q = qs + n * 32;
        for (int shift = 0; shift < 8; shift += 2) {
            float dl = d * (float)scales[is_idx++];
            for (int l = 0; l < 16; l++) {
                int low2 = (q[l] >> shift) & 3;
                int hsub = (hmask[l] & m) ? 0 : 4;
                sum += act[y++] * (dl * (float)(low2 - hsub));
            }
            dl = d * (float)scales[is_idx++];
            for (int l = 0; l < 16; l++) {
                int low2 = (q[16+l] >> shift) & 3;
                int hsub = (hmask[16+l] & m) ? 0 : 4;
                sum += act[y++] * (dl * (float)(low2 - hsub));
            }
            m <<= 1;
        }
    }
    return sum;
}

// Generic fallback: uses buffer for formats without fused dq_dot
__device__ float dq_dot_generic(const unsigned char* block, const float* act,
                                 int block_size, unsigned int fmt) {
    float buf[256];
    dequant_block(block, buf, fmt);
    float sum = 0.0f;
    for (int i = 0; i < block_size; i++)
        sum += act[i] * buf[i];
    return sum;
}

// Dispatch: use fused dq_dot for K-quants, generic for others
__device__ float dq_dot_dispatch(const unsigned char* block, const float* act,
                                  int block_size, unsigned int fmt) {
    switch (fmt) {
        case FMT_Q2K:  return dq_dot_q2k(block, act);
        case FMT_Q3K:  return dq_dot_q3k(block, act);
        default:       return dq_dot_generic(block, act, block_size, fmt);
    }
}

// ── Fused dequant + GEMV kernel ──────────────────────────────────────
//
// Grid:  (N, M, 1)  — one thread block per output element
// Block: (32, 1, 1) — one warp cooperates on K reduction
//
// For K-quants with fused dq_dot: no intermediate buffer, dequant and
// dot product happen in the same loop. Eliminates 256-float register spill.

extern "C" __global__ void quant_matmul_generic_f32(
    const float* __restrict__ activation,  // [M, K]
    const unsigned char* __restrict__ weight,  // [N, K] packed
    float* __restrict__ output,            // [M, N]
    unsigned int M, unsigned int K, unsigned int N,
    unsigned int format_id
) {
    const unsigned int col = blockIdx.x;  // output column (weight row)
    const unsigned int row = blockIdx.y;  // activation row
    const unsigned int lane = threadIdx.x;
    if (col >= N || row >= M) return;

    const int block_size = get_bs(format_id);
    const int block_bytes = get_bb(format_id);
    if (block_bytes == 0) return;

    const unsigned int blocks_per_row = K / block_size;
    const unsigned long long row_bytes = (unsigned long long)blocks_per_row * block_bytes;
    const float* act_row = activation + (unsigned long long)row * K;
    const unsigned char* w_row = weight + (unsigned long long)col * row_bytes;

    float acc = 0.0f;

    // Each thread processes a subset of blocks, then the warp reduces
    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const float* act_base = act_row + b * block_size;
        acc += dq_dot_dispatch(
            w_row + (unsigned long long)b * block_bytes,
            act_base, block_size, format_id);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0) {
        output[(unsigned long long)row * N + col] = acc;
    }
}

// ── Fused SwiGLU: gate_matmul + up_matmul + silu(gate)*up ──────────
//
// Grid:  (N, M, 1)  — one thread block per output element
// Block: (32, 1, 1) — one warp cooperates on K reduction
//
// Computes: output[row, col] = silu(dot(act, gate_w[col])) * dot(act, up_w[col])
// Eliminates 2 intermediate [M,N] tensors and 1 extra kernel launch.

__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void quant_swiglu_generic_f32(
    const float* __restrict__ activation,       // [M, K]
    const unsigned char* __restrict__ gate_w,    // [N, K] packed
    const unsigned char* __restrict__ up_w,      // [N, K] packed
    float* __restrict__ output,                  // [M, N]
    unsigned int M, unsigned int K, unsigned int N,
    unsigned int format_id
) {
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (col >= N || row >= M) return;

    const int block_size = get_bs(format_id);
    const int block_bytes = get_bb(format_id);
    if (block_bytes == 0) return;

    const unsigned int blocks_per_row = K / block_size;
    const unsigned long long row_bytes = (unsigned long long)blocks_per_row * block_bytes;
    const float* act_row = activation + (unsigned long long)row * K;
    const unsigned char* g_row = gate_w + (unsigned long long)col * row_bytes;
    const unsigned char* u_row = up_w   + (unsigned long long)col * row_bytes;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        const float* act_base = act_row + b * block_size;
        unsigned long long byte_off = (unsigned long long)b * block_bytes;
        gate_acc += dq_dot_dispatch(g_row + byte_off, act_base, block_size, format_id);
        up_acc   += dq_dot_dispatch(u_row + byte_off, act_base, block_size, format_id);
    }

    // Warp reduction for both accumulators
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        gate_acc += __shfl_down_sync(0xFFFFFFFF, gate_acc, offset);
        up_acc   += __shfl_down_sync(0xFFFFFFFF, up_acc, offset);
    }

    if (lane == 0) {
        output[(unsigned long long)row * N + col] = silu(gate_acc) * up_acc;
    }
}
