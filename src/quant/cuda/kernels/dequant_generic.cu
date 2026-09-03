// Generic dequantization CUDA kernel — fallback for all GGUF quant formats
// that lack dedicated optimized kernels.
//
// One thread per quant block. The format_id parameter selects the decode path
// via a switch statement. Not optimal, but correct for all 23 formats.
//
// Callers must prefer the optimized kernels in dequant.cu (Q4_0, Q8_0,
// Q4_K, Q6_K) when available; this is the catch-all fallback.
//
// IMPORTANT: All multi-byte loads from quant block data use memcpy to avoid
// misaligned access errors. Quant blocks are packed contiguously and their
// internal fields are not always naturally aligned.

#include <cuda_fp16.h>

#include "decode.cuh"
#include "iq_dequant.cuh"

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
// Quant blocks are packed contiguously; internal fields are not always
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

// ── Simple quant device functions ────────────────────────────────────

// Split-half nibble order, shared by every 4-bit format below: llama.cpp
// `dequantize_row_q4_0` writes the LOW nibble of qs[j] to element j and the
// HIGH nibble of the SAME byte to element j + 16. The two nibbles are 16
// elements apart, never adjacent. Q5_0/Q5_1 index the fifth-bit word `qh` with
// the same element index — bit j and bit j + 16.

__device__ void dequant_q4_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    const unsigned char* qs = block + 2;
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        out[j]      = (float)((int)(byte & 0x0F) - 8) * d;
        out[j + 16] = (float)((int)((byte >> 4) & 0x0F) - 8) * d;
    }
}

__device__ void dequant_q4_1_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    float m = load_f16_as_f32(block + 2);
    const unsigned char* qs = block + 4;
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        out[j]      = d * (float)(byte & 0x0F) + m;
        out[j + 16] = d * (float)((byte >> 4) & 0x0F) + m;
    }
}

__device__ void dequant_q5_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    unsigned int qh = load_u32(block + 2);
    const unsigned char* qs = block + 6;
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low  = (byte & 0x0F) | (((qh >> j) & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = (float)(low - 16) * d;
        out[j + 16] = (float)(high - 16) * d;
    }
}

__device__ void dequant_q5_1_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block);
    float m = load_f16_as_f32(block + 2);
    unsigned int qh = load_u32(block + 4);
    const unsigned char* qs = block + 8;
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low  = (byte & 0x0F) | (((qh >> j) & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = d * (float)low + m;
        out[j + 16] = d * (float)high + m;
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
    // block[2..4] is `s`, a precomputed dot-product sum in llama.cpp's
    // `block_q8_1` — NOT a min. Dequant is q * d and must ignore it.
    const signed char* qs = reinterpret_cast<const signed char*>(block + 4);
    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d;
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

    // Unpack 16 6-bit scales from 12 bytes — byte-by-byte to avoid unaligned access
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
        // llama.cpp `dequantize_row_q5_K`: sub-block PAIRS share one 32-byte run
        // of `qs` (even sub-block low nibbles, odd sub-block high nibbles of the
        // SAME bytes), and `qh` is indexed by ELEMENT with the BIT selected by
        // the sub-block index — not a flat bitstream over the 256 values.
        int qs_base = (j / 2) * 32;
        int is_high_nibble = j % 2;
        for (int l = 0; l < 32; l++) {
            int low4 = is_high_nibble ? ((qs[qs_base + l] >> 4) & 0x0F)
                                      : (qs[qs_base + l] & 0x0F);
            int high1 = (qh[l] >> j) & 0x01;
            float q = (float)(low4 | (high1 << 4));
            out[j * 32 + l] = dl * q - ml;
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
    // Split-half nibble order (llama.cpp `dequantize_row_iq4_nl`).
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        out[j]      = d * (float)KVALUES_IQ4NL[byte & 0x0F];
        out[j + 16] = d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
    }
}

__device__ void dequant_iq4_xs_block(const unsigned char* block, float* out) {
    // llama.cpp `block_iq4_xs`:
    //   { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
    // scales_h is TWO bytes at offset 2, so scales_l starts at 4 (no pad byte)
    // and scales_h supplies high scale bits for all EIGHT sub-blocks.
    float d = load_f16_as_f32(block);
    unsigned int scales_h = (unsigned int)load_u16(block + 2);
    const unsigned char* scales_l = block + 4;
    const unsigned char* qs = block + 8;

    for (int sb = 0; sb < 8; sb++) {
        int sl = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0x0F;
        int sh = (scales_h >> (2 * sb)) & 0x03;
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

__device__ void dequant_tq2_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block + GGUF_TQ2_0_D_OFFSET);
    for (int i = 0; i < 256; i++) {
        out[i] = d * (float)gguf_tq2_0_trit(block, i);
    }
}

__device__ void dequant_tq1_0_block(const unsigned char* block, float* out) {
    float d = load_f16_as_f32(block + GGUF_TQ1_0_D_OFFSET);
    for (int i = 0; i < 256; i++) {
        out[i] = d * (float)gguf_tq1_0_trit(block, i);
    }
}

// The seven IQ formats are codebook quantizations; their block layouts live
// once in iq_dequant.cuh, shared with the quant-matmul and GEMV/GEMM paths.

__device__ void dequant_iq2_xxs_block(const unsigned char* b, float* o) { iq2_xxs_dequant_block(b, o); }
__device__ void dequant_iq2_xs_block (const unsigned char* b, float* o) { iq2_xs_dequant_block(b, o);  }
__device__ void dequant_iq2_s_block  (const unsigned char* b, float* o) { iq2_s_dequant_block(b, o);   }
__device__ void dequant_iq3_xxs_block(const unsigned char* b, float* o) { iq3_xxs_dequant_block(b, o); }
__device__ void dequant_iq3_s_block  (const unsigned char* b, float* o) { iq3_s_dequant_block(b, o);   }
__device__ void dequant_iq1_s_block  (const unsigned char* b, float* o) { iq1_s_dequant_block(b, o);   }
__device__ void dequant_iq1_m_block  (const unsigned char* b, float* o) { iq1_m_dequant_block(b, o);   }

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
