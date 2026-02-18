// Dequantization CUDA kernels for boostr
// Supports: Q4_0, Q8_0, Q4_K, Q6_K → f32
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

} // extern "C"
