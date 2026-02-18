// Fused quantized matmul CUDA kernels for boostr
// Computes: activation [M, K] × weight[N, K]^T → output [M, N]
//
// Each thread computes one output element by dequantizing-and-accumulating
// the weight row on the fly. This avoids materializing the full dequantized weight.
//
// Supports: Q4_0, Q8_0, Q4_K, Q6_K weights × f32 activations → f32 output

#include <cuda_fp16.h>

extern "C" {

// ============================================================================
// Q4_0 × f32 → f32
// Weight block: 32 elements, 18 bytes
// ============================================================================

__global__ void quant_matmul_q4_0_f32(
    const float* __restrict__ activation,  // [M, K]
    const unsigned char* __restrict__ weight, // [N, K] as Q4_0 blocks
    float* __restrict__ output,            // [M, N]
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 18;

    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 18;
        __half d_half = *reinterpret_cast<const __half*>(block);
        float d = __half2float(d_half);
        const unsigned char* qs = block + 2;
        unsigned int base = b * 32;

        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            float low = (float)((int)(byte & 0x0F) - 8) * d;
            float high = (float)((int)((byte >> 4) & 0x0F) - 8) * d;
            sum += act_row[base + i * 2] * low;
            sum += act_row[base + i * 2 + 1] * high;
        }
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Q8_0 × f32 → f32
// Weight block: 32 elements, 34 bytes
// ============================================================================

__global__ void quant_matmul_q8_0_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int row_bytes = blocks_per_row * 34;

    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 34;
        __half d_half = *reinterpret_cast<const __half*>(block);
        float d = __half2float(d_half);
        const signed char* qs = reinterpret_cast<const signed char*>(block + 2);
        unsigned int base = b * 32;

        for (int i = 0; i < 32; i++) {
            sum += act_row[base + i] * ((float)qs[i] * d);
        }
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Q4_K × f32 → f32
// Weight block: 256 elements, 144 bytes
// ============================================================================

__global__ void quant_matmul_q4_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 144;

    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 144;
        __half d_half = *reinterpret_cast<const __half*>(block);
        __half dmin_half = *reinterpret_cast<const __half*>(block + 2);
        float d = __half2float(d_half);
        float dmin = __half2float(dmin_half);
        const unsigned char* sc = block + 4;
        const unsigned char* qs = block + 16;
        unsigned int base = b * 256;

        // Unpack 6-bit scales and mins
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
                float w = dl * q - ml;
                sum += act_row[base + j * 32 + l] * w;
            }
        }
    }

    output[row * N + col] = sum;
}

// ============================================================================
// Q6_K × f32 → f32
// Weight block: 256 elements, 210 bytes
// ============================================================================

__global__ void quant_matmul_q6_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int blocks_per_row = K / 256;
    unsigned int row_bytes = blocks_per_row * 210;

    const float* act_row = activation + row * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 210;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* sc = reinterpret_cast<const signed char*>(block + 192);
        __half d_half = *reinterpret_cast<const __half*>(block + 208);
        float d = __half2float(d_half);
        unsigned int base = b * 256;

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

                float w1 = d * (float)sc[sc_base + is]     * (float)q1;
                float w2 = d * (float)sc[sc_base + is + 2] * (float)q2;
                float w3 = d * (float)sc[sc_base + is + 4] * (float)q3;
                float w4 = d * (float)sc[sc_base + is + 6] * (float)q4;

                sum += act_row[base + y_base + l]      * w1;
                sum += act_row[base + y_base + l + 32] * w2;
                sum += act_row[base + y_base + l + 64] * w3;
                sum += act_row[base + y_base + l + 96] * w4;
            }
        }
    }

    output[row * N + col] = sum;
}

} // extern "C"
