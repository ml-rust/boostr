// Fused quantized matmul CUDA kernels for boostr
// Computes: activation [M, K] × weight[N, K]^T → output [M, N]
//
// Each thread computes one output element by dequantizing-and-accumulating
// the weight row on the fly. This avoids materializing the full dequantized weight.
//
// Supports: Q4_0, Q8_0, Q4_K, Q6_K weights × f32 activations → f32 output

#include <cuda_fp16.h>

#define WARP_SIZE 32

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
// Q4_K × f32 → f32  (output-tiled GEMM, 16×16 tile per block)
//
// Weight block: 256 elements, 144 bytes (8 sub-blocks of 32 elements each).
// Layout: 2-byte d (half), 2-byte dmin (half), 12-byte scales/mins, 128-byte qs.
//
// Block: 128 threads (4 warps), each thread owns 2 output elements.
//   thread t → output (tm, tn0) and (tm, tn1) where:
//     tm  = t % TM          (row within tile)
//     tn0 = (t / TM) * 2    (first col within tile)
//     tn1 = tn0 + 1         (second col within tile)
//
// Grid: (ceil(N/TN), ceil(M/TM), 1)
//   For M=2000, N=4096: (256, 125, 1) = 32,000 blocks  (vs 8M before)
//
// Each block loads the activation tile [TM × 256] into shared memory once
// per K-block, then all 128 threads read from smem for their weight rows.
// No inter-warp K-reduction needed: each thread independently accumulates
// all K-blocks for its two output elements.
//
// Scales/mins unpacking: 12-byte packed field at offset 4.
// Per-sub-block dequant factors (dl, ml) hoisted outside the element loop.
// ============================================================================

#define TM 16
#define TN 16

__global__ __launch_bounds__(128, 2) void quant_matmul_q4_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    // Tile origin in the output matrix
    const unsigned int tile_row = blockIdx.y * TM;
    const unsigned int tile_col = blockIdx.x * TN;

    // This thread's two output positions within the tile
    const int t = threadIdx.x;
    const int tm  = t % TM;           // row within tile [0, TM)
    const int tn0 = (t / TM) * 2;     // first col within tile
    const int tn1 = tn0 + 1;          // second col within tile

    // Absolute output row/col
    const unsigned int out_row = tile_row + tm;
    const unsigned int out_col0 = tile_col + tn0;
    const unsigned int out_col1 = tile_col + tn1;

    const bool valid_row  = (out_row  < M);
    const bool valid_col0 = (out_col0 < N);
    const bool valid_col1 = (out_col1 < N);

    const int blocks_per_row = (int)(K / 256);
    const unsigned long long weight_row_bytes = (unsigned long long)blocks_per_row * 144;

    // Pointers to weight rows for the two columns this thread owns
    const unsigned char* w_row0 = valid_col0 ? (weight + (unsigned long long)out_col0 * weight_row_bytes) : nullptr;
    const unsigned char* w_row1 = valid_col1 ? (weight + (unsigned long long)out_col1 * weight_row_bytes) : nullptr;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Shared memory for the activation tile: TM rows × 256 K-elements
    // 16 × 256 × 4 bytes = 16 KB
    __shared__ float act_smem[TM][256];

    for (int b = 0; b < blocks_per_row; b++) {
        // Cooperatively load activation tile into shared memory.
        // 128 threads load TM*256 = 4096 floats total, 32 floats each.
        const unsigned int act_base = (unsigned int)b * 256;
        #pragma unroll 4
        for (int i = t; i < TM * 256; i += 128) {
            int act_r = i / 256;
            int act_c = i % 256;
            unsigned int global_row = tile_row + act_r;
            act_smem[act_r][act_c] = (global_row < M)
                ? activation[(unsigned long long)global_row * K + act_base + act_c]
                : 0.0f;
        }
        __syncthreads();

        // Decode weight block for col0 and col1
        // Q4_K block layout (144 bytes):
        //   [0..1]   d    (half scalar)
        //   [2..3]   dmin (half scalar)
        //   [4..15]  12 bytes packed scales+mins (6-bit each, 8 scales + 8 mins)
        //   [16..143] 128 bytes qs (nibble-packed, 256 nibbles = 256 elements)

        float d0 = 0.0f, dmin0 = 0.0f;
        float d1 = 0.0f, dmin1 = 0.0f;
        unsigned char scales0[8], mins0[8];
        unsigned char scales1[8], mins1[8];
        const unsigned char* qs0 = nullptr;
        const unsigned char* qs1 = nullptr;

        if (valid_col0) {
            const unsigned char* blk0 = w_row0 + (unsigned long long)b * 144;
            d0    = __half2float(*reinterpret_cast<const __half*>(blk0));
            dmin0 = __half2float(*reinterpret_cast<const __half*>(blk0 + 2));
            const unsigned char* sc0 = blk0 + 4;
            for (int i = 0; i < 4; i++) {
                scales0[i] = sc0[i] & 0x3F;
                mins0[i]   = sc0[i + 4] & 0x3F;
            }
            for (int i = 4; i < 8; i++) {
                scales0[i] = (sc0[i + 4] & 0x0F) | ((sc0[i - 4] >> 6) << 4);
                mins0[i]   = (sc0[i + 4] >> 4)   | ((sc0[i]     >> 6) << 4);
            }
            qs0 = blk0 + 16;
        }

        if (valid_col1) {
            const unsigned char* blk1 = w_row1 + (unsigned long long)b * 144;
            d1    = __half2float(*reinterpret_cast<const __half*>(blk1));
            dmin1 = __half2float(*reinterpret_cast<const __half*>(blk1 + 2));
            const unsigned char* sc1 = blk1 + 4;
            for (int i = 0; i < 4; i++) {
                scales1[i] = sc1[i] & 0x3F;
                mins1[i]   = sc1[i + 4] & 0x3F;
            }
            for (int i = 4; i < 8; i++) {
                scales1[i] = (sc1[i + 4] & 0x0F) | ((sc1[i - 4] >> 6) << 4);
                mins1[i]   = (sc1[i + 4] >> 4)   | ((sc1[i]     >> 6) << 4);
            }
            qs1 = blk1 + 16;
        }

        // Each thread accumulates all 8 sub-blocks for its row (tm) and
        // two weight columns (col0, col1).
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int chunk   = j >> 1;
            int is_high = j &  1;

            if (valid_col0) {
                float dl0 = d0    * (float)scales0[j];
                float ml0 = dmin0 * (float)mins0[j];
                for (int lane = 0; lane < 32; lane++) {
                    float q = is_high
                        ? (float)((qs0[chunk * 32 + lane] >> 4) & 0x0F)
                        : (float)( qs0[chunk * 32 + lane]       & 0x0F);
                    acc0 += act_smem[tm][j * 32 + lane] * (dl0 * q - ml0);
                }
            }

            if (valid_col1) {
                float dl1 = d1    * (float)scales1[j];
                float ml1 = dmin1 * (float)mins1[j];
                for (int lane = 0; lane < 32; lane++) {
                    float q = is_high
                        ? (float)((qs1[chunk * 32 + lane] >> 4) & 0x0F)
                        : (float)( qs1[chunk * 32 + lane]       & 0x0F);
                    acc1 += act_smem[tm][j * 32 + lane] * (dl1 * q - ml1);
                }
            }
        }

        __syncthreads();
    }

    if (valid_row && valid_col0) output[(unsigned long long)out_row * N + out_col0] = acc0;
    if (valid_row && valid_col1) output[(unsigned long long)out_row * N + out_col1] = acc1;
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
