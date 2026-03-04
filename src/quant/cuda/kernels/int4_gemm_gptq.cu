// GPTQ INT4 GEMM with g_idx permutation
// GPTQ layout: qweight[K/8, N] sequential 4-bit packing, g_idx column permutation
// Dequant: w = (q - (qzero+1)) * scale

#include <stdint.h>

#define WARP_SIZE 32

__device__ __forceinline__ int unpack_int4_seq(uint32_t packed, int idx) {
    return (packed >> (idx * 4)) & 0xF;
}

// ============================================================================
// GEMV kernel — optimized for GPTQ decode (M<=4)
//
// Layout: qweight[K/8, N] — each u32 has 8 K-values for one output column.
// Challenge: g_idx causes random access to scales/qzeros per K-value.
//
// Strategy:
// - 128 threads per block, each thread owns one output column
// - Tile over K in chunks of 128 (= thread count, for efficient shared mem loading)
// - Load g_idx[K-tile] and input[K-tile] into shared memory cooperatively
// - Each thread dequantizes its own column's weights using shared g_idx/input
// - atomicAdd for cross-K-tile accumulation
// - Output MUST be pre-zeroed
//
// Grid: (ceil(K/128), ceil(N/128), M), Block: (128, 1, 1)
// ============================================================================
#define GEMV_BLOCK 128
#define GEMV_KTILE 128

extern "C" __global__ __launch_bounds__(128, 4) void int4_gemv_gptq_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const uint32_t* __restrict__ qzeros,  // [num_groups, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const int* __restrict__ g_idx,        // [K]
    float* __restrict__ output,           // [M, N]  -- MUST be pre-zeroed
    uint32_t M, uint32_t K, uint32_t N
) {
    const uint32_t col = GEMV_BLOCK * blockIdx.y + threadIdx.x;
    const uint32_t k_start = GEMV_KTILE * blockIdx.x;
    const uint32_t m = blockIdx.z;

    // Shared memory for cooperative loading of g_idx and input vector
    __shared__ int smem_gidx[GEMV_KTILE];
    __shared__ float smem_input[GEMV_KTILE];

    // Cooperatively load g_idx and input into shared memory
    // 128 threads loading 128 elements = 1 element per thread
    {
        uint32_t ki = k_start + threadIdx.x;
        if (ki < K) {
            smem_gidx[threadIdx.x] = g_idx[ki];
            smem_input[threadIdx.x] = input[m * K + ki];
        } else {
            smem_gidx[threadIdx.x] = 0;
            smem_input[threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    if (col >= N) return;

    const uint32_t n_packed_zeros = N / 8;
    const uint32_t z_w = col / 8;
    const uint32_t z_mod = (col % 8) * 4;
    const uint32_t h_packed = k_start / 8;

    float res = 0.0f;

    // Process K-tile: 128 K-values = 16 packed rows × 8 sub-elements
    // Cache the previous group's scale/zero to avoid redundant loads
    int prev_group = -1;
    float cached_scale = 0.0f;
    int cached_qzero = 0;

    #pragma unroll 4
    for (int k = 0; k < GEMV_KTILE; ++k) {
        uint32_t ki = k_start + k;
        if (ki >= K) break;

        float a = smem_input[k];
        int g = smem_gidx[k];

        // Only reload scale/qzero when group changes
        if (g != prev_group) {
            prev_group = g;
            cached_scale = scales[g * N + col];
            uint32_t zero_pack = qzeros[g * n_packed_zeros + z_w];
            cached_qzero = ((zero_pack >> z_mod) & 0xF) + 1;
        }

        int k_w = k / 8;
        int k_bit = (k % 8) * 4;
        uint32_t packed = qweight[(h_packed + k_w) * N + col];
        int q = (packed >> k_bit) & 0xF;

        res += a * cached_scale * ((float)q - (float)cached_qzero);
    }

    atomicAdd(&output[m * N + col], res);
}

// ============================================================================
// Tiled GEMM kernel — for prefill (M > 4)
// BM=32, BN=32, BK=32. Shared memory for input tile.
// Grid: (ceil(N/32), ceil(M/32)), Block: (32, 4)
// Each thread computes 8 output rows for 1 column.
// ============================================================================
#define BM 32
#define BN 32
#define BK 32

extern "C" __global__ void int4_gemm_gptq_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const uint32_t* __restrict__ qzeros,  // [num_groups, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const int* __restrict__ g_idx,        // [K]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N
) {
    const uint32_t tx = threadIdx.x;  // 0..31 (column within tile)
    const uint32_t ty = threadIdx.y;  // 0..3 (row group)

    const uint32_t block_row = blockIdx.y * BM;
    const uint32_t block_col = blockIdx.x * BN;

    const uint32_t col = block_col + tx;
    const uint32_t n_packed_zeros = N / 8;
    const uint32_t zero_col_pack = col / 8;
    const uint32_t zero_sub = col % 8;

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    __shared__ float smem_input[BM][BK];

    for (uint32_t bk = 0; bk < K; bk += BK) {
        // Cooperatively load input tile
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t row = ty * 8 + i;
            uint32_t global_row = block_row + row;
            uint32_t global_k = bk + tx;
            if (global_row < M && global_k < K)
                smem_input[row][tx] = input[global_row * K + global_k];
            else
                smem_input[row][tx] = 0.0f;
        }
        __syncthreads();

        if (col < N) {
            for (uint32_t dk = 0; dk < BK && (bk + dk) < K; dk++) {
                uint32_t ki = bk + dk;
                uint32_t pack_row = ki / 8;
                uint32_t sub_idx = ki % 8;
                uint32_t packed = qweight[pack_row * N + col];
                int q = unpack_int4_seq(packed, sub_idx);

                int group = g_idx[ki];
                uint32_t zero_pack = qzeros[group * n_packed_zeros + zero_col_pack];
                int qzero = unpack_int4_seq(zero_pack, zero_sub) + 1;
                float scale = scales[group * N + col];
                float w = ((float)q - (float)qzero) * scale;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    acc[i] += smem_input[ty * 8 + i][dk] * w;
                }
            }
        }
        __syncthreads();
    }

    if (col < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t global_row = block_row + ty * 8 + i;
            if (global_row < M) {
                output[global_row * N + col] = acc[i];
            }
        }
    }
}
