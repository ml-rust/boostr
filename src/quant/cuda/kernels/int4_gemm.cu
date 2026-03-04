// AWQ INT4 GEMM kernels
// AWQ packing: 8 INT4 per u32, shifts [0,16,4,20,8,24,12,28]
// Dequant: w = (q - zero) * scale

#include <stdint.h>

#define WARP_SIZE 32

// AWQ packs with order [0,2,4,6,1,3,5,7], reverse mapping:
__constant__ int AWQ_SHIFTS[8] = {0, 16, 4, 20, 8, 24, 12, 28};

__device__ __forceinline__ int unpack_int4_awq(uint32_t packed, int idx) {
    return (packed >> AWQ_SHIFTS[idx]) & 0xF;
}

// ============================================================================
// GEMV kernel — optimized for decode (M=1 typically)
//
// Key optimization: each warp processes 8 output columns from ONE packed u32.
// This means every qweight load yields 8 useful values instead of 1.
// 4 warps per block = 32 output columns per block.
// Shared memory caches input row for reuse across warps.
// Scale/zero cached per group (only changes every group_size iterations).
//
// Grid: (ceil(N/32), M, 1), Block: (128, 1, 1)
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 1) void int4_gemv_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const float* __restrict__ zeros,      // [num_groups, N]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t group_size
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t m = blockIdx.y;

    // Each warp handles 8 consecutive output columns (one packed u32 worth)
    // 4 warps per block = 32 columns per block
    const uint32_t pack_col = blockIdx.x * 4 + warp_id;
    const uint32_t n_packed = N / 8;

    if (pack_col >= n_packed) return;

    const uint32_t base_col = pack_col * 8;
    const float* act_row = input + m * K;

    // 8 accumulators, one per output column in this packed group
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Cache scale/zero per group (only reload every group_size iterations)
    uint32_t prev_group = 0xFFFFFFFF;
    float cached_scale[8];
    float cached_zero[8];

    for (uint32_t ki = lane_id; ki < K; ki += WARP_SIZE) {
        float a = act_row[ki];

        // Load and unpack one packed u32 → 8 INT4 values
        uint32_t packed = qweight[ki * n_packed + pack_col];

        // Cache scale/zero per group
        uint32_t group = ki / group_size;
        if (group != prev_group) {
            prev_group = group;
            const float* gs = scales + group * N + base_col;
            const float* gz = zeros + group * N + base_col;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                cached_scale[j] = gs[j];
                cached_zero[j] = gz[j];
            }
        }

        // Dequant and accumulate all 8 columns
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int q = unpack_int4_awq(packed, j);
            float w = ((float)q - cached_zero[j]) * cached_scale[j];
            acc[j] += a * w;
        }
    }

    // Warp reduction for each of the 8 accumulators
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc[j] += __shfl_down_sync(0xFFFFFFFF, acc[j], offset);
    }

    // Lane 0 writes 8 output columns
    if (lane_id == 0) {
        float* out_base = output + m * N + base_col;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (base_col + j < N)
                out_base[j] = acc[j];
        }
    }
}

// ============================================================================
// Tiled GEMM kernel — for prefill (M > 4)
// Shared memory tiling: BM×BK input tile + BK×BN weight tile
// BM=32, BN=32, BK=32 (BK aligned to group_size for scale/zero reuse)
// Grid: (ceil(N/32), ceil(M/32)), Block: (32, 4)
// Each thread computes 8 output elements (loop over BN/threads_x unrolled)
// ============================================================================
#define BM 32
#define BN 32
#define BK 32

extern "C" __global__ void int4_gemm_f32(
    const float* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ qweight, // [K, N/8]
    const float* __restrict__ scales,     // [num_groups, N]
    const float* __restrict__ zeros,      // [num_groups, N]
    float* __restrict__ output,           // [M, N]
    uint32_t M, uint32_t K, uint32_t N, uint32_t group_size
) {
    // Thread indices
    const uint32_t tx = threadIdx.x;  // 0..31 (column within tile)
    const uint32_t ty = threadIdx.y;  // 0..3 (row group)

    // Block output position
    const uint32_t block_row = blockIdx.y * BM;
    const uint32_t block_col = blockIdx.x * BN;

    // Each thread handles 8 rows (BM/4 = 8 rows per ty)
    const uint32_t n_packed = N / 8;

    // Accumulators: 8 rows × 1 col per thread
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    // Shared memory for input tile [BM, BK]
    __shared__ float smem_input[BM][BK];

    const uint32_t col = block_col + tx;
    const uint32_t pack_col = col / 8;
    const uint32_t sub = col % 8;

    // Tile over K dimension
    for (uint32_t bk = 0; bk < K; bk += BK) {
        // Cooperatively load input tile [BM, BK] into shared memory
        // 128 threads load 32×32 = 1024 elements = 8 per thread
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

        // Each thread processes its column for all BK k-values
        if (col < N) {
            for (uint32_t dk = 0; dk < BK && (bk + dk) < K; dk++) {
                uint32_t ki = bk + dk;
                uint32_t packed = qweight[ki * n_packed + pack_col];
                int q = unpack_int4_awq(packed, sub);

                uint32_t group = ki / group_size;
                float scale = scales[group * N + col];
                float zero = zeros[group * N + col];
                float w = ((float)q - zero) * scale;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    acc[i] += smem_input[ty * 8 + i][dk] * w;
                }
            }
        }
        __syncthreads();
    }

    // Write results
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
