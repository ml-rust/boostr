// MoE Grouped GEMM Kernel
// Variable-batch tiled matrix multiplication across experts.
//
// Each expert's token group is a separate matmul:
//   output[offset[e]..offset[e+1]] = tokens[offset[e]..offset[e+1]] @ weights[e]
//
// Uses shared memory tiling for efficient memory access.
// Grid: (ceil(out_dim/TILE_N), ceil(max_tokens/TILE_M), num_experts)

#include "../dtype_traits.cuh"

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

extern "C" __global__ void moe_grouped_gemm_f32(
    const float* __restrict__ tokens,         // [total_tokens, in_dim]
    const float* __restrict__ expert_weights,  // [num_experts, in_dim, out_dim]
    const long long* __restrict__ offsets,     // [num_experts + 1]
    float* __restrict__ output,               // [total_tokens, out_dim]
    int in_dim,
    int out_dim,
    int num_experts
) {
    int expert_idx = blockIdx.z;
    if (expert_idx >= num_experts) return;

    long long start = offsets[expert_idx];
    long long end = offsets[expert_idx + 1];
    int count = (int)(end - start);
    if (count <= 0) return;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int row = tile_row + threadIdx.y;
    int col = tile_col + threadIdx.x;

    if (row >= count || col >= out_dim) return;

    int global_row = (int)start + row;

    // Compute dot product: tokens[global_row, :] @ weights[expert_idx, :, col]
    const float* token_row = tokens + global_row * in_dim;
    const float* weight_col = expert_weights + expert_idx * in_dim * out_dim;

    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int k_start = 0; k_start < in_dim; k_start += TILE_K) {
        // Load tile from tokens
        int k_idx = k_start + threadIdx.x;
        if (row < count && k_idx < in_dim) {
            tile_a[threadIdx.y][threadIdx.x] = token_row[k_idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from weights (transposed access: weights[expert, k, col])
        int k_idx2 = k_start + threadIdx.y;
        if (k_idx2 < in_dim && col < out_dim) {
            tile_b[threadIdx.y][threadIdx.x] = weight_col[k_idx2 * out_dim + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate
        int k_end = min(TILE_K, in_dim - k_start);
        for (int kk = 0; kk < k_end; kk++) {
            acc += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < count && col < out_dim) {
        output[global_row * out_dim + col] = acc;
    }
}

// Fused grouped GEMM + SiLU activation
extern "C" __global__ void moe_grouped_gemm_silu_f32(
    const float* __restrict__ tokens,
    const float* __restrict__ expert_weights,
    const long long* __restrict__ offsets,
    float* __restrict__ output,
    int in_dim,
    int out_dim,
    int num_experts
) {
    int expert_idx = blockIdx.z;
    if (expert_idx >= num_experts) return;

    long long start = offsets[expert_idx];
    long long end = offsets[expert_idx + 1];
    int count = (int)(end - start);
    if (count <= 0) return;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int row = tile_row + threadIdx.y;
    int col = tile_col + threadIdx.x;

    if (row >= count || col >= out_dim) return;

    int global_row = (int)start + row;

    const float* token_row = tokens + global_row * in_dim;
    const float* weight_col = expert_weights + expert_idx * in_dim * out_dim;

    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int k_start = 0; k_start < in_dim; k_start += TILE_K) {
        int k_idx = k_start + threadIdx.x;
        if (row < count && k_idx < in_dim) {
            tile_a[threadIdx.y][threadIdx.x] = token_row[k_idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int k_idx2 = k_start + threadIdx.y;
        if (k_idx2 < in_dim && col < out_dim) {
            tile_b[threadIdx.y][threadIdx.x] = weight_col[k_idx2 * out_dim + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int k_end = min(TILE_K, in_dim - k_start);
        for (int kk = 0; kk < k_end; kk++) {
            acc += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < count && col < out_dim) {
        // SiLU: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-acc));
        output[global_row * out_dim + col] = acc * sigmoid;
    }
}

// Fused grouped GEMM + GeLU activation
extern "C" __global__ void moe_grouped_gemm_gelu_f32(
    const float* __restrict__ tokens,
    const float* __restrict__ expert_weights,
    const long long* __restrict__ offsets,
    float* __restrict__ output,
    int in_dim,
    int out_dim,
    int num_experts
) {
    int expert_idx = blockIdx.z;
    if (expert_idx >= num_experts) return;

    long long start = offsets[expert_idx];
    long long end = offsets[expert_idx + 1];
    int count = (int)(end - start);
    if (count <= 0) return;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int row = tile_row + threadIdx.y;
    int col = tile_col + threadIdx.x;

    if (row >= count || col >= out_dim) return;

    int global_row = (int)start + row;

    const float* token_row = tokens + global_row * in_dim;
    const float* weight_col = expert_weights + expert_idx * in_dim * out_dim;

    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int k_start = 0; k_start < in_dim; k_start += TILE_K) {
        int k_idx = k_start + threadIdx.x;
        if (row < count && k_idx < in_dim) {
            tile_a[threadIdx.y][threadIdx.x] = token_row[k_idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int k_idx2 = k_start + threadIdx.y;
        if (k_idx2 < in_dim && col < out_dim) {
            tile_b[threadIdx.y][threadIdx.x] = weight_col[k_idx2 * out_dim + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int k_end = min(TILE_K, in_dim - k_start);
        for (int kk = 0; kk < k_end; kk++) {
            acc += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < count && col < out_dim) {
        // GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = acc;
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi) ≈ 0.7978845608
        output[global_row * out_dim + col] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
