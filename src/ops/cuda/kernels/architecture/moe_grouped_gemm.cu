// MoE Grouped GEMM Kernel
// Variable-batch tiled matrix multiplication across experts.
//
// Each expert's token group is a separate matmul:
//   output[offset[e]..offset[e+1]] = tokens[offset[e]..offset[e+1]] @ weights[e]
//
// Uses shared memory tiling for efficient memory access.
// Grid: (ceil(out_dim/TILE_N), ceil(max_tokens/TILE_M), num_experts)
//
// F16/BF16 variants: load in native dtype, accumulate in F32, store in native dtype.
// Offsets are always int (I32).

#include "../dtype_traits.cuh"

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

// ============================================================================
// Helper macros for generating all dtype x activation variants
// ============================================================================

#define DEFINE_MOE_GROUPED_GEMM(SUFFIX, DTYPE, LOAD, STORE, EPILOGUE) \
extern "C" __global__ void moe_grouped_gemm##SUFFIX( \
    const DTYPE* __restrict__ tokens, \
    const DTYPE* __restrict__ expert_weights, \
    const int* __restrict__ offsets, \
    DTYPE* __restrict__ output, \
    int in_dim, \
    int out_dim, \
    int num_experts \
) { \
    int expert_idx = blockIdx.z; \
    if (expert_idx >= num_experts) return; \
    \
    int start = offsets[expert_idx]; \
    int end = offsets[expert_idx + 1]; \
    int count = end - start; \
    if (count <= 0) return; \
    \
    int tile_row = blockIdx.y * TILE_M; \
    int tile_col = blockIdx.x * TILE_N; \
    \
    int row = tile_row + threadIdx.y; \
    int col = tile_col + threadIdx.x; \
    \
    /* Do NOT early-return here — all threads must reach __syncthreads(). */ \
    int valid = (row < count && col < out_dim); \
    \
    int global_row = start + (row < count ? row : 0); \
    \
    const DTYPE* token_row = tokens + global_row * in_dim; \
    const DTYPE* weight_col = expert_weights + expert_idx * in_dim * out_dim; \
    \
    __shared__ float tile_a[TILE_M][TILE_K]; \
    __shared__ float tile_b[TILE_K][TILE_N]; \
    \
    float acc = 0.0f; \
    \
    for (int k_start = 0; k_start < in_dim; k_start += TILE_K) { \
        int k_idx = k_start + threadIdx.x; \
        if (row < count && k_idx < in_dim) { \
            tile_a[threadIdx.y][threadIdx.x] = LOAD(token_row[k_idx]); \
        } else { \
            tile_a[threadIdx.y][threadIdx.x] = 0.0f; \
        } \
        \
        int k_idx2 = k_start + threadIdx.y; \
        if (k_idx2 < in_dim && col < out_dim) { \
            tile_b[threadIdx.y][threadIdx.x] = LOAD(weight_col[k_idx2 * out_dim + col]); \
        } else { \
            tile_b[threadIdx.y][threadIdx.x] = 0.0f; \
        } \
        \
        __syncthreads(); \
        \
        int k_end = min(TILE_K, in_dim - k_start); \
        for (int kk = 0; kk < k_end; kk++) { \
            acc += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x]; \
        } \
        \
        __syncthreads(); \
    } \
    \
    if (valid) { \
        float result = EPILOGUE(acc); \
        STORE(output[global_row * out_dim + col], result); \
    } \
}

// Load/store helpers per dtype
#define LOAD_F32(x) (x)
#define STORE_F32(dst, val) (dst) = (val)

#define LOAD_F16(x) __half2float(x)
#define STORE_F16(dst, val) (dst) = __float2half(val)

#define LOAD_BF16(x) __bfloat162float(x)
#define STORE_BF16(dst, val) (dst) = __float2bfloat16(val)

// Epilogue helpers
#define EPILOGUE_NONE(x) (x)

static __device__ __forceinline__ float epilogue_silu(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return x * sigmoid;
}
#define EPILOGUE_SILU(x) epilogue_silu(x)

static __device__ __forceinline__ float epilogue_gelu(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}
#define EPILOGUE_GELU(x) epilogue_gelu(x)

// ============================================================================
// F32 variants (no conversion)
// ============================================================================
DEFINE_MOE_GROUPED_GEMM(_f32, float, LOAD_F32, STORE_F32, EPILOGUE_NONE)
DEFINE_MOE_GROUPED_GEMM(_silu_f32, float, LOAD_F32, STORE_F32, EPILOGUE_SILU)
DEFINE_MOE_GROUPED_GEMM(_gelu_f32, float, LOAD_F32, STORE_F32, EPILOGUE_GELU)

// ============================================================================
// F16 variants (load/store as __half, accumulate in F32)
// ============================================================================
DEFINE_MOE_GROUPED_GEMM(_f16, __half, LOAD_F16, STORE_F16, EPILOGUE_NONE)
DEFINE_MOE_GROUPED_GEMM(_silu_f16, __half, LOAD_F16, STORE_F16, EPILOGUE_SILU)
DEFINE_MOE_GROUPED_GEMM(_gelu_f16, __half, LOAD_F16, STORE_F16, EPILOGUE_GELU)

// ============================================================================
// BF16 variants (load/store as __nv_bfloat16, accumulate in F32)
// ============================================================================
DEFINE_MOE_GROUPED_GEMM(_bf16, __nv_bfloat16, LOAD_BF16, STORE_BF16, EPILOGUE_NONE)
DEFINE_MOE_GROUPED_GEMM(_silu_bf16, __nv_bfloat16, LOAD_BF16, STORE_BF16, EPILOGUE_SILU)
DEFINE_MOE_GROUPED_GEMM(_gelu_bf16, __nv_bfloat16, LOAD_BF16, STORE_BF16, EPILOGUE_GELU)
