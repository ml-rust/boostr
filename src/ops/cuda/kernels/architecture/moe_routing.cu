// MoE Top-K Routing Kernel
// Fused softmax + top-k selection + weight normalization
//
// Each block handles one token. Shared memory holds the softmax probabilities.
// After softmax, iteratively finds the top-k values.
//
// F16/BF16 variants: load in native dtype, compute in F32 (numerical stability),
// output weights always F32, indices always I32.

#include "../dtype_traits.cuh"

// ============================================================================
// F32 variant
// ============================================================================
extern "C" __global__ void moe_top_k_routing_f32(
    const float* __restrict__ logits,   // [num_tokens, num_experts]
    int* __restrict__ indices,          // [num_tokens, k]
    float* __restrict__ weights,        // [num_tokens, k]
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* probs = shared;  // [num_experts]

    // Step 1: Load logits and compute max for numerical stability
    float max_val = -1e30f;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = logits[token_idx * num_experts + e];
        probs[e] = val;
        if (val > max_val) max_val = val;
    }
    __syncthreads();

    // Block-wide max reduction
    __shared__ float block_max;
    if (tid == 0) {
        block_max = -1e30f;
        for (int e = 0; e < num_experts; e++) {
            if (probs[e] > block_max) block_max = probs[e];
        }
    }
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = expf(probs[e] - block_max);
        probs[e] = val;
    }
    __syncthreads();

    // Block-wide sum reduction
    __shared__ float block_sum;
    if (tid == 0) {
        block_sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            block_sum += probs[e];
        }
    }
    __syncthreads();

    // Step 3: Normalize to get softmax probabilities
    for (int e = tid; e < num_experts; e += blockDim.x) {
        probs[e] /= block_sum;
    }
    __syncthreads();

    // Step 4: Top-k selection (single thread — k is small, typically 1-8)
    if (tid == 0) {
        float top_sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int e = 0; e < num_experts; e++) {
                if (probs[e] > best_val) {
                    best_val = probs[e];
                    best_idx = e;
                }
            }
            indices[token_idx * k + ki] = best_idx;
            weights[token_idx * k + ki] = best_val;
            top_sum += best_val;
            probs[best_idx] = -1.0f;  // Mark as used
        }

        // Normalize top-k weights to sum to 1
        if (top_sum > 0.0f) {
            float inv_sum = 1.0f / top_sum;
            for (int ki = 0; ki < k; ki++) {
                weights[token_idx * k + ki] *= inv_sum;
            }
        }
    }
}

// ============================================================================
// F16 variant — load as __half, compute in F32
// ============================================================================
extern "C" __global__ void moe_top_k_routing_f16(
    const __half* __restrict__ logits,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* probs = shared;

    float max_val = -1e30f;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = __half2float(logits[token_idx * num_experts + e]);
        probs[e] = val;
        if (val > max_val) max_val = val;
    }
    __syncthreads();

    __shared__ float block_max;
    if (tid == 0) {
        block_max = -1e30f;
        for (int e = 0; e < num_experts; e++) {
            if (probs[e] > block_max) block_max = probs[e];
        }
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = expf(probs[e] - block_max);
        probs[e] = val;
    }
    __syncthreads();

    __shared__ float block_sum;
    if (tid == 0) {
        block_sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            block_sum += probs[e];
        }
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x) {
        probs[e] /= block_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float top_sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int e = 0; e < num_experts; e++) {
                if (probs[e] > best_val) {
                    best_val = probs[e];
                    best_idx = e;
                }
            }
            indices[token_idx * k + ki] = best_idx;
            weights[token_idx * k + ki] = best_val;
            top_sum += best_val;
            probs[best_idx] = -1.0f;
        }

        if (top_sum > 0.0f) {
            float inv_sum = 1.0f / top_sum;
            for (int ki = 0; ki < k; ki++) {
                weights[token_idx * k + ki] *= inv_sum;
            }
        }
    }
}

// ============================================================================
// BF16 variant — load as __nv_bfloat16, compute in F32
// ============================================================================
extern "C" __global__ void moe_top_k_routing_bf16(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ indices,
    float* __restrict__ weights,
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* probs = shared;

    float max_val = -1e30f;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = __bfloat162float(logits[token_idx * num_experts + e]);
        probs[e] = val;
        if (val > max_val) max_val = val;
    }
    __syncthreads();

    __shared__ float block_max;
    if (tid == 0) {
        block_max = -1e30f;
        for (int e = 0; e < num_experts; e++) {
            if (probs[e] > block_max) block_max = probs[e];
        }
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x) {
        float val = expf(probs[e] - block_max);
        probs[e] = val;
    }
    __syncthreads();

    __shared__ float block_sum;
    if (tid == 0) {
        block_sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            block_sum += probs[e];
        }
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x) {
        probs[e] /= block_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float top_sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int e = 0; e < num_experts; e++) {
                if (probs[e] > best_val) {
                    best_val = probs[e];
                    best_idx = e;
                }
            }
            indices[token_idx * k + ki] = best_idx;
            weights[token_idx * k + ki] = best_val;
            top_sum += best_val;
            probs[best_idx] = -1.0f;
        }

        if (top_sum > 0.0f) {
            float inv_sum = 1.0f / top_sum;
            for (int ki = 0; ki < k; ki++) {
                weights[token_idx * k + ki] *= inv_sum;
            }
        }
    }
}
