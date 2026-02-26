// Speculative Decoding Kernels
//
// Element-wise acceptance probability and expected token count computation.
// Token verification (verify_speculative_tokens) is handled by impl_generic
// using numr's philox_uniform for reproducible, backend-consistent RNG.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../dtype_traits.cuh"

// -----------------------------------------------------------------
// Element-wise acceptance and residual probability computation
// -----------------------------------------------------------------
extern "C" __global__ void compute_acceptance_probs_kernel(
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    float* __restrict__ acceptance_probs,
    float* __restrict__ residual_probs,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float dp = draft_probs[idx];
    float tp = target_probs[idx];

    // Acceptance: min(1, target / draft)
    float accept = (dp > 1e-10f) ? fminf(1.0f, tp / dp) : 1.0f;
    acceptance_probs[idx] = accept;

    // Residual: max(0, target - draft)
    residual_probs[idx] = fmaxf(0.0f, tp - dp);
}

// -----------------------------------------------------------------
// Expected tokens computation (one thread per batch element)
// -----------------------------------------------------------------
extern "C" __global__ void compute_expected_tokens_kernel(
    const float* __restrict__ acceptance_rates,  // [batch, K]
    float* __restrict__ expected_tokens,         // [batch]
    int batch_size,
    int max_spec_tokens                          // K
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    float cumulative_prob = 1.0f;
    float expected = 0.0f;

    for (int i = 0; i < max_spec_tokens; i++) {
        cumulative_prob *= acceptance_rates[batch_idx * max_spec_tokens + i];
        expected += cumulative_prob;
    }

    // +1 for bonus token
    expected_tokens[batch_idx] = expected + 1.0f;
}
