// MoE Token Permutation Kernel
// Parallel scatter of tokens into expert-grouped order.
//
// Given sorted indices mapping, copies token data to the permuted positions.
// One thread per element (token_idx * hidden_dim).

#include "../dtype_traits.cuh"

extern "C" __global__ void moe_permute_scatter_f32(
    const float* __restrict__ tokens,        // [num_tokens, hidden_dim]
    const long long* __restrict__ src_indices, // [total] — source token index for each output position
    float* __restrict__ permuted,            // [total, hidden_dim]
    int total,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total * hidden_dim;
    if (idx >= total_elements) return;

    int out_pos = idx / hidden_dim;
    int dim = idx % hidden_dim;

    long long src_token = src_indices[out_pos];
    permuted[out_pos * hidden_dim + dim] = tokens[src_token * hidden_dim + dim];
}

// Inverse permutation: gather from expert output back to original order
extern "C" __global__ void moe_unpermute_gather_f32(
    const float* __restrict__ expert_output,  // [total, hidden_dim]
    const long long* __restrict__ inv_perm,   // [total] — inverse permutation
    float* __restrict__ unsorted,             // [total, hidden_dim]
    int total,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total * hidden_dim;
    if (idx >= total_elements) return;

    int out_pos = idx / hidden_dim;
    int dim = idx % hidden_dim;

    long long src_pos = inv_perm[out_pos];
    unsorted[out_pos * hidden_dim + dim] = expert_output[src_pos * hidden_dim + dim];
}
