// Fused KV Cache Update Kernel
// Updates both K and V caches in a single kernel launch
// Reduces kernel launches from 2 to 1 per layer (32 -> 16 for 16-layer model)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused K+V cache update kernel
// Updates cache[:, :, position:position+new_len, :] = new_kv for both K and V
template<typename T>
__device__ __forceinline__ void kv_cache_update_impl(
    T* k_cache,                  // K cache [batch, heads, max_seq, head_dim]
    T* v_cache,                  // V cache [batch, heads, max_seq, head_dim]
    const T* new_k,              // New K values [batch, heads, new_len, head_dim]
    const T* new_v,              // New V values [batch, heads, new_len, head_dim]
    int outer_size,              // batch * num_kv_heads
    int max_seq_len,             // Maximum sequence length (cache dimension)
    int new_len,                 // Number of new tokens (typically 1 for decode)
    int head_dim,                // Head dimension
    int position,                // Current position in cache
    int total_elements           // Total elements per tensor (outer_size * new_len * head_dim)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose linear index to [outer, seq, head_dim]
    int outer_idx = idx / (new_len * head_dim);
    int remainder = idx % (new_len * head_dim);
    int seq_idx = remainder / head_dim;
    int dim_idx = remainder % head_dim;

    // Compute destination position in cache
    int cache_seq_idx = position + seq_idx;
    if (cache_seq_idx >= max_seq_len) return;

    // Compute linear indices
    int src_linear = idx;  // Source is contiguous [outer, new_len, head_dim]
    int dst_linear = outer_idx * (max_seq_len * head_dim)
                   + cache_seq_idx * head_dim
                   + dim_idx;

    // Update both K and V caches
    k_cache[dst_linear] = new_k[src_linear];
    v_cache[dst_linear] = new_v[src_linear];
}

extern "C" __global__ void kv_cache_update_f32(
    float* k_cache,
    float* v_cache,
    const float* new_k,
    const float* new_v,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements
) {
    kv_cache_update_impl<float>(k_cache, v_cache, new_k, new_v,
                                outer_size, max_seq_len, new_len, head_dim,
                                position, total_elements);
}

extern "C" __global__ void kv_cache_update_f16(
    __half* k_cache,
    __half* v_cache,
    const __half* new_k,
    const __half* new_v,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements
) {
    kv_cache_update_impl<__half>(k_cache, v_cache, new_k, new_v,
                                 outer_size, max_seq_len, new_len, head_dim,
                                 position, total_elements);
}

extern "C" __global__ void kv_cache_update_bf16(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements
) {
    kv_cache_update_impl<__nv_bfloat16>(k_cache, v_cache, new_k, new_v,
                                        outer_size, max_seq_len, new_len, head_dim,
                                        position, total_elements);
}

// ============================================================================
// Multi-layer batched KV cache update (further reduces launches)
// Updates ALL layers in a single kernel launch
// ============================================================================

// Multi-layer fused update - each thread handles one element across all layers
// Grid: total_elements_per_layer threads
// Each thread updates the same position in both K and V for one layer
template<typename T>
__device__ __forceinline__ void kv_cache_update_batched_impl(
    T** k_caches,                // Array of K cache pointers [num_layers]
    T** v_caches,                // Array of V cache pointers [num_layers]
    const T** new_ks,            // Array of new K pointers [num_layers]
    const T** new_vs,            // Array of new V pointers [num_layers]
    int num_layers,              // Number of layers
    int outer_size,              // batch * num_kv_heads
    int max_seq_len,             // Maximum sequence length
    int new_len,                 // Number of new tokens
    int head_dim,                // Head dimension
    int position,                // Current position in cache
    int total_elements_per_layer // Elements per layer
) {
    // 2D grid: x = element index, y = layer index
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer_idx = blockIdx.y;

    if (elem_idx >= total_elements_per_layer || layer_idx >= num_layers) return;

    // Decompose linear index to [outer, seq, head_dim]
    int outer_idx = elem_idx / (new_len * head_dim);
    int remainder = elem_idx % (new_len * head_dim);
    int seq_idx = remainder / head_dim;
    int dim_idx = remainder % head_dim;

    // Compute destination position in cache
    int cache_seq_idx = position + seq_idx;
    if (cache_seq_idx >= max_seq_len) return;

    // Compute linear indices
    int src_linear = elem_idx;
    int dst_linear = outer_idx * (max_seq_len * head_dim)
                   + cache_seq_idx * head_dim
                   + dim_idx;

    // Get pointers for this layer
    T* k_cache = k_caches[layer_idx];
    T* v_cache = v_caches[layer_idx];
    const T* new_k = new_ks[layer_idx];
    const T* new_v = new_vs[layer_idx];

    // Update both K and V caches for this layer
    k_cache[dst_linear] = new_k[src_linear];
    v_cache[dst_linear] = new_v[src_linear];
}

extern "C" __global__ void kv_cache_update_batched_f32(
    float** k_caches,
    float** v_caches,
    const float** new_ks,
    const float** new_vs,
    int num_layers,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements_per_layer
) {
    kv_cache_update_batched_impl<float>(k_caches, v_caches, new_ks, new_vs,
                                        num_layers, outer_size, max_seq_len,
                                        new_len, head_dim, position,
                                        total_elements_per_layer);
}

extern "C" __global__ void kv_cache_update_batched_f16(
    __half** k_caches,
    __half** v_caches,
    const __half** new_ks,
    const __half** new_vs,
    int num_layers,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements_per_layer
) {
    kv_cache_update_batched_impl<__half>(k_caches, v_caches, new_ks, new_vs,
                                         num_layers, outer_size, max_seq_len,
                                         new_len, head_dim, position,
                                         total_elements_per_layer);
}

extern "C" __global__ void kv_cache_update_batched_bf16(
    __nv_bfloat16** k_caches,
    __nv_bfloat16** v_caches,
    const __nv_bfloat16** new_ks,
    const __nv_bfloat16** new_vs,
    int num_layers,
    int outer_size,
    int max_seq_len,
    int new_len,
    int head_dim,
    int position,
    int total_elements_per_layer
) {
    kv_cache_update_batched_impl<__nv_bfloat16>(k_caches, v_caches, new_ks, new_vs,
                                                num_layers, outer_size, max_seq_len,
                                                new_len, head_dim, position,
                                                total_elements_per_layer);
}
