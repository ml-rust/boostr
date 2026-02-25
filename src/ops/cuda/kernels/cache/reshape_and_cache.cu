// Reshape and Cache Kernel - Optimized KV cache writes for PagedAttention
//
// Writes new K/V tokens into the paged KV cache with:
// 1. Vectorized memory operations (128-bit coalesced access)
// 2. Slot mapping indirection (logical → physical blocks)
// 3. Support for non-contiguous block placement
//
// Based on vLLM's cache_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

struct Float4 { float x, y, z, w; };

extern "C" {

// ============================================================================
// Reshape and Cache — write new tokens into paged KV cache
// Input:  [num_tokens, num_heads, head_dim]
// Cache:  [num_blocks, block_size, num_heads, head_dim]
// Slot:   [num_tokens] → slot index
// ============================================================================

__global__ void reshape_and_cache_f32(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;
    const int input_offset = (token_idx * num_heads + head_idx) * head_dim;
    const int cache_offset = ((block_idx * block_size + block_offset) * num_heads + head_idx) * head_dim;

    const int vec_size = 4;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        Float4 k_vec = *reinterpret_cast<const Float4*>(&key[input_offset + elem_offset]);
        Float4 v_vec = *reinterpret_cast<const Float4*>(&value[input_offset + elem_offset]);
        *reinterpret_cast<Float4*>(&key_cache[cache_offset + elem_offset]) = k_vec;
        *reinterpret_cast<Float4*>(&value_cache[cache_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[cache_offset + i] = key[input_offset + i];
            value_cache[cache_offset + i] = value[input_offset + i];
        }
    }
}

__global__ void reshape_and_cache_f16(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;
    const int input_offset = (token_idx * num_heads + head_idx) * head_dim;
    const int cache_offset = ((block_idx * block_size + block_offset) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 k_vec = *reinterpret_cast<const float4*>(&key[input_offset + elem_offset]);
        float4 v_vec = *reinterpret_cast<const float4*>(&value[input_offset + elem_offset]);
        *reinterpret_cast<float4*>(&key_cache[cache_offset + elem_offset]) = k_vec;
        *reinterpret_cast<float4*>(&value_cache[cache_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[cache_offset + i] = key[input_offset + i];
            value_cache[cache_offset + i] = value[input_offset + i];
        }
    }
}

__global__ void reshape_and_cache_bf16(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;
    const int input_offset = (token_idx * num_heads + head_idx) * head_dim;
    const int cache_offset = ((block_idx * block_size + block_offset) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 k_vec = *reinterpret_cast<const float4*>(&key[input_offset + elem_offset]);
        float4 v_vec = *reinterpret_cast<const float4*>(&value[input_offset + elem_offset]);
        *reinterpret_cast<float4*>(&key_cache[cache_offset + elem_offset]) = k_vec;
        *reinterpret_cast<float4*>(&value_cache[cache_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[cache_offset + i] = key[input_offset + i];
            value_cache[cache_offset + i] = value[input_offset + i];
        }
    }
}

// ============================================================================
// Copy Blocks — for prefix caching / block copying
// ============================================================================

__global__ void copy_blocks_f32(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 4;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        Float4 k_vec = *reinterpret_cast<Float4*>(&key_cache[src_offset + elem_offset]);
        Float4 v_vec = *reinterpret_cast<Float4*>(&value_cache[src_offset + elem_offset]);
        *reinterpret_cast<Float4*>(&key_cache[dst_offset + elem_offset]) = k_vec;
        *reinterpret_cast<Float4*>(&value_cache[dst_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[dst_offset + i] = key_cache[src_offset + i];
            value_cache[dst_offset + i] = value_cache[src_offset + i];
        }
    }
}

__global__ void copy_blocks_f16(
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 k_vec = *reinterpret_cast<float4*>(&key_cache[src_offset + elem_offset]);
        float4 v_vec = *reinterpret_cast<float4*>(&value_cache[src_offset + elem_offset]);
        *reinterpret_cast<float4*>(&key_cache[dst_offset + elem_offset]) = k_vec;
        *reinterpret_cast<float4*>(&value_cache[dst_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[dst_offset + i] = key_cache[src_offset + i];
            value_cache[dst_offset + i] = value_cache[src_offset + i];
        }
    }
}

__global__ void copy_blocks_bf16(
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 k_vec = *reinterpret_cast<float4*>(&key_cache[src_offset + elem_offset]);
        float4 v_vec = *reinterpret_cast<float4*>(&value_cache[src_offset + elem_offset]);
        *reinterpret_cast<float4*>(&key_cache[dst_offset + elem_offset]) = k_vec;
        *reinterpret_cast<float4*>(&value_cache[dst_offset + elem_offset]) = v_vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            key_cache[dst_offset + i] = key_cache[src_offset + i];
            value_cache[dst_offset + i] = value_cache[src_offset + i];
        }
    }
}

// ============================================================================
// Swap Blocks — for CPU offloading
// ============================================================================

__global__ void swap_blocks_f32(
    float* __restrict__ src_cache,
    float* __restrict__ dst_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 4;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        Float4 vec = *reinterpret_cast<Float4*>(&src_cache[src_offset + elem_offset]);
        *reinterpret_cast<Float4*>(&dst_cache[dst_offset + elem_offset]) = vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            dst_cache[dst_offset + i] = src_cache[src_offset + i];
        }
    }
}

__global__ void swap_blocks_f16(
    __half* __restrict__ src_cache,
    __half* __restrict__ dst_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 vec = *reinterpret_cast<float4*>(&src_cache[src_offset + elem_offset]);
        *reinterpret_cast<float4*>(&dst_cache[dst_offset + elem_offset]) = vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            dst_cache[dst_offset + i] = src_cache[src_offset + i];
        }
    }
}

__global__ void swap_blocks_bf16(
    __nv_bfloat16* __restrict__ src_cache,
    __nv_bfloat16* __restrict__ dst_cache,
    const int32_t* __restrict__ block_mapping,
    const int num_pairs,
    const int num_heads,
    const int head_dim,
    const int block_size
) {
    const int pair_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot_in_block = blockIdx.z;
    const int tid = threadIdx.x;

    if (pair_idx >= num_pairs || head_idx >= num_heads || slot_in_block >= block_size) return;

    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];
    const int src_offset = ((src_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;
    const int dst_offset = ((dst_block * block_size + slot_in_block) * num_heads + head_idx) * head_dim;

    const int vec_size = 8;
    const int num_vecs = head_dim / vec_size;

    if (tid < num_vecs) {
        const int elem_offset = tid * vec_size;
        float4 vec = *reinterpret_cast<float4*>(&src_cache[src_offset + elem_offset]);
        *reinterpret_cast<float4*>(&dst_cache[dst_offset + elem_offset]) = vec;
    }

    const int remainder_start = num_vecs * vec_size;
    if (tid == 0) {
        for (int i = remainder_start; i < head_dim; i++) {
            dst_cache[dst_offset + i] = src_cache[src_offset + i];
        }
    }
}

} // extern "C"
