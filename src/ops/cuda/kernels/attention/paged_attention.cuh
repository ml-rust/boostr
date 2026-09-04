// Shared device helper for paged_attention.cu and paged_attention_fp8.cu.
#pragma once

// Compute physical address for a KV token using block table
// block_table: [batch_size, max_num_blocks] - logical to physical block mapping
// token_idx: Logical token index within sequence
// block_size: Number of tokens per block (typically 16)
// num_kv_heads: Number of KV heads (for multi-head interleaved layout)
// kv_head_idx: Which KV head to access
// head_dim: Dimension of each head
// Returns: Physical offset into K_blocks or V_blocks array
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
__device__ __forceinline__ int get_paged_kv_offset(
    const int* __restrict__ block_table,
    int batch_idx,
    int max_num_blocks,
    int token_idx,
    int block_size,
    int num_kv_heads,
    int kv_head_idx,
    int head_dim
) {
    int logical_block = token_idx / block_size;
    int block_offset = token_idx % block_size;
    int physical_block = block_table[batch_idx * max_num_blocks + logical_block];
    return physical_block * block_size * num_kv_heads * head_dim
         + block_offset * num_kv_heads * head_dim
         + kv_head_idx * head_dim;
}
