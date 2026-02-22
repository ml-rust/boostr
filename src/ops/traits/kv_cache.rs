//! KV Cache operations traits
//!
//! Fused kernel operations for efficient KV cache management during inference.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused KV cache update — writes new K and V tokens into caches in a single kernel.
///
/// Reduces kernel launches from 2 to 1 per layer.
///
/// # Layout contract
///
/// - `k_cache`, `v_cache`: `[B, num_kv_heads, max_seq_len, head_dim]` — preallocated cache
/// - `new_k`, `new_v`: `[B, num_kv_heads, new_len, head_dim]` — new tokens to insert
/// - `position`: starting write position in the sequence dimension
///
/// After this call, `cache[:, :, position:position+new_len, :] = new_kv`.
pub trait KvCacheOps<R: Runtime> {
    fn kv_cache_update(
        &self,
        k_cache: &Tensor<R>,
        v_cache: &Tensor<R>,
        new_k: &Tensor<R>,
        new_v: &Tensor<R>,
        position: usize,
    ) -> Result<()>;

    /// Reshape and cache — writes new K/V tokens into paged KV cache blocks.
    ///
    /// Used with PagedAttention for non-contiguous KV storage.
    ///
    /// # Layout contract
    ///
    /// - `key`, `value`: `[num_tokens, num_heads, head_dim]` — new tokens
    /// - `key_cache`, `value_cache`: `[num_blocks, block_size, num_heads, head_dim]`
    /// - `slot_mapping`: `[num_tokens]` (i64) — maps token index to slot in cache
    ///
    /// Slot `s` maps to block `s / block_size`, offset `s % block_size`.
    fn reshape_and_cache(
        &self,
        key: &Tensor<R>,
        value: &Tensor<R>,
        key_cache: &Tensor<R>,
        value_cache: &Tensor<R>,
        slot_mapping: &Tensor<R>,
        block_size: usize,
    ) -> Result<()>;
}
