//! KV Cache operations traits
//!
//! Fused kernel operations for efficient KV cache management during inference.

use crate::error::Result;
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
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

    /// Fused KV cache update for every layer in one launch.
    ///
    /// Same write as [`KvCacheOps::kv_cache_update`] applied to each layer,
    /// but issued as a single kernel instead of one launch per layer.
    /// `position` is shared across all layers.
    ///
    /// # Layout contract
    ///
    /// - `k_caches`, `v_caches`, `new_ks`, `new_vs`: one entry per layer, all
    ///   four slices the same length, every tensor the same dtype
    /// - every `k_caches`/`v_caches` entry shares one `[B, num_kv_heads,
    ///   max_seq_len, head_dim]` shape; every `new_ks`/`new_vs` entry shares
    ///   one `[B, num_kv_heads, new_len, head_dim]` shape — layers differ
    ///   only in data, never in shape
    /// - `max_seq_len`: cache sequence-dimension size, shared across layers
    /// - `position`: starting write position, shared across layers
    #[allow(clippy::too_many_arguments)]
    fn kv_cache_update_batched(
        &self,
        k_caches: &[&Tensor<R>],
        v_caches: &[&Tensor<R>],
        new_ks: &[&Tensor<R>],
        new_vs: &[&Tensor<R>],
        max_seq_len: usize,
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

    /// Quantize one new token of K/V to INT4 and append into the cache in place.
    ///
    /// For autoregressive decoding: each generated token is quantized and
    /// written directly into its cache slot, without materializing a
    /// dequantized or full-tensor intermediate.
    ///
    /// # Layout contract
    ///
    /// - `k_cache`, `v_cache`: `[batch, num_heads, max_seq_len, head_dim/2]` u8,
    ///   packed INT4 (2 values per byte)
    /// - `k_scales`, `k_zeros`, `v_scales`, `v_zeros`:
    ///   `[batch, num_heads, max_seq_len * groups_per_token]` F16
    /// - `new_k`, `new_v`: `[batch, num_heads, head_dim]` — the token to append
    /// - `position`: slot in the cache sequence dimension to write
    /// - `group_size`: elements per quantization group
    ///
    /// `groups_per_token = ceil(head_dim / group_size)`. Writes only slot
    /// `position` in each cache; call once per generated token.
    #[allow(clippy::too_many_arguments)]
    fn append_kv_int4(
        &self,
        k_cache: &Tensor<R>,
        v_cache: &Tensor<R>,
        k_scales: &Tensor<R>,
        k_zeros: &Tensor<R>,
        v_scales: &Tensor<R>,
        v_zeros: &Tensor<R>,
        new_k: &Tensor<R>,
        new_v: &Tensor<R>,
        position: usize,
        group_size: Int4GroupSize,
    ) -> Result<()>;

    /// Duplicate physical blocks within the same K and V caches.
    ///
    /// Used for prefix caching / prefix-sharing: forking a sequence copies
    /// its shared prefix blocks into new physical blocks before the fork
    /// diverges, without touching the original blocks.
    ///
    /// # Layout contract
    ///
    /// - `key_cache`, `value_cache`: `[num_blocks, block_size, num_heads, head_dim]`
    /// - `block_mapping`: `[num_pairs * 2]` (I32) — `[i*2]` is the source
    ///   block index, `[i*2+1]` is the destination block index, for pair `i`
    ///
    /// Every pair copies its source block into its destination block inside
    /// both `key_cache` and `value_cache`. This does not perform host
    /// (CPU) offload — both caches stay on the same device.
    fn copy_blocks(
        &self,
        key_cache: &Tensor<R>,
        value_cache: &Tensor<R>,
        block_mapping: &Tensor<R>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()>;

    /// Move blocks between two distinct device-resident cache buffers.
    ///
    /// Unlike [`KvCacheOps::copy_blocks`], which duplicates blocks within one
    /// K/V cache pair, this moves blocks between two separate buffers
    /// (`src_cache` and `dst_cache`), one tensor at a time — call it once for
    /// K and once for V.
    ///
    /// # Layout contract
    ///
    /// - `src_cache`, `dst_cache`: `[num_blocks, block_size, num_heads, head_dim]`
    /// - `block_mapping`: `[num_pairs * 2]` (I32) — `[i*2]` is the source
    ///   block index, `[i*2+1]` is the destination block index, for pair `i`
    ///
    /// Both buffers must already be resident on the same device. This does
    /// NOT perform host (CPU) offload — a caller must never feed it host
    /// memory.
    fn swap_blocks(
        &self,
        src_cache: &Tensor<R>,
        dst_cache: &Tensor<R>,
        block_mapping: &Tensor<R>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()>;
}
