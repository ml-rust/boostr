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
}
