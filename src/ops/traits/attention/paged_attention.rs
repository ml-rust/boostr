//! Paged Attention operations trait
//!
//! vLLM-style non-contiguous KV cache attention using block table indirection.
//! Based on "Efficient Memory Management for Large Language Model Serving with PagedAttention"
//! (Kwon et al., 2023).
//!
//! Key differences from FlashAttentionOps:
//! - K/V stored in non-contiguous blocks: `[num_blocks, block_size, head_dim]`
//! - Block table maps logical token positions to physical block addresses
//! - 2-3x memory efficiency vs contiguous KV cache
//! - Supports variable sequence lengths without padding

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Paged Flash Attention — fused attention with block-table KV cache indirection
///
/// # Layout contract
///
/// - `q`: `[B, num_heads, S_q, head_dim]` — queries (contiguous)
/// - `k_blocks`: `[num_blocks, block_size, num_kv_heads, head_dim]` — non-contiguous key blocks
/// - `v_blocks`: `[num_blocks, block_size, num_kv_heads, head_dim]` — non-contiguous value blocks
/// - `block_table`: `[B, max_num_blocks]` — i32, logical → physical block mapping
/// - Output: `[B, num_heads, S_q, head_dim]`
/// - Logsumexp: `[B, num_heads, S_q]` (F32, for backward pass)
///
/// # Block table
///
/// Each sequence has its own row in the block table. Entry `block_table[b][i]`
/// gives the physical block index for the i-th logical block of sequence `b`.
/// Token at logical position `t` is in block `t / block_size`, offset `t % block_size`.
#[allow(clippy::too_many_arguments)]
pub trait PagedAttentionOps<R: Runtime> {
    /// Paged attention forward pass (F32, F16, BF16)
    fn paged_attention_fwd(
        &self,
        q: &Tensor<R>,
        k_blocks: &Tensor<R>,
        v_blocks: &Tensor<R>,
        block_table: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Paged attention forward pass for FP8 tensors
    fn paged_attention_fwd_fp8(
        &self,
        q: &Tensor<R>,
        k_blocks: &Tensor<R>,
        v_blocks: &Tensor<R>,
        block_table: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Paged attention backward pass
    ///
    /// dK_blocks and dV_blocks use atomics internally (paged blocks may be shared),
    /// so they must be zero-initialized before calling.
    ///
    /// # Returns
    /// `(dq, dk_blocks, dv_blocks)` — gradients matching input shapes
    fn paged_attention_bwd(
        &self,
        dout: &Tensor<R>,
        q: &Tensor<R>,
        k_blocks: &Tensor<R>,
        v_blocks: &Tensor<R>,
        output: &Tensor<R>,
        lse: &Tensor<R>,
        block_table: &Tensor<R>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
