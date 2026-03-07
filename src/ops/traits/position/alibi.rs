//! ALiBi (Attention with Linear Biases) traits
//!
//! Adds position-dependent bias to attention scores before softmax.
//! Formula: bias[i,j] = -slope * |i - j|
//! Slope per head: slope_h = 2^(-8h/H)
//!
//! Used in BLOOM, MPT, Falcon for length extrapolation.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// ALiBi attention operations
///
/// Add ALiBi bias to attention scores in-place. Called AFTER Q@K^T
/// but BEFORE softmax.
///
/// # Layout
/// - `scores`: `[batch, num_heads, seq_len_q, seq_len_k]` — modified in-place
pub trait AlibiOps<R: Runtime> {
    /// Add ALiBi bias to attention scores in-place
    fn alibi_add_bias(
        &self,
        scores: &Tensor<R>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Result<()>;

    /// Add ALiBi bias + causal mask to attention scores in-place.
    ///
    /// Combines ALiBi bias with causal masking in a single pass:
    /// - For positions where `ki > qi + position`: sets score to `-inf`
    /// - Otherwise: adds ALiBi bias `-slope * |qi + position - ki|`
    ///
    /// `position` is the absolute position of the first query token
    /// (e.g., during decode with KV cache, `position` = number of prior tokens).
    fn alibi_add_bias_causal(
        &self,
        scores: &Tensor<R>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        position: usize,
    ) -> Result<()>;
}
