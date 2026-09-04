//! ALiBi (Attention with Linear Biases) traits
//!
//! Adds position-dependent bias to attention scores before softmax.
//! Formula: `bias[i,j]` = -slope * |i - j|
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

    /// Backward pass for the MATERIALIZED biased-attention path.
    ///
    /// This is the backward of [`var_attention_with_bias`], which computes
    /// `softmax(Q @ K^T * scale + bias) @ V` and keeps the post-softmax
    /// probabilities in memory. Those probabilities come back in as `probs` —
    /// nothing here recomputes them.
    ///
    /// It is NOT the backward of a flash / online-softmax forward. That forward
    /// never materializes `probs`; it saves the log-sum-exp instead and its
    /// backward reconstructs each score tile. Use `flash_attention_bwd` there.
    ///
    /// # Bias
    /// An additive bias contributes NO gradient to Q, K or V:
    /// `d(scores + bias)/d(scores) = 1`, and the bias itself depends only on
    /// positions. So no slope, and no bias tensor, is an argument here — and
    /// these kernels serve ANY additive bias (ALiBi, a causal mask, a relative
    /// position table), not only ALiBi.
    ///
    /// # Math
    /// - `grad_probs  = grad_output @ V^T`
    /// - `grad_scores = probs * (grad_probs - rowsum(grad_probs * probs))`
    /// - `grad_q = (grad_scores @ K) * scale`
    /// - `grad_k = (grad_scores^T @ Q) * scale`
    /// - `grad_v = probs^T @ grad_output`
    ///
    /// # Layout
    /// - `grad_output`: `[batch_size, num_heads, seq_len_q, head_dim]`
    /// - `probs`: `[batch_size, num_heads, seq_len_q, seq_len_k]`
    /// - `q`: `[batch_size, num_heads, seq_len_q, head_dim]`
    /// - `k`, `v`: `[batch_size, num_heads, seq_len_k, head_dim]`
    ///
    /// `seq_len_q` and `seq_len_k` are read from the shapes. GQA/MQA heads must
    /// already be repeated to `num_heads`.
    ///
    /// # Returns
    /// `(grad_q, grad_k, grad_v)`, each in the input dtype and layout.
    ///
    /// [`var_attention_with_bias`]: crate::ops::var_attention_with_bias
    #[allow(clippy::too_many_arguments)]
    fn alibi_attention_bwd(
        &self,
        grad_output: &Tensor<R>,
        probs: &Tensor<R>,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
