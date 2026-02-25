//! Mixture of Experts (MoE) operations trait
//!
//! Inference-path MoE operations: routing, token permutation, and grouped GEMM.
//! These compose numr primitives (softmax, topk, scatter, matmul) into fused
//! operations that backends can optimize with dedicated kernels.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Activation function for fused grouped GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEActivation {
    /// SiLU (Swish): x * sigmoid(x)
    SiLU,
    /// GELU: x * Φ(x)
    GeLU,
    /// No activation (identity)
    None,
}

/// Mixture of Experts operations.
///
/// Composite ops for MoE inference: top-k routing, token permutation
/// (scatter/gather), and variable-batch grouped GEMM across experts.
///
/// All operations work on `Tensor<R>` (no autograd) — these are inference-path
/// optimizations. The existing `nn/moe/` modules use autograd-composed ops
/// which handle gradients for training.
pub trait MoEOps<R: Runtime> {
    /// Top-k expert routing with softmax normalization.
    ///
    /// Applies softmax to logits, selects the top-k experts per token,
    /// and normalizes the selected weights to sum to 1.
    ///
    /// # Layout
    ///
    /// - `logits`: `[num_tokens, num_experts]` — raw gate logits
    /// - `k`: number of experts per token
    /// - Returns `(indices, weights)`:
    ///   - `indices`: `[num_tokens, k]` I32 — selected expert indices
    ///   - `weights`: `[num_tokens, k]` F32 — normalized routing weights
    fn moe_top_k_routing(&self, logits: &Tensor<R>, k: usize) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Permute tokens into expert-grouped order.
    ///
    /// Scatters tokens so that all tokens assigned to the same expert are
    /// contiguous, enabling efficient grouped GEMM.
    ///
    /// # Layout
    ///
    /// - `tokens`: `[num_tokens, hidden_dim]` — input token embeddings
    /// - `indices`: `[num_tokens, k]` I32 — expert assignments from routing
    /// - Returns `(permuted, expert_offsets, sort_indices)`:
    ///   - `permuted`: `[num_tokens * k, hidden_dim]` — reordered tokens
    ///   - `expert_offsets`: `[num_experts + 1]` I32 — CSR-style offset array
    ///   - `sort_indices`: `[num_tokens * k]` I32 — permutation for unpermute
    fn moe_permute_tokens(
        &self,
        tokens: &Tensor<R>,
        indices: &Tensor<R>,
        num_experts: usize,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Unpermute expert outputs back to original token order.
    ///
    /// Gathers expert outputs using the sort indices from `moe_permute_tokens`
    /// and applies weighted combination for tokens routed to multiple experts.
    ///
    /// # Layout
    ///
    /// - `expert_output`: `[num_tokens * k, hidden_dim]` — expert computation results
    /// - `sort_indices`: `[num_tokens * k]` I32 — from `moe_permute_tokens`
    /// - `weights`: `[num_tokens, k]` — routing weights
    /// - `num_tokens`: original number of tokens
    /// - Returns: `[num_tokens, hidden_dim]` — weighted-combined output
    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<R>,
        sort_indices: &Tensor<R>,
        weights: &Tensor<R>,
        num_tokens: usize,
    ) -> Result<Tensor<R>>;

    /// Grouped GEMM across experts.
    ///
    /// Performs per-expert matrix multiplication on contiguous token groups.
    /// Each expert's tokens (defined by offset ranges) are multiplied by
    /// that expert's weight matrix.
    ///
    /// # Layout
    ///
    /// - `permuted_tokens`: `[total_tokens, in_dim]` — expert-grouped tokens
    /// - `expert_weights`: `[num_experts, in_dim, out_dim]` — per-expert weight matrices
    /// - `expert_offsets`: `[num_experts + 1]` I32 — CSR-style offset array
    /// - Returns: `[total_tokens, out_dim]` — concatenated expert outputs
    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<R>,
        expert_weights: &Tensor<R>,
        expert_offsets: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Fused grouped GEMM with activation.
    ///
    /// Same as `moe_grouped_gemm` but applies an activation function
    /// (SiLU or GeLU) to each expert's output before concatenation.
    ///
    /// # Layout
    ///
    /// Same as `moe_grouped_gemm`, plus:
    /// - `activation`: which activation to apply after each expert's matmul
    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<R>,
        expert_weights: &Tensor<R>,
        expert_offsets: &Tensor<R>,
        activation: MoEActivation,
    ) -> Result<Tensor<R>>;
}
