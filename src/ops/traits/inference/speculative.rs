//! Speculative decoding operations trait
//!
//! Token-level verification for speculative decoding (Leviathan et al., 2023).
//! Given draft model probabilities and target model probabilities, determines
//! which draft tokens to accept and samples a bonus token from the residual
//! distribution.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result of speculative token verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of tokens accepted (0..=K)
    pub num_accepted: usize,
    /// The accepted token IDs
    pub accepted_tokens: Vec<u32>,
    /// Bonus token from residual sampling (always present)
    pub bonus_token: u32,
    /// First rejection position (-1 if all accepted)
    pub first_rejection_pos: i32,
}

/// Speculative decoding token verification.
///
/// Compares draft and target model probability distributions to accept/reject
/// draft tokens, then samples a bonus token from the residual or next-position
/// distribution.
///
/// This is the core verification kernel — model orchestration (draft generation,
/// target scoring, adaptive depth) lives in `boostr::inference::speculative`.
pub trait SpeculativeOps<R: Runtime> {
    /// Verify draft tokens against target model probabilities.
    ///
    /// Implements the acceptance/rejection algorithm:
    /// - For each draft token t_i at position i:
    ///   - accept_prob = min(1, p_target(t_i) / p_draft(t_i))
    ///   - Accept if uniform() < accept_prob, else reject and stop
    /// - If rejected at position i: sample bonus from max(0, p_target - p_draft)
    /// - If all accepted: sample bonus from target's K+1 position
    ///
    /// # Layout
    ///
    /// - `draft_probs`: `[batch, K, vocab_size]` F32 — draft model probabilities
    /// - `target_probs`: `[batch, K+1, vocab_size]` F32 — target model probabilities
    /// - `draft_tokens`: `[batch, K]` I32 — draft token IDs
    /// - `seed`: RNG seed for reproducibility
    ///
    /// # Returns
    ///
    /// One `VerificationResult` per batch element.
    fn verify_speculative_tokens(
        &self,
        draft_probs: &Tensor<R>,
        target_probs: &Tensor<R>,
        draft_tokens: &Tensor<R>,
        seed: u64,
    ) -> Result<Vec<VerificationResult>>;

    /// Compute acceptance probabilities for analysis/diagnostics.
    ///
    /// For each (batch, position, vocab) entry, computes:
    /// - acceptance_prob = min(1, p_target / p_draft)
    /// - residual_prob = max(0, p_target - p_draft)
    ///
    /// # Layout
    ///
    /// - `draft_probs`: `[batch, K, vocab_size]` F32
    /// - `target_probs`: `[batch, K, vocab_size]` F32
    ///
    /// # Returns
    ///
    /// `(acceptance_probs, residual_probs)` both `[batch, K, vocab_size]` F32
    fn compute_acceptance_probs(
        &self,
        draft_probs: &Tensor<R>,
        target_probs: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Compute expected tokens per verification step for adaptive depth.
    ///
    /// Expected tokens = sum_{i=1}^{K} prod_{j=1}^{i} accept_rate_j + 1 (bonus)
    ///
    /// # Layout
    ///
    /// - `acceptance_rates`: `[batch, K]` F32 — per-position acceptance rates
    ///
    /// # Returns
    ///
    /// `[batch]` F32 — expected tokens per batch element
    fn compute_expected_tokens(&self, acceptance_rates: &Tensor<R>) -> Result<Tensor<R>>;
}
