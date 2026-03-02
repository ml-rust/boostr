//! Sampling penalty operations trait
//!
//! Applies repetition, frequency, and presence penalties to logits in-place
//! during autoregressive inference. Prevents degenerate repetition by penalizing
//! tokens that have recently appeared.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sampling penalty operations for inference-time logit modification.
///
/// All backends must produce identical results — the algorithm is deterministic
/// and order-independent (each unique token is penalized independently).
pub trait SamplingOps<R: Runtime> {
    /// Apply repetition, frequency, and presence penalties to logits in-place.
    ///
    /// For each unique token in `token_ids` with corresponding `token_counts`:
    /// - **Repetition penalty**: if logit > 0, divide by `repeat_penalty`;
    ///   if logit < 0, multiply by `repeat_penalty`
    /// - **Frequency penalty**: subtract `frequency_penalty * count`
    /// - **Presence penalty**: subtract `presence_penalty`
    ///
    /// # Layout
    ///
    /// - `logits`: `[vocab_size]` F32 — modified in-place
    /// - `token_ids`: `[num_unique]` I64 — unique token vocabulary indices
    /// - `token_counts`: `[num_unique]` I32 — occurrence count per token
    ///
    /// Callers are responsible for:
    /// - Narrowing logits to the last sequence position before calling
    /// - Computing unique token IDs and counts from the penalty window
    fn apply_sampling_penalties(
        &self,
        logits: &Tensor<R>,
        token_ids: &Tensor<R>,
        token_counts: &Tensor<R>,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Result<()>;

    /// Sample a single token from logits using the full stochastic pipeline.
    ///
    /// Applies temperature scaling, softmax, top-k, top-p, min-p filtering,
    /// then multinomial sampling — all on-device. Randomness is generated
    /// on-device via `RandomOps::rand`.
    ///
    /// # Arguments
    ///
    /// - `logits`: `[vocab_size]` F32 — already penalized
    /// - `temperature`: temperature scaling (1.0 = no scaling)
    /// - `top_k`: keep only top-k tokens (0 = disabled)
    /// - `top_p`: nucleus sampling threshold (1.0 = disabled)
    /// - `min_p`: minimum probability relative to max (0.0 = disabled)
    ///
    /// Returns the sampled token ID.
    fn sample_token(
        &self,
        logits: &Tensor<R>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32>;
}
