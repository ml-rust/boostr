//! Residual/normalisation scheme.
//!
//! These three are genuinely distinct and none can emulate another:
//! `RmsNorm` with an all-ones weight still divides by the RMS, so a sandwich
//! layer with identity-valued post-norms is *not* a pre-norm layer.

use serde::{Deserialize, Serialize};

/// Where normalisation sits relative to the residual add.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormScheme {
    /// BERT / XLM-RoBERTa / NomicBert: `x = norm(x + sublayer(x))`.
    #[default]
    PostNorm,
    /// Qwen3: `x = x + sublayer(norm(x))`.
    PreNorm,
    /// Gemma: `x = x + post_norm(sublayer(pre_norm(x)))`.
    Sandwich,
}
