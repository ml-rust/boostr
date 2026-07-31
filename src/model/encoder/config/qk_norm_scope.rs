//! Which axis a QK-norm normalises over.
//!
//! Two architectures in this encoder normalise Q and K before attention, and
//! they do it over different axes with different norm types. Nothing about the
//! weight shape makes the difference safe to infer: for a model whose head
//! dimension happens to equal its hidden size the two are indistinguishable,
//! and picking wrong changes the numbers without changing any shape.

use serde::{Deserialize, Serialize};

/// The axis a QK-norm is applied over.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QkNormScope {
    /// Gemma / Qwen3: normalise each head independently, over `head_dim`,
    /// after Q and K have been reshaped to `[.., heads, head_dim]`.
    #[default]
    PerHead,
    /// jina-bert-v2: normalise over the whole `hidden_size` projection output,
    /// before the reshape into heads — so every head shares one mean and one
    /// variance. Mirrors llama.cpp's `llm_build_bert`, which reshapes Q to
    /// `[n_embd_head * n_head, n_tokens]` before calling `build_norm`.
    Hidden,
}
