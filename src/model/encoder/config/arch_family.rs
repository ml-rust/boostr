//! Architecture family: selects position-id generation and embedding behaviour.

use serde::{Deserialize, Serialize};

/// Architecture family for position-id generation and embedding behaviour.
///
/// BERT uses simple 0-based position ids.  XLM-RoBERTa (used by e.g.
/// bge-reranker-v2-m3) reserves position `pad_token_id` for padding and
/// numbers real tokens starting from `pad_token_id + 1`.
/// NomicBert replaces learned position embeddings with RoPE and uses a
/// SwiGLU FFN.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArchFamily {
    /// Standard BERT: position_ids = [0, 1, ..., S-1].
    #[default]
    Bert,
    /// XLM-RoBERTa: position_ids computed as cumsum(input_ids != pad_id) + pad_id,
    /// with padding positions assigned position_id = pad_id.
    XlmRoberta,
    /// NomicBert: RoPE positions (no learned position embedding), SwiGLU FFN,
    /// fused QKV projection, token-type embedding row 0, mean pooling.
    NomicBert,
    /// Gemma3-embedding: RoPE positions, sandwich RMSNorm, GQA, QK-norm,
    /// GeGLU FFN, token-embedding scale sqrt(hidden_size), mean pooling.
    /// Interleaves local (windowed, low RoPE base) and global blocks.
    /// No learned position embedding. No biases anywhere.
    GemmaEmbedding,
    /// Qwen3-embedding: RoPE positions, pre-norm RMSNorm with plain residuals,
    /// GQA, QK-norm, SwiGLU FFN, causal attention, last-token pooling.
    /// Head dim is given explicitly and is NOT `hidden_size / head_count`.
    Qwen3,
}

impl ArchFamily {
    /// Whether this family derives positions from RoPE rather than a learned
    /// absolute position embedding table.
    pub fn uses_rope(self) -> bool {
        matches!(self, Self::NomicBert | Self::GemmaEmbedding | Self::Qwen3)
    }
}
