//! GGUF-embedded tokenizer (SentencePiece compatible)
//!
//! Extracts vocabulary from GGUF metadata and delegates tokenization
//! to splintr's `SentencePieceTokenizer`.

use crate::error::{Error, Result};
use crate::format::Gguf;
use crate::format::gguf::value::GgufValue;
use splintr::SentencePieceTokenizer;

/// Tokenizer that uses the vocabulary embedded in a GGUF file.
///
/// Extracts token strings, scores, and special token IDs from GGUF metadata,
/// then delegates all encoding/decoding to `splintr::SentencePieceTokenizer`.
pub struct GgufTokenizer {
    inner: SentencePieceTokenizer,
}

impl GgufTokenizer {
    /// Create a tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let metadata = gguf.metadata();

        // Extract token strings
        let tokens_array =
            metadata
                .get_array("tokenizer.ggml.tokens")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing tokenizer.ggml.tokens".into(),
                })?;

        let mut tokens = Vec::with_capacity(tokens_array.len());
        for (id, value) in tokens_array.iter().enumerate() {
            match value {
                GgufValue::String(s) => tokens.push(s.clone()),
                _ => {
                    return Err(Error::ModelError {
                        reason: format!("tokenizer.ggml.tokens[{}] is not a string", id),
                    });
                }
            }
        }

        // Extract scores (optional)
        let scores = if let Some(scores_array) = metadata.get_array("tokenizer.ggml.scores") {
            let mut out = Vec::with_capacity(scores_array.len());
            for (i, v) in scores_array.iter().enumerate() {
                let score = v.as_f32().ok_or_else(|| Error::ModelError {
                    reason: format!("tokenizer.ggml.scores[{i}] is not an f32"),
                })?;
                out.push(score);
            }
            out
        } else {
            vec![]
        };

        // Extract special token IDs
        let bos_token_id = metadata.get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        let inner = SentencePieceTokenizer::new(tokens, scores, bos_token_id, eos_token_id)
            .map_err(|e| Error::ModelError {
                reason: format!("Failed to create SentencePiece tokenizer: {}", e),
            })?;

        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text))
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids).map_err(|e| Error::ModelError {
            reason: format!("Decode error: {}", e),
        })
    }

    /// Check if a token is the EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.inner.is_eos(token_id)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id()
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }
}
