//! GGUF-embedded tokenizer with auto-detection.
//!
//! Reads `tokenizer.ggml.model` from GGUF metadata and constructs the
//! appropriate splintr backend: WordPiece for `"bert"`, SentencePiece for
//! `"llama"`/`"gpt2"`.

use crate::error::{Error, Result};
use crate::format::Gguf;
use crate::format::gguf::value::GgufValue;
use splintr::{SentencePieceTokenizer, Tokenize, TokenizeError, WordPieceTokenizer};

/// Tokenizer that uses the vocabulary embedded in a GGUF file.
///
/// Auto-detects the tokenizer type from `tokenizer.ggml.model` metadata:
/// - `"bert"` → [`WordPieceTokenizer`]
/// - `"llama"` / `"gpt2"` → [`SentencePieceTokenizer`]
pub struct GgufTokenizer {
    inner: Box<dyn Tokenize>,
}

impl GgufTokenizer {
    /// Create a tokenizer from GGUF metadata, auto-detecting the type.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let metadata = gguf.metadata();

        let model_type = metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("llama");

        // Extract token strings (required for all types)
        let tokens = extract_tokens(metadata)?;

        match model_type {
            "bert" => Self::build_wordpiece(metadata, tokens),
            _ => Self::build_sentencepiece(metadata, tokens),
        }
    }

    fn build_wordpiece(
        metadata: &crate::format::GgufMetadata,
        tokens: Vec<String>,
    ) -> Result<Self> {
        // Find [UNK] token ID from vocab or token_type array
        let unk_token_id = find_special_token_id(&tokens, metadata, "[UNK]", 0);

        // Check if this is an uncased model (default true for BERT)
        // GGUF doesn't have a standard key for this, so we heuristic:
        // if vocab contains lowercase "the" but not "The", it's uncased
        let do_lower_case = tokens.iter().any(|t| t == "the") && !tokens.iter().any(|t| t == "The");

        let inner = WordPieceTokenizer::new(tokens, unk_token_id, 200, do_lower_case);
        Ok(Self {
            inner: Box::new(inner),
        })
    }

    fn build_sentencepiece(
        metadata: &crate::format::GgufMetadata,
        tokens: Vec<String>,
    ) -> Result<Self> {
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

        let bos_token_id = metadata.get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        let inner = SentencePieceTokenizer::new(tokens, scores, bos_token_id, eos_token_id)
            .map_err(|e| Error::ModelError {
                reason: format!("Failed to create SentencePiece tokenizer: {}", e),
            })?;

        Ok(Self {
            inner: Box::new(inner),
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids).map_err(|e| Error::ModelError {
            reason: format!("Decode error: {}", e),
        })
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

impl Tokenize for GgufTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> std::result::Result<String, TokenizeError> {
        self.inner.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Extract token strings from GGUF metadata.
fn extract_tokens(metadata: &crate::format::GgufMetadata) -> Result<Vec<String>> {
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
    Ok(tokens)
}

/// Find a special token ID, checking the token_type array first, then falling back
/// to searching the vocab for the token string.
fn find_special_token_id(
    tokens: &[String],
    metadata: &crate::format::GgufMetadata,
    token_str: &str,
    default: u32,
) -> u32 {
    // First try to find by matching the token string in the vocab
    for (id, t) in tokens.iter().enumerate() {
        if t == token_str {
            return id as u32;
        }
    }

    // Check metadata for explicit ID
    let key = match token_str {
        "[UNK]" => "tokenizer.ggml.unknown_token_id",
        "[PAD]" => "tokenizer.ggml.padding_token_id",
        "[CLS]" => "tokenizer.ggml.cls_token_id",
        "[SEP]" => "tokenizer.ggml.sep_token_id",
        _ => return default,
    };

    metadata.get_u32(key).unwrap_or(default)
}
