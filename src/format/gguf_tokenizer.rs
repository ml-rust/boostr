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
    eos_token_id: u32,
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
        // Convert a SentencePiece-marked vocab to WordPiece convention first —
        // see `normalize_wordpiece_vocab`. Everything below (the [UNK] lookup
        // and the uncased heuristic) matches against plain strings like "the",
        // so it MUST run on the normalized vocab or it silently misfires.
        let tokens = normalize_wordpiece_vocab(tokens);

        // Find [UNK] token ID from vocab or token_type array
        let unk_token_id = find_special_token_id(&tokens, metadata, "[UNK]", 0);

        // Check if this is an uncased model (default true for BERT)
        // GGUF doesn't have a standard key for this, so we heuristic:
        // if vocab contains lowercase "the" but not "The", it's uncased
        let do_lower_case = tokens.iter().any(|t| t == "the") && !tokens.iter().any(|t| t == "The");

        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);
        let inner = WordPieceTokenizer::new(tokens, unk_token_id, 200, do_lower_case);
        Ok(Self {
            inner: Box::new(inner),
            eos_token_id,
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
            eos_token_id,
        })
    }

    /// EOS token ID from GGUF metadata.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Whether a token ID is the EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
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

    fn cls_sep_ids(&self) -> Option<(u32, u32)> {
        self.inner.cls_sep_ids()
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

/// The SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
const SP_WORD_BOUNDARY: char = '\u{2581}';

/// Rewrite a GGUF BERT vocab that uses SentencePiece word-boundary markers into
/// the WordPiece convention [`WordPieceTokenizer`] expects.
///
/// Some GGUF converters store a WordPiece vocab with SentencePiece marking:
/// word-INITIAL pieces get a leading `▁` and continuation pieces are bare,
/// instead of BERT's bare-initial / `##`-continuation. `nomic-embed-text-v1.5`
/// is one such file — 23695 of its 30522 tokens carry `▁` and **zero** carry
/// `##`, so `vocab[1996]` is `"▁the"` where bert-base-uncased has `"the"`, and
/// `vocab[2015]` is `"s"` where bert-base-uncased has `"##s"`.
///
/// Handing those strings to `WordPieceTokenizer` unchanged means no word ever
/// matches its own vocab entry, so greedy longest-match shatters every word
/// into stray fragments: `"hello the quick brown fox"` round-tripped as
/// `"hell o the qui ck bro wn fo x"`. The resulting ids are near-random, and
/// mean-pooled embeddings of ANY two texts collapse onto the corpus average —
/// measured cosine distance between unrelated sentences was ~0.0005, which
/// makes dense retrieval pure noise while still looking healthy end to end.
///
/// The mapping is total and lossless for this convention:
/// - `▁X` → `X`      (word-initial)
/// - `[SPECIAL]` → unchanged
/// - `X` → `##X`     (continuation)
///
/// Punctuation and digits carry `▁` too (`"▁!"`, `"▁1"`), so they land on the
/// word-initial branch exactly as BERT expects.
///
/// A vocab that already uses `##`, or that has no `▁` at all, is returned
/// untouched — detection is on the vocab's own contents, never on the model
/// name, so a correctly-marked file is never rewritten.
fn normalize_wordpiece_vocab(tokens: Vec<String>) -> Vec<String> {
    let has_sp_marker = tokens.iter().any(|t| t.starts_with(SP_WORD_BOUNDARY));
    let has_wordpiece_marker = tokens.iter().any(|t| t.starts_with("##"));
    if !has_sp_marker || has_wordpiece_marker {
        return tokens;
    }

    tokens
        .into_iter()
        .map(|t| {
            if let Some(stripped) = t.strip_prefix(SP_WORD_BOUNDARY) {
                stripped.to_owned()
            } else if t.starts_with('[') && t.ends_with(']') {
                // [PAD], [CLS], [SEP], [UNK], [unusedN] — never continuations.
                t
            } else {
                format!("##{t}")
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::normalize_wordpiece_vocab;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The `nomic-embed-text-v1.5` shape: `▁`-marked word-initial pieces, bare
    /// continuations, bracketed specials.
    #[test]
    fn sentencepiece_marked_bert_vocab_is_converted_to_wordpiece() {
        let got = normalize_wordpiece_vocab(v(&[
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "▁the", "▁hello", "s", "ing", "▁!", "▁1",
        ]));
        assert_eq!(
            got,
            v(&[
                "[PAD]", "[CLS]", "[SEP]", "[UNK]", "the", "hello", "##s", "##ing", "!", "1",
            ]),
            "▁X must become X, bare X must become ##X, specials must be untouched"
        );
    }

    /// A vocab already in WordPiece convention must be returned byte-identical —
    /// otherwise fixing one model's tokenizer would break every other BERT GGUF.
    #[test]
    fn already_wordpiece_vocab_is_left_untouched() {
        let original = v(&["[PAD]", "[CLS]", "the", "##s", "hello", "!"]);
        assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
    }

    /// Mixed marking (some `▁`, some `##`) means the file is already using the
    /// WordPiece continuation marker, so rewriting would corrupt it.
    #[test]
    fn mixed_marking_is_left_untouched() {
        let original = v(&["▁the", "##s", "hello"]);
        assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
    }

    /// No `▁` anywhere → nothing to convert.
    #[test]
    fn unmarked_vocab_is_left_untouched() {
        let original = v(&["the", "hello", "world"]);
        assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
    }
}
