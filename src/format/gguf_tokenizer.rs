//! GGUF-embedded tokenizer with algorithm selection.
//!
//! `tokenizer.ggml.model` names which tokenization *algorithm* the vocabulary
//! was built with, and the four values in circulation are genuinely different
//! algorithms over superficially similar data:
//!
//! | `model` | scores | merges | algorithm |
//! |---|---|---|---|
//! | `bert` | ignored | — | WordPiece, greedy longest match with `##` |
//! | `t5` | log-probabilities | — | Unigram, Viterbi maximum-sum segmentation |
//! | `llama` | **merge ranks** | — | SentencePiece BPE, best adjacent pair first |
//! | `gpt2` | — | **required** | byte-level BPE over an explicit merge list |
//!
//! Collapsing these is not a rounding error. Running Unigram Viterbi over a
//! `llama` vocabulary maximises the wrong objective — its scores are ranks, so
//! cheap early-id fragments outscore whole words and `▁sourdough` becomes
//! `▁s|ou|rd|ou|gh`. Running it over a `gpt2` vocabulary is worse still: those
//! files carry no scores at all, so every token scores equally and the `merges`
//! list that defines the tokenizer is never read.
//!
//! Neither failure is visible downstream. The ids are in range, the embedding
//! shapes are right, and retrieval quietly degrades to near-noise. So the
//! routing below dispatches on the declared model and rejects what it cannot
//! honour rather than guessing.

use crate::error::{Error, Result};
use crate::format::Gguf;
use crate::format::gguf::value::GgufValue;
use rustc_hash::FxHashMap;
use splintr::{
    QWEN2_PATTERN, SentencePieceTokenizer, SpmTokenizer, Tokenize, TokenizeError, Tokenizer,
    WordPieceTokenizer,
};

/// Tokenizer that uses the vocabulary embedded in a GGUF file.
pub struct GgufTokenizer {
    inner: Box<dyn Tokenize>,
    eos_token_id: u32,
    /// Sentence-start token to prepend, per `tokenizer.ggml.add_bos_token`.
    prepend_bos: Option<u32>,
    /// Sentence-end token to append, per `tokenizer.ggml.add_eos_token`.
    append_eos: Option<u32>,
}

/// Which sentence-boundary tokens a GGUF asks to be added.
///
/// Applied here rather than inside each backend because the three backends
/// disagree: the Unigram tokenizer prepends BOS and never appends EOS, the BPE
/// tokenizer adds neither. Reading the file's own `add_bos_token` /
/// `add_eos_token` in one place makes the file authoritative for every
/// architecture instead of inheriting whichever convention a backend happened
/// to implement.
///
/// This matters beyond tidiness for last-token pooling: Qwen3-Embedding is
/// trained to summarise a sequence into its final `<|endoftext|>` position, so a
/// dropped EOS means pooling reads a content token instead of the summary.
struct SentenceBoundaries {
    bos: Option<u32>,
    eos: Option<u32>,
}

impl SentenceBoundaries {
    /// Read the flags, defaulting each to `bos_default` / `eos_default` when the
    /// file does not say.
    fn read(metadata: &crate::format::GgufMetadata, bos_default: bool, eos_default: bool) -> Self {
        Self {
            bos: add_special(metadata, "tokenizer.ggml.add_bos_token", bos_default)
                .then(|| metadata.get_u32("tokenizer.ggml.bos_token_id"))
                .flatten(),
            eos: add_special(metadata, "tokenizer.ggml.add_eos_token", eos_default)
                .then(|| metadata.get_u32("tokenizer.ggml.eos_token_id"))
                .flatten(),
        }
    }

    /// No boundary tokens — for BERT, which wraps with `[CLS]`/`[SEP]` instead.
    fn none() -> Self {
        Self {
            bos: None,
            eos: None,
        }
    }
}

impl GgufTokenizer {
    /// Create a tokenizer from GGUF metadata, dispatching on
    /// `tokenizer.ggml.model`.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let metadata = gguf.metadata();
        let model_type = metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("llama");
        let tokens = extract_tokens(metadata)?;

        match model_type {
            "bert" => Self::build_wordpiece(metadata, tokens),
            "t5" => Self::build_unigram(metadata, tokens),
            "llama" => Self::build_spm(metadata, tokens),
            "gpt2" => Self::build_byte_level_bpe(metadata, tokens),
            other => Err(Error::ModelError {
                reason: format!(
                    "unsupported tokenizer.ggml.model '{other}'. Supported: bert \
                     (WordPiece), t5 (Unigram), llama (SentencePiece BPE), gpt2 \
                     (byte-level BPE)."
                ),
            }),
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

        let unk_token_id = find_special_token_id(&tokens, metadata, "[UNK]", 0);

        // GGUF has no standard key for casing, so heuristic: a vocab holding
        // lowercase "the" but not "The" is uncased.
        let do_lower_case = tokens.iter().any(|t| t == "the") && !tokens.iter().any(|t| t == "The");

        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);
        let inner = WordPieceTokenizer::new(tokens, unk_token_id, 200, do_lower_case);
        Ok(Self::assemble(
            Box::new(inner),
            eos_token_id,
            SentenceBoundaries::none(),
        ))
    }

    /// Finish a tokenizer from its backend and boundary policy.
    fn assemble(
        inner: Box<dyn Tokenize>,
        eos_token_id: u32,
        boundaries: SentenceBoundaries,
    ) -> Self {
        Self {
            inner,
            eos_token_id,
            prepend_bos: boundaries.bos,
            append_eos: boundaries.eos,
        }
    }

    /// `t5`: true Unigram. Scores are log-probabilities and Viterbi is correct.
    fn build_unigram(metadata: &crate::format::GgufMetadata, tokens: Vec<String>) -> Result<Self> {
        let scores = extract_scores(metadata)?;
        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        // `None` for BOS: boundary tokens are added by `encode` below, so the
        // backend must not also prepend one.
        let inner = SentencePieceTokenizer::new(tokens, scores, None, eos_token_id)
            .map_err(|e| Error::ModelError {
                reason: format!("Unigram tokenizer: {e}"),
            })?
            .with_prefix_space(add_space_prefix(metadata, true));

        Ok(Self::assemble(
            Box::new(inner),
            eos_token_id,
            SentenceBoundaries::read(metadata, true, true),
        ))
    }

    /// `llama`: SentencePiece BPE. Scores are merge ranks, so segmentation is
    /// repeated best-adjacent-pair merging, not Viterbi.
    fn build_spm(metadata: &crate::format::GgufMetadata, tokens: Vec<String>) -> Result<Self> {
        let scores = extract_scores(metadata)?;
        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        let inner = SpmTokenizer::new(tokens, scores, None, None)
            .map_err(|e| Error::ModelError {
                reason: format!("SentencePiece BPE tokenizer: {e}"),
            })?
            .with_prefix_space(add_space_prefix(metadata, true));

        Ok(Self::assemble(
            Box::new(inner),
            eos_token_id,
            SentenceBoundaries::read(metadata, true, false),
        ))
    }

    /// `gpt2`: byte-level BPE. The `merges` list defines the tokenizer, so a
    /// file without one cannot be tokenized correctly and is refused.
    fn build_byte_level_bpe(
        metadata: &crate::format::GgufMetadata,
        tokens: Vec<String>,
    ) -> Result<Self> {
        let merges =
            metadata
                .get_array("tokenizer.ggml.merges")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF declares tokenizer.ggml.model = gpt2 but carries no \
                         tokenizer.ggml.merges; byte-level BPE is defined by that \
                         list and cannot be reconstructed from the vocabulary alone"
                        .into(),
                })?;

        // Token strings are already byte-level-encoded ("Ġhello"); the encoder is
        // keyed on those bytes because encode byte-level-encodes before lookup.
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.reserve(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            encoder.insert(token.as_bytes().to_vec(), id as u32);
        }

        let merge_ranks = build_merge_ranks(merges, &tokens);
        let specials = special_token_map(metadata, &tokens);

        let eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);

        let inner = Tokenizer::new_byte_level(encoder, specials, QWEN2_PATTERN)
            .map_err(|e| Error::ModelError {
                reason: format!("byte-level BPE tokenizer: {e}"),
            })?
            .with_merge_ranks(merge_ranks)
            .with_added_token_matching(true);

        Ok(Self::assemble(
            Box::new(inner),
            eos_token_id,
            SentenceBoundaries::read(metadata, false, false),
        ))
    }

    /// Encode, wrapping the backend's ids in whatever boundary tokens the file
    /// declares.
    fn encode_with_boundaries(&self, text: &str) -> Vec<u32> {
        let body = self.inner.encode(text);
        if self.prepend_bos.is_none() && self.append_eos.is_none() {
            return body;
        }
        let mut out = Vec::with_capacity(body.len() + 2);
        out.extend(self.prepend_bos);
        out.extend(body);
        out.extend(self.append_eos);
        out
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
        self.encode_with_boundaries(text)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids).map_err(|e| Error::ModelError {
            reason: format!("Decode error: {e}"),
        })
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

impl Tokenize for GgufTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with_boundaries(text)
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
                    reason: format!("tokenizer.ggml.tokens[{id}] is not a string"),
                });
            }
        }
    }
    Ok(tokens)
}

/// Extract per-token scores, or an empty vec when the file carries none.
fn extract_scores(metadata: &crate::format::GgufMetadata) -> Result<Vec<f32>> {
    let Some(array) = metadata.get_array("tokenizer.ggml.scores") else {
        return Ok(vec![]);
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, v) in array.iter().enumerate() {
        out.push(v.as_f32().ok_or_else(|| Error::ModelError {
            reason: format!("tokenizer.ggml.scores[{i}] is not an f32"),
        })?);
    }
    Ok(out)
}

/// Read a boolean GGUF flag, falling back to `default` when absent.
fn add_special(metadata: &crate::format::GgufMetadata, key: &str, default: bool) -> bool {
    match metadata.get(key) {
        Some(GgufValue::Bool(b)) => *b,
        _ => default,
    }
}

/// SentencePiece `add_dummy_prefix` (`tokenizer.ggml.add_space_prefix`).
fn add_space_prefix(metadata: &crate::format::GgufMetadata, default: bool) -> bool {
    add_special(metadata, "tokenizer.ggml.add_space_prefix", default)
}

/// Build a bytes → merge-rank map from the GGUF `merges` list.
///
/// Ranks are assigned so the base alphabet (vocab entries that are never a merge
/// result) always merges before any real merge, then merges follow in list
/// order. Mirrors the equivalent construction for HuggingFace `tokenizer.json`,
/// because merge priority is independent of token id.
fn build_merge_ranks(merges: &[GgufValue], tokens: &[String]) -> FxHashMap<Vec<u8>, u32> {
    // Each entry is "a b"; byte-level tokens encode real spaces as `Ġ`, so the
    // first space is always the separator.
    let merged: Vec<String> = merges
        .iter()
        .filter_map(|m| m.as_string())
        .map(|s| s.replacen(' ', "", 1))
        .collect();

    let merge_set: std::collections::HashSet<&str> = merged.iter().map(String::as_str).collect();
    let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

    // Base alphabet first, in id order for determinism.
    for token in tokens.iter().filter(|t| !merge_set.contains(t.as_str())) {
        let next = ranks.len() as u32;
        ranks.entry(token.as_bytes().to_vec()).or_insert(next);
    }

    let base_count = ranks.len() as u32;
    for (i, token) in merged.iter().enumerate() {
        ranks
            .entry(token.as_bytes().to_vec())
            .or_insert(base_count + i as u32);
    }
    ranks
}

/// Map of special/control token strings to ids, for added-token matching.
fn special_token_map(
    metadata: &crate::format::GgufMetadata,
    tokens: &[String],
) -> FxHashMap<String, u32> {
    let mut specials = FxHashMap::default();

    // token_type 3 = CONTROL in the GGUF vocabulary enum.
    if let Some(types) = metadata.get_array("tokenizer.ggml.token_type") {
        for (id, t) in types.iter().enumerate() {
            if t.as_u32() == Some(3)
                && let Some(tok) = tokens.get(id)
            {
                specials.insert(tok.clone(), id as u32);
            }
        }
    }
    specials
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

/// Find a special token ID, checking the vocab for the token string first, then
/// falling back to the metadata key.
fn find_special_token_id(
    tokens: &[String],
    metadata: &crate::format::GgufMetadata,
    token_str: &str,
    default: u32,
) -> u32 {
    for (id, t) in tokens.iter().enumerate() {
        if t == token_str {
            return id as u32;
        }
    }

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
#[path = "gguf_tokenizer_tests.rs"]
mod tests;
