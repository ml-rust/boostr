//! Lift a GGUF file's `tokenizer.ggml.*` metadata into a [`splintr::GgufVocab`].
//!
//! This module is one half of a deliberate split, and the split is the whole
//! point of the file:
//!
//! - **boostr owns the container.** Reading the GGUF header, the key-value
//!   metadata block and the tensor table is model-runtime work that boostr
//!   already does to load weights, so extracting the vocabulary keys is free
//!   here.
//! - **splintr owns tokenization.** Which algorithm `tokenizer.ggml.model`
//!   names, how the flags map onto llama.cpp's behaviour, and what each absent
//!   key defaults to is tokenizer knowledge, and it lives behind
//!   [`splintr::from_gguf_vocab`].
//!
//! So nothing below inspects `model`, branches on a dialect, or constructs a
//! tokenizer — it only copies metadata into a plain struct. **Do not re-add
//! tokenizer logic here.** The previous arrangement had both crates deciding
//! things like `add_space_prefix`'s default, and the two answers drifted apart
//! silently: the ids stayed in range, the embedding shapes stayed right, and
//! retrieval quality was the only symptom.
//!
//! The same rule explains why every optional field is passed through as `None`
//! when its key is absent rather than filled with a boostr-side default.
//! `None` means "the file does not say"; splintr's loader is the one that knows
//! what that means for each dialect, and a default written here would be a
//! second, competing answer.

use splintr::GgufVocab;

use crate::error::{Error, Result};
use crate::format::GgufMetadata;
use crate::format::gguf::value::GgufValue;

/// Read the `tokenizer.ggml.*` block of a GGUF file into a [`GgufVocab`], ready
/// to hand to [`splintr::from_gguf_vocab`].
///
/// Fails only on data that is present but malformed — a non-string entry in
/// `tokens` or `merges`, a non-numeric `scores` or `token_type` — because such a
/// file cannot be tokenized at all and a silently dropped entry would shift
/// every id after it. A key that is simply missing is never an error.
pub fn extract_gguf_vocab(metadata: &GgufMetadata) -> Result<GgufVocab> {
    Ok(GgufVocab {
        // llama.cpp's fallback when the key is absent, which splintr's contract
        // asks the caller to supply rather than inventing its own.
        model: metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("llama")
            .to_owned(),
        tokens: extract_tokens(metadata)?,
        scores: extract_scores(metadata)?,
        merges: extract_merges(metadata)?,
        token_type: extract_token_type(metadata)?,
        add_space_prefix: flag(metadata, "tokenizer.ggml.add_space_prefix"),
        remove_extra_whitespaces: flag(metadata, "tokenizer.ggml.remove_extra_whitespaces"),
        add_bos_token: flag(metadata, "tokenizer.ggml.add_bos_token"),
        add_eos_token: flag(metadata, "tokenizer.ggml.add_eos_token"),
        bos_token_id: metadata.get_u32("tokenizer.ggml.bos_token_id"),
        eos_token_id: metadata.get_u32("tokenizer.ggml.eos_token_id"),
        unknown_token_id: metadata.get_u32("tokenizer.ggml.unknown_token_id"),
        padding_token_id: metadata.get_u32("tokenizer.ggml.padding_token_id"),
        cls_token_id: metadata.get_u32("tokenizer.ggml.cls_token_id"),
        sep_token_id: metadata.get_u32("tokenizer.ggml.sep_token_id"),
        pre: metadata.get_string("tokenizer.ggml.pre").map(str::to_owned),
        // Carried through verbatim: SentencePiece's normalization table is a
        // darts-clone trie that only splintr's Unigram path knows how to walk.
        // Omitting it is not cosmetic — the characters the table folds (tab,
        // newline, NBSP, ZWJ, fullwidth punctuation) reach Viterbi unnormalized,
        // and a vocabulary that only ever saw their folded forms has no piece
        // for them, so each one degrades to `<unk>`.
        precompiled_charsmap: metadata.get_u8_array("tokenizer.ggml.precompiled_charsmap"),
    })
}

/// Extract token strings from GGUF metadata.
fn extract_tokens(metadata: &GgufMetadata) -> Result<Vec<String>> {
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

/// Extract per-token scores, or `None` when the file carries none.
fn extract_scores(metadata: &GgufMetadata) -> Result<Option<Vec<f32>>> {
    let Some(array) = metadata.get_array("tokenizer.ggml.scores") else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, v) in array.iter().enumerate() {
        out.push(v.as_f32().ok_or_else(|| Error::ModelError {
            reason: format!("tokenizer.ggml.scores[{i}] is not an f32"),
        })?);
    }
    Ok(Some(out))
}

/// Extract the `"a b"` merge list, or `None` when the file carries none.
fn extract_merges(metadata: &GgufMetadata) -> Result<Option<Vec<String>>> {
    let Some(array) = metadata.get_array("tokenizer.ggml.merges") else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, v) in array.iter().enumerate() {
        let s = v.as_string().ok_or_else(|| Error::ModelError {
            reason: format!("tokenizer.ggml.merges[{i}] is not a string"),
        })?;
        out.push(s.to_owned());
    }
    Ok(Some(out))
}

/// Extract the per-id GGUF token-type enum, or `None` when the file carries none.
fn extract_token_type(metadata: &GgufMetadata) -> Result<Option<Vec<u32>>> {
    let Some(array) = metadata.get_array("tokenizer.ggml.token_type") else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, v) in array.iter().enumerate() {
        out.push(v.as_u32().ok_or_else(|| Error::ModelError {
            reason: format!("tokenizer.ggml.token_type[{i}] is not an integer"),
        })?);
    }
    Ok(Some(out))
}

/// Read a boolean GGUF flag, or `None` when the file does not declare it.
fn flag(metadata: &GgufMetadata, key: &str) -> Option<bool> {
    match metadata.get(key) {
        Some(GgufValue::Bool(b)) => Some(*b),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: Vec<(&str, GgufValue)>) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        for (k, v) in pairs {
            m.kv.insert(k.to_string(), v);
        }
        m
    }

    fn tokens_value(tokens: &[&str]) -> GgufValue {
        GgufValue::Array(
            tokens
                .iter()
                .map(|t| GgufValue::String((*t).to_string()))
                .collect(),
        )
    }

    #[test]
    fn absent_keys_stay_none() {
        let m = meta(vec![("tokenizer.ggml.tokens", tokens_value(&["a", "b"]))]);
        let v = extract_gguf_vocab(&m).expect("extract");

        // The default lives in splintr, not here — every optional key the file
        // omits must arrive as `None`.
        assert_eq!(v.model, "llama");
        assert_eq!(v.tokens, vec!["a".to_string(), "b".to_string()]);
        assert!(v.scores.is_none());
        assert!(v.merges.is_none());
        assert!(v.token_type.is_none());
        assert!(v.add_space_prefix.is_none());
        assert!(v.remove_extra_whitespaces.is_none());
        assert!(v.add_bos_token.is_none());
        assert!(v.add_eos_token.is_none());
        assert!(v.bos_token_id.is_none());
        assert!(v.pre.is_none());
        assert!(v.precompiled_charsmap.is_none());
    }

    #[test]
    fn charsmap_round_trips_as_bytes() {
        let bytes = [0u8, 1, 255, 128];
        let m = meta(vec![
            ("tokenizer.ggml.tokens", tokens_value(&["a"])),
            (
                "tokenizer.ggml.precompiled_charsmap",
                GgufValue::Array(bytes.iter().map(|b| GgufValue::Uint8(*b)).collect()),
            ),
        ]);
        let v = extract_gguf_vocab(&m).expect("extract");
        assert_eq!(v.precompiled_charsmap.as_deref(), Some(&bytes[..]));
    }

    #[test]
    fn flags_and_ids_pass_through() {
        let m = meta(vec![
            ("tokenizer.ggml.tokens", tokens_value(&["a"])),
            ("tokenizer.ggml.model", GgufValue::String("t5".into())),
            ("tokenizer.ggml.pre", GgufValue::String("qwen2".into())),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
            (
                "tokenizer.ggml.scores",
                GgufValue::Array(vec![GgufValue::Float32(-1.5)]),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(vec![GgufValue::Uint32(3)]),
            ),
        ]);
        let v = extract_gguf_vocab(&m).expect("extract");
        assert_eq!(v.model, "t5");
        assert_eq!(v.pre.as_deref(), Some("qwen2"));
        assert_eq!(v.add_space_prefix, Some(false));
        assert_eq!(v.add_bos_token, Some(true));
        assert_eq!(v.eos_token_id, Some(2));
        assert_eq!(v.scores, Some(vec![-1.5f32]));
        assert_eq!(v.token_type, Some(vec![3]));
    }

    #[test]
    fn malformed_tokens_are_refused() {
        let m = meta(vec![(
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![GgufValue::Uint32(7)]),
        )]);
        assert!(extract_gguf_vocab(&m).is_err());
    }
}
