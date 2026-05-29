//! Kokoro phoneme vocabulary — IPA/punctuation string → integer id.
//!
//! Upstream ships a `vocab.json` alongside the checkpoint: a flat
//! `{"symbol": id}` object with 178 entries. This module parses that file
//! (or accepts an in-memory mapping) and implements
//! [`crate::model::audio::g2p::PhonemeVocab`].
//!
//! We don't hardcode the 178-entry table here. The table is upstream data
//! that ships with the model directory; hardcoding it would mean drift every
//! time Kokoro ships a new vocab. Reading the checkpoint-bundled `vocab.json`
//! is the cleanest path and matches how other Kokoro ports (Python, ONNX
//! runtime demos) behave.

use crate::error::{Error, Result};
use crate::model::audio::g2p::PhonemeVocab;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Kokoro phoneme vocab. Thin wrapper around `HashMap<String, u32>` that
/// implements `PhonemeVocab` and adds file-loading conveniences.
#[derive(Debug, Clone, Default)]
pub struct KokoroPhonemeVocab {
    table: HashMap<String, u32>,
}

impl KokoroPhonemeVocab {
    pub fn new(table: HashMap<String, u32>) -> Self {
        Self { table }
    }

    /// Parse from on-disk `vocab.json`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
            reason: format!("reading kokoro vocab {}: {e}", path.display()),
        })?;
        Self::from_json_bytes(&bytes)
    }

    /// Parse from in-memory JSON bytes.
    ///
    /// Accepts three shapes:
    /// * Flat `{"symbol": id}` — a standalone `vocab.json` file.
    /// * Wrapped `{"vocab": {...}}` — some forks stash it under a single key.
    /// * **Kokoro's inlined form**: the full `config.json` with a `"vocab"`
    ///   field alongside other model fields. Upstream ships exactly this.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            /// `{"vocab": {...}, ...other fields...}` — Kokoro's `config.json`.
            /// Must be tried first because it matches more permissively.
            WithVocabKey { vocab: HashMap<String, u32> },
            /// Flat `{"symbol": id}` — a standalone `vocab.json`.
            Flat(HashMap<String, u32>),
        }
        let parsed: Raw = serde_json::from_slice(bytes).map_err(|e| Error::ModelError {
            reason: format!("invalid kokoro vocab JSON: {e}"),
        })?;
        let table = match parsed {
            Raw::Flat(m) => m,
            Raw::WithVocabKey { vocab } => vocab,
        };
        if table.is_empty() {
            return Err(Error::ModelError {
                reason: "kokoro vocab is empty".into(),
            });
        }
        Ok(Self { table })
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Encode a sequence of phoneme tokens into integer ids. Tokens not in
    /// the vocab are skipped; caller can choose whether that's acceptable or
    /// should trigger an error based on how strict their pipeline is.
    pub fn encode_skipping_unknown(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .filter_map(|t| self.table.get(t).copied())
            .collect()
    }

    /// Strict encoding — errors on the first unknown token, reporting which
    /// one fell through. Suited for production paths that prefer failing loud
    /// over silently dropping phonemes.
    pub fn encode_strict(&self, tokens: &[String]) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            match self.table.get(t) {
                Some(&id) => out.push(id),
                None => {
                    return Err(Error::ModelError {
                        reason: format!("phoneme {t:?} at position {i} not in vocab"),
                    });
                }
            }
        }
        Ok(out)
    }
}

impl PhonemeVocab for KokoroPhonemeVocab {
    fn lookup(&self, phoneme: &str) -> Option<u32> {
        self.table.get(phoneme).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_json() {
        let json = r#"{"h": 1, "ɛ": 2, "l": 3}"#.as_bytes();
        let vocab = KokoroPhonemeVocab::from_json_bytes(json).unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.lookup("h"), Some(1));
        assert_eq!(vocab.lookup("?"), None);
    }

    #[test]
    fn parses_wrapped_json() {
        let json = br#"{"vocab": {"a": 10, "b": 20}}"#;
        let vocab = KokoroPhonemeVocab::from_json_bytes(json).unwrap();
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.lookup("a"), Some(10));
    }

    #[test]
    fn rejects_empty_vocab() {
        let json = br#"{}"#;
        assert!(KokoroPhonemeVocab::from_json_bytes(json).is_err());
    }

    #[test]
    fn rejects_invalid_json() {
        let json = b"not json";
        assert!(KokoroPhonemeVocab::from_json_bytes(json).is_err());
    }

    #[test]
    fn encode_skipping_unknown_drops_misses() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 1);
        m.insert("b".to_string(), 2);
        let vocab = KokoroPhonemeVocab::new(m);
        let ids = vocab.encode_skipping_unknown(&["a".into(), "x".into(), "b".into(), "y".into()]);
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn encode_strict_errors_on_first_unknown() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 1);
        let vocab = KokoroPhonemeVocab::new(m);
        let err = vocab
            .encode_strict(&["a".into(), "nope".into()])
            .unwrap_err();
        match err {
            Error::ModelError { reason } => assert!(reason.contains("nope")),
            _ => panic!("wrong error variant"),
        }
    }
}
