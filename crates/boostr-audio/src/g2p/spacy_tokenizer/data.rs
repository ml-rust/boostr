//! The embedded `en_core_web_sm` tokenizer data.
//!
//! `assets/spacy_en/tokenizer.json` is written by
//! `assets/spacy_en/build_tokenizer.py`; that script documents the fields
//! and how the Python patterns become `regexr` patterns.

use serde::Deserialize;

use crate::error::{Error, Result};

/// `assets/spacy_en/tokenizer.json`.
const EN_CORE_WEB_SM: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/spacy_en/tokenizer.json"
));

/// One token of a special case: its `ORTH` and optional `NORM`.
pub(crate) type SpecialPiece = (String, Option<String>);

/// The tokenizer settings and lexeme tables of one spaCy pipeline.
#[derive(Debug, Deserialize)]
pub(crate) struct TokenizerData {
    /// spaCy and pipeline versions the data came from.
    pub source: String,
    /// `Tokenizer.faster_heuristics`.
    pub faster_heuristics: bool,
    /// `prefix_search`, for a leftmost search.
    pub prefix: Option<String>,
    /// `suffix_search`, for a leftmost search.
    pub suffix: Option<String>,
    /// `infix_finditer`, for non-overlapping iteration.
    pub infix: Option<String>,
    /// `token_match`, anchored at the start.
    pub token_match: Option<String>,
    /// `url_match`, anchored at the start.
    pub url_match: Option<String>,
    /// `Tokenizer.rules`: chunk and its pieces, sorted by chunk.
    pub rules: Vec<(String, Vec<SpecialPiece>)>,
    /// `spacy.lang.norm_exceptions.BASE_NORMS`.
    pub base_norms: Vec<(String, String)>,
    /// The `lexeme_norm` lookups table, keyed by [`super::hash_string`].
    pub lexeme_norm: Vec<(u64, String)>,
}

impl TokenizerData {
    /// Parses the embedded `en_core_web_sm` 3.8.0 data.
    pub(crate) fn en_core_web_sm() -> Result<Self> {
        serde_json::from_str(EN_CORE_WEB_SM).map_err(|e| Error::ModelError {
            reason: format!("assets/spacy_en/tokenizer.json does not parse: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_data_parses() {
        let data = TokenizerData::en_core_web_sm()
            .unwrap_or_else(|e| panic!("embedded tokenizer data: {e}"));
        assert!(data.source.contains("en_core_web_sm 3.8.0"));
        assert!(data.faster_heuristics);
        assert!(data.prefix.is_some() && data.suffix.is_some() && data.infix.is_some());
        assert!(data.token_match.is_none() && data.url_match.is_some());
        assert!(data.rules.iter().any(|(chunk, _)| chunk == "don't"));
        assert!(!data.base_norms.is_empty() && !data.lexeme_norm.is_empty());
    }
}
