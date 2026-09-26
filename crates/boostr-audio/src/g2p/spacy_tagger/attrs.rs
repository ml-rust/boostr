//! spaCy's `FeatureExtractor` input: `Doc.to_array(attrs)` for the lexical
//! attributes `tok2vec` embeds.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::g2p::spacy_tokenizer::{SpacyToken, hash_string};

/// One `FeatureExtractor` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    /// `NORM`: ID of `Token.norm_`.
    Norm,
    /// `PREFIX`: ID of `Token.prefix_`.
    Prefix,
    /// `SUFFIX`: ID of `Token.suffix_`.
    Suffix,
    /// `SHAPE`: ID of `Token.shape_`.
    Shape,
    /// `SPACY`: 1 when a space follows the token, else 0.
    Spacy,
    /// `IS_SPACE`: 1 when the token is whitespace, else 0.
    IsSpace,
}

impl Attr {
    /// Parses a spaCy attribute name such as `"NORM"`.
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "NORM" => Self::Norm,
            "PREFIX" => Self::Prefix,
            "SUFFIX" => Self::Suffix,
            "SHAPE" => Self::Shape,
            "SPACY" => Self::Spacy,
            "IS_SPACE" => Self::IsSpace,
            other => {
                return Err(Error::ModelError {
                    reason: format!(
                        "spaCy tagger attribute {other:?} is not one of NORM, PREFIX, SUFFIX, \
                         SHAPE, SPACY, IS_SPACE"
                    ),
                });
            }
        })
    }
}

/// spaCy `StringStore` string-to-ID mapping: a symbol ID for a string in
/// `spacy.symbols.IDS`, else [`hash_string`].
#[derive(Debug, Clone)]
pub struct StringIds {
    symbols: HashMap<String, u64>,
}

impl StringIds {
    /// From the `spacy.symbols.IDS` table.
    pub fn new(symbols: impl IntoIterator<Item = (String, u64)>) -> Self {
        Self {
            symbols: symbols.into_iter().collect(),
        }
    }

    /// `StringStore.add(s)`: the ID `Doc.to_array` reports for `s`.
    pub fn id(&self, s: &str) -> u64 {
        self.symbols
            .get(s)
            .copied()
            .unwrap_or_else(|| hash_string(s))
    }

    /// `Doc.to_array(attrs)` as `u64`: `tokens.len()` rows of `attrs.len()`
    /// IDs, row-major.
    pub fn features(&self, tokens: &[SpacyToken], attrs: &[Attr]) -> Vec<u64> {
        let mut out = Vec::with_capacity(tokens.len() * attrs.len());
        for t in tokens {
            for attr in attrs {
                out.push(match attr {
                    Attr::Norm => self.id(&t.norm),
                    Attr::Prefix => self.id(&t.prefix),
                    Attr::Suffix => self.id(&t.suffix),
                    Attr::Shape => self.id(&t.shape),
                    Attr::Spacy => u64::from(t.spacy),
                    Attr::IsSpace => u64::from(t.is_space),
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> StringIds {
        StringIds::new([
            (String::new(), 0),
            ("X".to_string(), 101),
            ("nsubj".to_string(), 429),
        ])
    }

    fn token(text: &str, norm: &str, shape: &str, spacy: bool) -> SpacyToken {
        SpacyToken {
            text: text.to_string(),
            spacy,
            norm: norm.to_string(),
            prefix: text.chars().take(1).collect(),
            suffix: text.to_string(),
            shape: shape.to_string(),
            is_space: false,
        }
    }

    #[test]
    fn symbols_take_precedence_over_hashes() {
        let ids = ids();
        assert_eq!(ids.id("X"), 101);
        assert_eq!(ids.id("nsubj"), 429);
        assert_eq!(ids.id(""), 0);
        assert_eq!(ids.id("because"), 16950148841647037698);
    }

    #[test]
    fn features_match_doc_to_array() {
        // `nlp.make_doc("I nsubj").to_array([NORM, PREFIX, SHAPE, SPACY, IS_SPACE])`
        // in spaCy 3.8 with en_core_web_sm 3.8.0.
        let tokens = [
            token("I", "i", "X", true),
            token("nsubj", "nsubj", "xxxx", false),
        ];
        let attrs = [
            Attr::Norm,
            Attr::Prefix,
            Attr::Shape,
            Attr::Spacy,
            Attr::IsSpace,
        ];
        let got = ids().features(&tokens, &attrs);
        assert_eq!(
            got,
            [
                hash_string("i"),
                hash_string("I"),
                101,
                1,
                0,
                429,
                hash_string("n"),
                hash_string("xxxx"),
                0,
                0
            ]
        );
    }

    #[test]
    fn attr_names_parse() {
        assert_eq!(Attr::from_name("IS_SPACE").ok(), Some(Attr::IsSpace));
        assert!(Attr::from_name("LOWER").is_err());
    }
}
