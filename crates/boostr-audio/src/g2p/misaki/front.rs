//! misaki's text front end: `G2P.preprocess`, then `G2P.tokenize` with the
//! spaCy `en_core_web_sm` tokenizer and tagger.

use std::sync::OnceLock;

use super::preprocess::{apply_features, preprocess};
use super::token::MToken;
use crate::error::Result;
use crate::g2p::spacy_tagger::EnglishTagger;
use crate::g2p::spacy_tokenizer::SpacyTokenizer;

/// Raw text to the tagged tokens misaki's G2P loop consumes. Dialect-
/// independent, so US and GB `MisakiG2p` share one instance through
/// [`TextFrontEnd::shared`] instead of each building their own 5.8 MB
/// tokenizer and tagger.
#[derive(Debug)]
pub(crate) struct TextFrontEnd {
    tokenizer: SpacyTokenizer,
    tagger: EnglishTagger,
}

static SHARED: OnceLock<TextFrontEnd> = OnceLock::new();

impl TextFrontEnd {
    /// Builds the embedded tokenizer and tagger.
    fn new() -> Result<Self> {
        Ok(Self {
            tokenizer: SpacyTokenizer::en_core_web_sm()?,
            tagger: EnglishTagger::new()?,
        })
    }

    /// The process-wide instance, built on first use and reused by every
    /// dialect and engine.
    pub(crate) fn shared() -> Result<&'static Self> {
        if let Some(front) = SHARED.get() {
            return Ok(front);
        }
        let built = Self::new()?;
        Ok(SHARED.get_or_init(|| built))
    }

    /// misaki `G2P.preprocess(text)` then `G2P.tokenize`: one [`MToken`] per
    /// spaCy token with its fine tag and whitespace, link features applied.
    pub(crate) fn tokens(&self, text: &str) -> Result<Vec<MToken>> {
        let pre = preprocess(text);
        let spacy = self.tokenizer.tokenize(&pre.text);
        let tags = self.tagger.tag(&spacy)?;
        let mut tokens: Vec<MToken> = spacy
            .iter()
            .zip(tags)
            .map(|(t, tag)| MToken::new(t.text.as_str(), tag, t.whitespace()))
            .collect();
        apply_features(&mut tokens, &pre.words, &pre.features)?;
        Ok(tokens)
    }
}
