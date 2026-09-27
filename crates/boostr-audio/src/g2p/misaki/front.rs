//! misaki's text front end: `G2P.preprocess`, then `G2P.tokenize` with the
//! spaCy `en_core_web_sm` tokenizer and tagger.

use super::preprocess::{apply_features, preprocess};
use super::token::MToken;
use crate::error::Result;
use crate::g2p::spacy_tagger::EnglishTagger;
use crate::g2p::spacy_tokenizer::SpacyTokenizer;

/// Raw text to the tagged tokens misaki's G2P loop consumes. Dialect-
/// independent: callers build one instance and share it (`Arc`, via
/// [`crate::g2p::misaki::MisakiG2p::with_text_front`]) between the US and GB
/// `MisakiG2p` instead of each loading their own tagger.
#[derive(Debug)]
pub struct TextFrontEnd {
    tokenizer: SpacyTokenizer,
    tagger: EnglishTagger,
}

impl TextFrontEnd {
    /// Builds the embedded tokenizer and pairs it with an already-loaded
    /// `tagger` (see [`EnglishTagger::from_dir`]).
    pub fn new(tagger: EnglishTagger) -> Result<Self> {
        Ok(Self {
            tokenizer: SpacyTokenizer::en_core_web_sm()?,
            tagger,
        })
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
