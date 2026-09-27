//! `EnglishTagger`: spaCy `en_core_web_sm` 3.8.0 fine tags (`Token.tag_`)
//! for tokens from [`SpacyTokenizer`](crate::g2p::spacy_tokenizer::SpacyTokenizer).

use std::path::Path;

use boostr::model::text::spacy_tagger::SpacyTagger;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

use super::attrs::{Attr, StringIds};
use super::data::{TaggerMeta, weights_image};
use crate::error::{Error, Result};
use crate::g2p::spacy_tokenizer::SpacyToken;

/// The `tok2vec` + `tagger` components of `en_core_web_sm` 3.8.0, as
/// `spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])` runs them.
///
/// Weights load from a directory (`tagger.safetensors` + `tagger.json`,
/// written by `assets/spacy_en/convert_tagger.py`) and run on numr's CPU
/// backend. Each call tags one document, as `nlp(text)` does.
pub struct EnglishTagger {
    model: SpacyTagger<CpuRuntime>,
    client: CpuClient,
    ids: StringIds,
    attrs: Vec<Attr>,
    labels: Vec<String>,
    source: String,
}

impl std::fmt::Debug for EnglishTagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnglishTagger")
            .field("source", &self.source)
            .field("attrs", &self.attrs)
            .field("labels", &self.labels.len())
            .finish()
    }
}

impl EnglishTagger {
    /// Reads `tagger.safetensors` + `tagger.json` from `dir` and builds the
    /// network. `dir` is normally `<kokoro dir>/g2p/tagger` or
    /// `$BOOSTR_MODELS_DIR/spacy-en-tagger`; see
    /// `assets/spacy_en/convert_tagger.py` for how to produce it.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let meta = TaggerMeta::from_dir(dir)?;
        let attrs = meta
            .attrs
            .iter()
            .map(|a| Attr::from_name(a))
            .collect::<Result<Vec<_>>>()?;
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let model = SpacyTagger::from_safetensors_bytes(&client, meta.config, weights_image(dir)?)?;
        if model.n_labels() != meta.labels.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "spaCy tagger at {} has {} output classes and {} labels",
                    dir.display(),
                    model.n_labels(),
                    meta.labels.len()
                ),
            });
        }
        Ok(Self {
            model,
            client,
            ids: StringIds::new(meta.symbols),
            attrs,
            labels: meta.labels,
            source: meta.source,
        })
    }

    /// The spaCy and pipeline versions the weights came from.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The tag set, in output-class order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Width of one [`Self::tok2vec`] row.
    pub fn width(&self) -> usize {
        self.model.config().width
    }

    /// Penn Treebank tag per token, `Token.tag_`. An empty slice gives an
    /// empty vector.
    pub fn tag(&self, tokens: &[SpacyToken]) -> Result<Vec<String>> {
        let features = self.ids.features(tokens, &self.attrs);
        self.model
            .predict(&self.client, &features)?
            .into_iter()
            .map(|i| {
                self.labels
                    .get(i)
                    .cloned()
                    .ok_or_else(|| Error::ModelError {
                        reason: format!("spaCy tagger class {i} has no label"),
                    })
            })
            .collect()
    }

    /// The `tok2vec` output: `tokens.len()` rows of [`Self::width`] floats,
    /// row-major.
    pub fn tok2vec(&self, tokens: &[SpacyToken]) -> Result<Vec<f32>> {
        let features = self.ids.features(tokens, &self.attrs);
        Ok(self.model.tok2vec(&self.client, &features)?.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g2p::spacy_tagger::test_support::tagger_dir;
    use crate::g2p::spacy_tokenizer::SpacyTokenizer;

    macro_rules! require_tagger_dir {
        () => {
            match tagger_dir() {
                Some(dir) => dir,
                None => {
                    eprintln!(
                        "skipping: spaCy tagger dir unavailable; set SPACY_EN_TAGGER_DIR or \
                         BOOSTR_MODELS_DIR"
                    );
                    return;
                }
            }
        };
    }

    #[test]
    fn tags_a_sentence() {
        let dir = require_tagger_dir!();
        let tokenizer =
            SpacyTokenizer::en_core_web_sm().unwrap_or_else(|e| panic!("tokenizer: {e}"));
        let tagger = EnglishTagger::from_dir(&dir).unwrap_or_else(|e| panic!("tagger: {e}"));
        let tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog.");
        let tags = tagger.tag(&tokens).unwrap_or_else(|e| panic!("tag: {e}"));
        // spaCy 3.8.14, en_core_web_sm 3.8.0, tok2vec + tagger.
        assert_eq!(
            tags,
            ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN", "."]
        );
        let vectors = tagger
            .tok2vec(&tokens)
            .unwrap_or_else(|e| panic!("tok2vec: {e}"));
        assert_eq!(vectors.len(), tokens.len() * tagger.width());
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EnglishTagger>();
    }

    #[test]
    fn empty_document() {
        let dir = require_tagger_dir!();
        let tagger = EnglishTagger::from_dir(&dir).unwrap_or_else(|e| panic!("tagger: {e}"));
        assert!(
            tagger
                .tag(&[])
                .unwrap_or_else(|e| panic!("tag: {e}"))
                .is_empty()
        );
        assert!(
            tagger
                .tok2vec(&[])
                .unwrap_or_else(|e| panic!("tok2vec: {e}"))
                .is_empty()
        );
    }
}
