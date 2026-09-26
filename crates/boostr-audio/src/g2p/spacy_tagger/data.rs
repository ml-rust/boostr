//! The embedded `en_core_web_sm` tagger weights and metadata.
//!
//! `assets/spacy_en/convert_tagger.py` writes both files and documents the
//! tensor names and JSON fields.

use std::io::Read;

use boostr::model::text::spacy_tagger::SpacyTaggerConfig;
use flate2::read::ZlibDecoder;
use serde::Deserialize;

use crate::error::{Error, Result};

/// `assets/spacy_en/tagger.safetensors.zlib`.
const WEIGHTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/spacy_en/tagger.safetensors.zlib"
));

/// `assets/spacy_en/tagger.json`.
const META: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/spacy_en/tagger.json"
));

/// `tagger.json`.
#[derive(Debug, Deserialize)]
pub(crate) struct TaggerMeta {
    /// spaCy and pipeline versions the weights came from.
    pub source: String,
    /// `FeatureExtractor` columns, in table order.
    pub attrs: Vec<String>,
    /// Network geometry and hash seeds.
    pub config: SpacyTaggerConfig,
    /// Tag per output class.
    pub labels: Vec<String>,
    /// `spacy.symbols.IDS`.
    pub symbols: Vec<(String, u64)>,
}

impl TaggerMeta {
    /// Parses the embedded `tagger.json`.
    pub(crate) fn en_core_web_sm() -> Result<Self> {
        serde_json::from_str(META).map_err(|e| Error::ModelError {
            reason: format!("assets/spacy_en/tagger.json does not parse: {e}"),
        })
    }
}

/// Inflates the embedded SafeTensors image.
pub(crate) fn weights_image() -> Result<Vec<u8>> {
    let mut out = Vec::new();
    ZlibDecoder::new(WEIGHTS)
        .read_to_end(&mut out)
        .map_err(|e| Error::ModelError {
            reason: format!("assets/spacy_en/tagger.safetensors.zlib does not inflate: {e}"),
        })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_meta_parses() {
        let meta = TaggerMeta::en_core_web_sm().unwrap_or_else(|e| panic!("tagger.json: {e}"));
        assert!(meta.source.contains("en_core_web_sm 3.8.0"));
        assert_eq!(meta.attrs.len(), meta.config.seeds.len());
        assert_eq!(meta.labels.len(), 50);
        assert!(meta.labels.iter().any(|l| l == "NNP"));
        assert!(meta.symbols.iter().any(|(s, id)| s == "X" && *id == 101));
    }

    #[test]
    fn embedded_weights_inflate() {
        let image = weights_image().unwrap_or_else(|e| panic!("weights: {e}"));
        let header = u64::from_le_bytes(image[..8].try_into().unwrap_or_default());
        assert!(header > 0 && (header as usize) < image.len());
    }
}
