//! `tagger.json` metadata and `tagger.safetensors` weights, read from a
//! directory at run time.
//!
//! `assets/spacy_en/convert_tagger.py` writes both files and documents the
//! tensor names and JSON fields.

use std::path::Path;

use boostr::model::text::spacy_tagger::SpacyTaggerConfig;
use serde::Deserialize;

use crate::error::{Error, Result};

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
    /// Parses `dir/tagger.json`.
    pub(crate) fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tagger.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::ModelError {
            reason: format!("reading {}: {e}", path.display()),
        })?;
        serde_json::from_str(&text).map_err(|e| Error::ModelError {
            reason: format!("{} does not parse: {e}", path.display()),
        })
    }
}

/// Reads `dir/tagger.safetensors` (plain, uncompressed SafeTensors bytes).
pub(crate) fn weights_image(dir: &Path) -> Result<Vec<u8>> {
    let path = dir.join("tagger.safetensors");
    std::fs::read(&path).map_err(|e| Error::ModelError {
        reason: format!("reading {}: {e}", path.display()),
    })
}
