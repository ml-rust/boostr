//! The embedded g2p_lm bundle: `config.json`, `vocab.json` and a
//! zlib-compressed SafeTensors image, written by `assets/g2p_lm/pack_model.py`.
//!
//! `build.rs` refuses to compile this feature until those three files exist
//! (`assets/g2p_lm/{model.safetensors.zlib,config.json,vocab.json}`), naming
//! the packer command to produce them, so an `include_bytes!` failure here
//! never reaches a consumer as a bare "file not found".

use std::io::Read;

use flate2::read::ZlibDecoder;

use crate::error::{Error, Result};

use super::model::NeuralG2pFallback;

const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/g2p_lm/config.json"
));
const VOCAB: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/g2p_lm/vocab.json"
));
const WEIGHTS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/g2p_lm/model.safetensors.zlib"
));

impl NeuralG2pFallback {
    /// Builds [`NeuralG2pFallback`] from the embedded g2p_lm bundle.
    /// `british`: `true` for the `gb` dialect, `false` for `us` (see
    /// [`NeuralG2pFallback::from_dir`]).
    pub fn from_embedded(british: bool) -> Result<Self> {
        let mut weights = Vec::new();
        ZlibDecoder::new(WEIGHTS)
            .read_to_end(&mut weights)
            .map_err(|e| Error::ModelError {
                reason: format!("embedded g2p_lm model.safetensors.zlib does not inflate: {e}"),
            })?;
        Self::from_parts(CONFIG, VOCAB, weights, british)
    }
}
