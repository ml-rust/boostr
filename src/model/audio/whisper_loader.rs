//! Load a standalone Whisper checkpoint (HF layout) into a [`WhisperBundle`].
//!
//! Expected directory layout (what `openai/whisper-*` ships on HuggingFace):
//! ```text
//! <dir>/
//!   config.json                    # architecture + vocab_size + hidden sizes
//!   tokenizer.json                 # byte-level BPE vocab + merges
//!   model.safetensors              # weights (all prefixes under `model.*`)
//!   generation_config.json         # optional — ignored here, consumers of the
//!                                  # bundle can parse it separately if needed
//!   preprocessor_config.json       # optional — mel params (num_mel_bins, etc.)
//! ```
//!
//! Auto-detects the [`WhisperVariant`] from config.json:
//! - `"_name_or_path"` containing `"v3"` / `"large-v3"` → V3
//! - `"_name_or_path"` containing `".en"` or `num_languages == 0` → EnglishOnly
//! - `vocab_size == 51866` → V3
//! - `vocab_size == 51864` → EnglishOnly
//! - otherwise → V2 (the common multilingual case)

use std::path::{Path, PathBuf};

use serde::Deserialize;
use splintr::{Tokenizer, WhisperVariant};

use crate::error::{Error, Result};
use crate::model::audio::whisper_model::WhisperModel;
use crate::model::config::AudioConfig;
use crate::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Everything a caller needs to run Whisper transcription: the model, the
/// tokenizer, and the variant metadata that tells callers which language tokens
/// / control tokens to emit as the SOT prompt.
pub struct WhisperBundle<R: Runtime> {
    pub model: WhisperModel<R>,
    pub tokenizer: Tokenizer,
    pub variant: WhisperVariant,
    pub config: AudioConfig,
    /// Number of mel filterbank bins (80 for tiny/base/small/medium/large, 128 for v3).
    pub num_mel_bins: usize,
}

impl<R: Runtime<DType = DType>> WhisperBundle<R> {
    /// Load a bundle from an HF-style Whisper checkpoint directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
        let dir = dir.as_ref();

        let cfg_path = dir.join("config.json");
        let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| Error::ModelError {
            reason: format!("reading {}: {e}", cfg_path.display()),
        })?;
        let hf: HfWhisperConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| Error::ModelError {
                reason: format!("parsing {}: {e}", cfg_path.display()),
            })?;

        let variant = detect_variant(&hf);
        let audio_config = hf.to_audio_config();
        let num_mel_bins = hf.num_mel_bins.unwrap_or(80);

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = splintr::whisper_tokenizer_from_path(&tok_path, variant).map_err(|e| {
            Error::ModelError {
                reason: format!("loading {}: {e}", tok_path.display()),
            }
        })?;

        let weights_path = find_safetensors(dir)?;
        let mut varmap = VarMap::<R>::from_safetensors(&weights_path, device)?;
        let mut vb = VarBuilder::new(&mut varmap, device);
        let model = WhisperModel::from_varbuilder(&mut vb, &audio_config)?;

        Ok(Self {
            model,
            tokenizer,
            variant,
            config: audio_config,
            num_mel_bins,
        })
    }

    /// Build the "start-of-transcript" prompt for greedy decoding.
    ///
    /// Layout (multilingual): `[<|sot|>, <|lang|>, <|task|>, <|notimestamps|>]`.
    /// Layout (english-only):  `[<|sot|>, <|transcribe|>, <|notimestamps|>]` —
    /// but english-only checkpoints don't have `<|transcribe|>` in their
    /// special table, so we fall through to just `[<|sot|>, <|notimestamps|>]`.
    ///
    /// `language` accepts BCP-47-ish codes (`"en"`, `"zh"`, `"yue"`, ...). Pass
    /// `None` to skip the language token (english-only) or to let the decoder
    /// auto-detect via a separate preliminary decode.
    pub fn sot_prompt(&self, language: Option<&str>, translate: bool) -> Vec<u32> {
        let mut out = vec![self.variant.sot_token_id()];
        if let Some(code) = language {
            if let Some(id) = self.variant.language_token_id(code) {
                out.push(id);
            }
        }
        if translate {
            if let Some(id) = self.variant.translate_token_id() {
                out.push(id);
            }
        } else if let Some(id) = self.variant.transcribe_token_id() {
            out.push(id);
        }
        out.push(self.variant.notimestamps_token_id());
        out
    }
}

// ── internals ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // fields kept for config completeness / future use
struct HfWhisperConfig {
    #[serde(default)]
    _name_or_path: Option<String>,
    #[serde(default)]
    model_type: Option<String>,

    // Dimensions
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    #[serde(default)]
    decoder_layers: Option<usize>,
    #[serde(default)]
    decoder_attention_heads: Option<usize>,
    #[serde(default)]
    encoder_ffn_dim: Option<usize>,
    #[serde(default)]
    decoder_ffn_dim: Option<usize>,

    // Positional / vocab
    #[serde(default = "default_max_source_positions")]
    max_source_positions: usize,
    #[serde(default = "default_max_target_positions")]
    max_target_positions: usize,
    #[serde(default = "default_num_mel_bins_opt")]
    num_mel_bins: Option<usize>,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

fn default_max_source_positions() -> usize {
    1500
}
fn default_max_target_positions() -> usize {
    448
}
fn default_num_mel_bins_opt() -> Option<usize> {
    Some(80)
}
fn default_vocab_size() -> usize {
    51865
}

impl HfWhisperConfig {
    fn to_audio_config(&self) -> AudioConfig {
        AudioConfig {
            encoder_type: "whisper".to_string(),
            hidden_size: self.d_model,
            num_layers: self.encoder_layers,
            num_heads: self.encoder_attention_heads,
            num_mel_bins: self.num_mel_bins.unwrap_or(80),
            max_audio_len: self.max_source_positions * 2, // encoder conv downsamples 2x
            projector_type: "linear".to_string(),
            vocab_size: self.vocab_size,
            decoder_layers: self.decoder_layers,
            max_target_positions: self.max_target_positions,
            intermediate_size: self.decoder_ffn_dim.or(self.encoder_ffn_dim),
        }
    }
}

fn detect_variant(cfg: &HfWhisperConfig) -> WhisperVariant {
    // `_name_or_path` is the most reliable hint when present.
    if let Some(name) = &cfg._name_or_path {
        let lower = name.to_ascii_lowercase();
        if lower.contains(".en") || lower.contains("-en-") || lower.ends_with("en") {
            return WhisperVariant::EnglishOnly;
        }
        if lower.contains("v3") {
            return WhisperVariant::V3Multilingual;
        }
        if lower.contains("v2") {
            return WhisperVariant::V2Multilingual;
        }
        if lower.contains("v1") {
            return WhisperVariant::V1Multilingual;
        }
    }

    // Fall back to vocab_size — unambiguous for english-only (51864) and v3 (51866).
    match cfg.vocab_size {
        51864 => WhisperVariant::EnglishOnly,
        51866 => WhisperVariant::V3Multilingual,
        _ => WhisperVariant::V2Multilingual,
    }
}

fn find_safetensors(dir: &Path) -> Result<PathBuf> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    let entries = std::fs::read_dir(dir).map_err(|e| Error::ModelError {
        reason: format!("reading {}: {e}", dir.display()),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::ModelError {
            reason: format!("reading dir entry: {e}"),
        })?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return Ok(entry.path());
        }
    }
    Err(Error::ModelError {
        reason: format!(
            "no safetensors file found in {} (expected model.safetensors)",
            dir.display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_v3_from_name() {
        let cfg = HfWhisperConfig {
            _name_or_path: Some("openai/whisper-large-v3".into()),
            model_type: None,
            d_model: 1280,
            encoder_layers: 32,
            encoder_attention_heads: 20,
            decoder_layers: Some(32),
            decoder_attention_heads: Some(20),
            encoder_ffn_dim: Some(5120),
            decoder_ffn_dim: Some(5120),
            max_source_positions: 1500,
            max_target_positions: 448,
            num_mel_bins: Some(128),
            vocab_size: 51866,
        };
        assert_eq!(detect_variant(&cfg), WhisperVariant::V3Multilingual);
    }

    #[test]
    fn detect_english_only_from_vocab() {
        let cfg = HfWhisperConfig {
            _name_or_path: None,
            model_type: None,
            d_model: 512,
            encoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_layers: Some(6),
            decoder_attention_heads: Some(8),
            encoder_ffn_dim: Some(2048),
            decoder_ffn_dim: Some(2048),
            max_source_positions: 1500,
            max_target_positions: 448,
            num_mel_bins: Some(80),
            vocab_size: 51864,
        };
        assert_eq!(detect_variant(&cfg), WhisperVariant::EnglishOnly);
    }

    #[test]
    fn detect_v2_fallback() {
        let cfg = HfWhisperConfig {
            _name_or_path: Some("openai/whisper-base".into()),
            model_type: None,
            d_model: 512,
            encoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_layers: Some(6),
            decoder_attention_heads: Some(8),
            encoder_ffn_dim: Some(2048),
            decoder_ffn_dim: Some(2048),
            max_source_positions: 1500,
            max_target_positions: 448,
            num_mel_bins: Some(80),
            vocab_size: 51865,
        };
        assert_eq!(detect_variant(&cfg), WhisperVariant::V2Multilingual);
    }
}
