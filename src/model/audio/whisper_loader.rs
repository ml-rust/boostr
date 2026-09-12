//! Load the weights of a standalone Whisper checkpoint (HF layout).
//!
//! Expected directory layout (what `openai/whisper-*` ships on HuggingFace):
//! ```text
//! <dir>/
//!   config.json                    # architecture + vocab_size + hidden sizes
//!   model.safetensors              # weights (all prefixes under `model.*`)
//!                                  # — or, for a sharded checkpoint,
//!                                  # model.safetensors.index.json plus its shards
//! ```
//!
//! The tokenizer, `generation_config.json` and the transcription front end are
//! not read here. They belong to `boostr_audio::whisper::WhisperBundle`, which
//! wraps the [`WhisperCheckpoint`] this module produces.
//!
//! Auto-detects the [`WhisperVariant`] from config.json:
//! - `"_name_or_path"` containing `"v3"` / `"large-v3"` → V3
//! - `"_name_or_path"` containing `".en"` or `num_languages == 0` → EnglishOnly
//! - `vocab_size == 51866` → V3
//! - `vocab_size == 51864` → EnglishOnly
//! - otherwise → V2 (the common multilingual case)

use std::path::{Path, PathBuf};

use serde::Deserialize;
use splintr::WhisperVariant;

use crate::error::{Error, Result};
use crate::model::audio::whisper_model::WhisperModel;
use crate::model::config::AudioConfig;
use crate::nn::weight::Weight;
use crate::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A Whisper checkpoint's weights plus the metadata read from its
/// `config.json`: everything a tokenizer-carrying bundle builds on.
pub struct WhisperCheckpoint<R: Runtime> {
    /// Encoder and decoder, built from the checkpoint's weights.
    pub model: WhisperModel<R>,
    /// The architecture as `config.json` describes it.
    pub config: AudioConfig,
    /// Which token table the checkpoint was trained with.
    pub variant: WhisperVariant,
    /// Number of mel filterbank bins (80 for tiny/base/small/medium/large, 128 for v3).
    pub num_mel_bins: usize,
}

/// A weight-casting callback paired with the dtype it casts to, threaded
/// through [`WhisperModel::load_checkpoint`]. Named to keep the signature
/// under clippy's `type_complexity` threshold.
type CastToDtype<'a, R> = (&'a dyn Fn(&Tensor<R>) -> Result<Tensor<R>>, DType);

impl<R: Runtime<DType = DType>> WhisperModel<R> {
    /// Load a checkpoint directory, keeping each weight in the dtype the file
    /// stores it in.
    ///
    /// `openai/whisper-large-v3` ships **fp16** weights, so this yields an fp16
    /// model. numr's ops require the input and the weight to share a dtype, so
    /// feeding it an f32 mel fails with `conv1d requires same dtype`. Use
    /// [`Self::from_checkpoint_with_dtype`] with `DType::F32` to load such a
    /// checkpoint for f32 compute — at 2x the memory, which is why it is not
    /// the default here.
    pub fn from_checkpoint<P: AsRef<Path>>(
        dir: P,
        device: &R::Device,
    ) -> Result<WhisperCheckpoint<R>> {
        Self::load_checkpoint(dir.as_ref(), device, None)
    }

    /// Load a checkpoint directory, casting every weight to `dtype` on the way in.
    ///
    /// This is what lets an fp16 checkpoint such as `whisper-large-v3` run
    /// against an f32 mel. A client is required because numr builds no client
    /// from a device alone, and the caller already holds one to run `encode`.
    pub fn from_checkpoint_with_dtype<P: AsRef<Path>, C: TypeConversionOps<R>>(
        dir: P,
        device: &R::Device,
        client: &C,
        dtype: DType,
    ) -> Result<WhisperCheckpoint<R>> {
        let cast = |t: &Tensor<R>| client.cast(t, dtype).map_err(Error::Numr);
        Self::load_checkpoint(dir.as_ref(), device, Some((&cast, dtype)))
    }

    fn load_checkpoint(
        dir: &Path,
        device: &R::Device,
        cast_to: Option<CastToDtype<'_, R>>,
    ) -> Result<WhisperCheckpoint<R>> {
        let cfg_path = dir.join("config.json");
        let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| Error::ModelError {
            reason: format!("reading {}: {e}", cfg_path.display()),
        })?;
        let hf: HfWhisperConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| Error::ModelError {
                reason: format!("parsing {}: {e}", cfg_path.display()),
            })?;

        let variant = detect_variant(&hf);
        let config = hf.to_audio_config();
        let num_mel_bins = hf.num_mel_bins.unwrap_or(80);

        // Both arms feed the same `varmap`, so the dtype cast below applies to a
        // sharded checkpoint exactly as it does to a single-file one.
        let mut varmap = match find_safetensors(dir)? {
            SafetensorsLayout::Single(path) => VarMap::<R>::from_safetensors(&path, device)?,
            // `from_safetensors_sharded` reads `model.safetensors.index.json` from the
            // directory itself and loads every shard it names.
            SafetensorsLayout::Sharded(index) => VarMap::<R>::from_safetensors_sharded(dir, device)
                .map_err(|e| Error::ModelError {
                    reason: format!("loading sharded checkpoint via {}: {e}", index.display()),
                })?,
        };
        if let Some((cast, target)) = cast_to {
            let names: Vec<String> = varmap.names().map(str::to_string).collect();
            for name in names {
                // Only plain tensors are castable; a safetensors load produces
                // nothing else, but skipping rather than erroring keeps this
                // correct if a quantized weight ever reaches here.
                let Ok(Weight::Standard(tensor)) = varmap.get(&name) else {
                    continue;
                };
                if tensor.dtype() == target {
                    continue;
                }
                let converted = cast(tensor)?;
                varmap.insert(name, converted);
            }
        }
        let mut vb = VarBuilder::new(&mut varmap, device);
        let model = WhisperModel::from_varbuilder(&mut vb, &config)?;

        Ok(WhisperCheckpoint {
            model,
            config,
            variant,
            num_mel_bins,
        })
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

/// Which safetensors layout a checkpoint directory ships.
#[derive(Debug, PartialEq, Eq)]
enum SafetensorsLayout {
    /// One `model.safetensors` holding every weight.
    Single(PathBuf),
    /// A `model.safetensors.index.json` naming the shard each weight lives in.
    ///
    /// Holds the index path; the loader passes the *directory* to
    /// `VarMap::from_safetensors_sharded`, which reads the index itself.
    Sharded(PathBuf),
}

/// Decide how to load a checkpoint directory's weights.
///
/// Selection only — no file is opened — so the choice is testable without real
/// weights.
///
/// `model.safetensors` wins when present: `whisper-large-v3` ships a single-file
/// copy, and that path is verified working.
///
/// This never falls back to an arbitrary `*.safetensors` entry. The old code took
/// the first one `read_dir` returned, which for a sharded checkpoint is one shard
/// out of many, in whatever order the filesystem happened to yield — the model then
/// loaded with most of its weights missing.
fn find_safetensors(dir: &Path) -> Result<SafetensorsLayout> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(SafetensorsLayout::Single(single));
    }

    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        return Ok(SafetensorsLayout::Sharded(index));
    }

    Err(Error::ModelError {
        reason: format!(
            "no safetensors weights found in {}: expected either model.safetensors \
             (single-file checkpoint) or model.safetensors.index.json (sharded checkpoint)",
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

    /// The single-file fast path must be unchanged: `whisper-large-v3` ships one
    /// `model.safetensors` and is verified working against it.
    #[test]
    fn find_safetensors_picks_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let single = dir.path().join("model.safetensors");
        std::fs::write(&single, b"").unwrap();

        assert_eq!(
            find_safetensors(dir.path()).unwrap(),
            SafetensorsLayout::Single(single)
        );
    }

    /// A sharded checkpoint must resolve to the index, never to one shard.
    ///
    /// The old code returned the first `*.safetensors` entry `read_dir` yielded,
    /// which here is an arbitrary one of the two shards — a model loaded from it is
    /// missing most of its weights.
    #[test]
    fn find_safetensors_picks_index_for_sharded_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index, b"{\"weight_map\":{}}").unwrap();
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"").unwrap();
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"").unwrap();

        assert_eq!(
            find_safetensors(dir.path()).unwrap(),
            SafetensorsLayout::Sharded(index)
        );
    }

    /// `model.safetensors` wins over an index when a checkpoint ships both.
    #[test]
    fn find_safetensors_prefers_single_file_over_index() {
        let dir = tempfile::tempdir().unwrap();
        let single = dir.path().join("model.safetensors");
        std::fs::write(&single, b"").unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            b"{\"weight_map\":{}}",
        )
        .unwrap();

        assert_eq!(
            find_safetensors(dir.path()).unwrap(),
            SafetensorsLayout::Single(single)
        );
    }

    /// A stray shard with neither a single file nor an index is not a checkpoint,
    /// and the error must name both files it looked for.
    #[test]
    fn find_safetensors_errors_naming_both_candidates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"").unwrap();

        let err = find_safetensors(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("model.safetensors"),
            "error must name model.safetensors: {msg}"
        );
        assert!(
            msg.contains("model.safetensors.index.json"),
            "error must name model.safetensors.index.json: {msg}"
        );
    }
}
