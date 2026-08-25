//! Load a standalone Whisper checkpoint (HF layout) into a [`WhisperBundle`].
//!
//! Expected directory layout (what `openai/whisper-*` ships on HuggingFace):
//! ```text
//! <dir>/
//!   config.json                    # architecture + vocab_size + hidden sizes
//!   tokenizer.json                 # byte-level BPE vocab + merges
//!   model.safetensors              # weights (all prefixes under `model.*`)
//!   generation_config.json         # optional — decoding constraints, parsed
//!                                  # into [`WhisperGenerationConfig`]
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
use splintr::{AnyTokenizer, PretrainedVocab, WhisperVariant, from_json_path, from_vocab};

use crate::error::{Error, Result};
use crate::model::audio::whisper_model::{GenerateOptions, WhisperModel};
use crate::model::config::AudioConfig;
use crate::nn::weight::Weight;
use crate::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Everything a caller needs to run Whisper transcription: the model, the
/// tokenizer, and the variant metadata that tells callers which language tokens
/// / control tokens to emit as the SOT prompt.
pub struct WhisperBundle<R: Runtime> {
    pub model: WhisperModel<R>,
    pub tokenizer: AnyTokenizer,
    pub variant: WhisperVariant,
    pub config: AudioConfig,
    /// Number of mel filterbank bins (80 for tiny/base/small/medium/large, 128 for v3).
    pub num_mel_bins: usize,
    /// Decoding constraints read from the checkpoint's `generation_config.json`,
    /// or the variant defaults when the checkpoint ships no such file.
    pub generation: WhisperGenerationConfig,
}

/// Decoding constraints from a Whisper checkpoint's `generation_config.json`.
///
/// The file is optional. When it is absent the loader falls back to
/// [`WhisperGenerationConfig::for_variant`], which suppresses **nothing** — a
/// decode run under empty suppression drifts from the reference implementation
/// as soon as the model prefers a token OpenAI's config forbids (punctuation,
/// markup, sound-effect tags). An empty `suppress_tokens` is therefore the
/// signal that this checkpoint carried no config, not that it suppressed
/// nothing: every `openai/whisper-*` release ships a non-empty list.
///
/// The deprecated `forced_decoder_ids` field is read from no checkpoint and
/// deliberately ignored: it overrides the language/task prefix that
/// [`WhisperBundle::sot_prompt`] already builds correctly, and in HuggingFace
/// itself it truncates generation to a single token.
#[derive(Debug, Clone)]
pub struct WhisperGenerationConfig {
    /// Token IDs forbidden at every generated position.
    pub suppress_tokens: Vec<u32>,
    /// Token IDs forbidden at the first generated position only.
    pub begin_suppress_tokens: Vec<u32>,
    /// Token IDs that end generation. `eos_token_id` in the file, which may be
    /// a single integer or a list.
    pub eos_token_ids: Vec<u32>,
    /// Maximum length of the **full** decoder sequence, prefix included (448 for
    /// every Whisper release).
    pub max_length: usize,
}

/// Whisper's decoder position budget — `max_target_positions` in config.json and
/// `max_length` in generation_config.json alike.
const WHISPER_MAX_LENGTH: usize = 448;

impl WhisperGenerationConfig {
    /// Defaults for a checkpoint that ships no `generation_config.json`: no
    /// suppression, the variant's own `<|endoftext|>`, and Whisper's 448-token
    /// sequence budget.
    pub fn for_variant(variant: WhisperVariant) -> Self {
        Self {
            suppress_tokens: Vec::new(),
            begin_suppress_tokens: Vec::new(),
            eos_token_ids: vec![variant.eos_token_id()],
            max_length: WHISPER_MAX_LENGTH,
        }
    }
}

/// A weight-casting callback paired with the dtype it casts to, threaded
/// through [`WhisperBundle::load`]. Named to keep the `load` signature under
/// clippy's `type_complexity` threshold.
type CastToDtype<'a, R> = (&'a dyn Fn(&Tensor<R>) -> Result<Tensor<R>>, DType);

impl<R: Runtime<DType = DType>> WhisperBundle<R> {
    /// Load a bundle from an HF-style Whisper checkpoint directory, keeping
    /// each weight in the dtype the file stores it in.
    ///
    /// `openai/whisper-large-v3` ships **fp16** weights, so this yields an fp16
    /// model. numr's ops require the input and the weight to share a dtype, so
    /// feeding it an f32 mel fails with `conv1d requires same dtype`. Use
    /// [`Self::from_dir_with_dtype`] with `Some(DType::F32)` to load such a
    /// checkpoint for f32 compute — at 2x the memory, which is why it is not
    /// the default here.
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
        Self::load(dir.as_ref(), device, None)
    }

    /// Load a bundle, casting every weight to `dtype` on the way in.
    ///
    /// This is what lets an fp16 checkpoint such as `whisper-large-v3` run
    /// against an f32 mel. A client is required because numr builds no client
    /// from a device alone, and the caller already holds one to run `encode`.
    pub fn from_dir_with_dtype<P: AsRef<Path>, C: TypeConversionOps<R>>(
        dir: P,
        device: &R::Device,
        client: &C,
        dtype: DType,
    ) -> Result<Self> {
        let cast = |t: &Tensor<R>| client.cast(t, dtype).map_err(Error::Numr);
        Self::load(dir.as_ref(), device, Some((&cast, dtype)))
    }

    fn load(dir: &Path, device: &R::Device, cast_to: Option<CastToDtype<'_, R>>) -> Result<Self> {
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

        // Multilingual v1/v2/v3 load zero-config from splintr's bundled vocab;
        // anything else loads its own `tokenizer.json`. Both paths yield an
        // `AnyTokenizer`, so no wrapper is needed to unify them.
        let tokenizer = match whisper_pretrained_vocab(variant) {
            Some(vocab) => from_vocab(vocab).map_err(|e| Error::ModelError {
                reason: format!("loading bundled {variant:?} whisper tokenizer: {e}"),
            })?,
            None => {
                let tok_path = dir.join("tokenizer.json");
                from_json_path(&tok_path).map_err(|e| Error::ModelError {
                    reason: format!("loading {}: {e}", tok_path.display()),
                })?
            }
        };

        let generation = load_generation_config(dir, variant)?;

        let weights_path = find_safetensors(dir)?;
        let mut varmap = VarMap::<R>::from_safetensors(&weights_path, device)?;
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
        let model = WhisperModel::from_varbuilder(&mut vb, &audio_config)?;

        Ok(Self {
            model,
            tokenizer,
            variant,
            config: audio_config,
            num_mel_bins,
            generation,
        })
    }

    /// [`GenerateOptions`] prefilled from the checkpoint's own generation config.
    ///
    /// Callers that decode without these constraints get a transcript that
    /// diverges from the reference implementation, so this is the default way to
    /// drive [`WhisperModel::generate`]; override individual fields afterwards
    /// when a run needs a tighter budget or extra stop tokens.
    pub fn generate_options(&self) -> GenerateOptions {
        let mut eos_token_ids = self.generation.eos_token_ids.clone();
        let variant_eos = self.variant.eos_token_id();
        if eos_token_ids.is_empty() {
            eos_token_ids.push(variant_eos);
        }

        // `max_length` counts the whole decoder sequence, `max_new_tokens` does
        // not count the prefix — subtract the longest prompt this variant builds
        // (sot, language, task, notimestamps).
        let prefix_len = self.sot_prompt(Some("en"), false).len();

        GenerateOptions {
            max_new_tokens: self.generation.max_length.saturating_sub(prefix_len),
            eos_token_ids,
            suppress_tokens: self.generation.suppress_tokens.clone(),
            begin_suppress_tokens: self.generation.begin_suppress_tokens.clone(),
        }
    }

    /// Build the "start-of-transcript" prompt for greedy decoding.
    ///
    /// Layout (multilingual): `[<|sot|>, <|lang|>, <|task|>, <|notimestamps|>]`.
    /// Layout (english-only):  `[<|sot|>, <|transcribe|>, <|notimestamps|>]` —
    /// english-only checkpoints carry `<|translate|>`/`<|transcribe|>` in their
    /// special table too, so the task token is always emitted; only the language
    /// token is skipped when `language` is `None`.
    ///
    /// `language` accepts BCP-47-ish codes (`"en"`, `"zh"`, `"yue"`, ...). Pass
    /// `None` to skip the language token (english-only) or to let the decoder
    /// auto-detect via a separate preliminary decode.
    pub fn sot_prompt(&self, language: Option<&str>, translate: bool) -> Vec<u32> {
        let mut out = vec![self.variant.sot_token_id()];
        if let Some(code) = language
            && let Some(id) = self.variant.language_token_id(code)
        {
            out.push(id);
        }
        if translate {
            out.push(self.variant.translate_token_id());
        } else {
            out.push(self.variant.transcribe_token_id());
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

/// Raw `generation_config.json`. Every field is optional — checkpoints ship
/// wildly different subsets, and a missing field falls back to the variant
/// default rather than failing the load.
#[derive(Debug, Deserialize)]
struct HfGenerationConfig {
    #[serde(default)]
    suppress_tokens: Option<Vec<u32>>,
    #[serde(default)]
    begin_suppress_tokens: Option<Vec<u32>>,
    #[serde(default)]
    eos_token_id: Option<TokenIdField>,
    #[serde(default)]
    max_length: Option<usize>,
}

/// `eos_token_id` is a bare integer on every `openai/whisper-*` release, but the
/// HF schema also permits a list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIdField {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIdField {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

/// Read `generation_config.json` if the checkpoint ships one.
///
/// A missing file is not an error — it yields
/// [`WhisperGenerationConfig::for_variant`]. A file that exists but cannot be
/// read or parsed IS an error: silently decoding without the constraints it
/// carries produces a subtly wrong transcript.
fn load_generation_config(dir: &Path, variant: WhisperVariant) -> Result<WhisperGenerationConfig> {
    let path = dir.join("generation_config.json");
    if !path.exists() {
        return Ok(WhisperGenerationConfig::for_variant(variant));
    }

    let bytes = std::fs::read(&path).map_err(|e| Error::ModelError {
        reason: format!("reading {}: {e}", path.display()),
    })?;
    let raw: HfGenerationConfig =
        serde_json::from_slice(&bytes).map_err(|e| Error::ModelError {
            reason: format!("parsing {}: {e}", path.display()),
        })?;

    let eos_token_ids = raw
        .eos_token_id
        .map(TokenIdField::into_vec)
        .unwrap_or_else(|| vec![variant.eos_token_id()]);

    Ok(WhisperGenerationConfig {
        suppress_tokens: raw.suppress_tokens.unwrap_or_default(),
        begin_suppress_tokens: raw.begin_suppress_tokens.unwrap_or_default(),
        eos_token_ids,
        max_length: raw.max_length.unwrap_or(WHISPER_MAX_LENGTH),
    })
}

/// Map a [`WhisperVariant`] to splintr's bundled pretrained vocab, if one
/// exists. Multilingual v1/v2/v3 are bundled; English-only (and any future
/// variant without bundled support) returns `None` and loads from
/// `tokenizer.json` instead.
fn whisper_pretrained_vocab(variant: WhisperVariant) -> Option<PretrainedVocab> {
    match variant {
        WhisperVariant::V1Multilingual => Some(PretrainedVocab::WhisperV1),
        WhisperVariant::V2Multilingual => Some(PretrainedVocab::WhisperV2),
        WhisperVariant::V3Multilingual => Some(PretrainedVocab::WhisperV3),
        WhisperVariant::EnglishOnly => None,
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
    fn generation_config_defaults_when_file_missing() {
        let dir = Path::new("/nonexistent-whisper-checkpoint");
        let defaults = load_generation_config(dir, WhisperVariant::V2Multilingual)
            .expect("missing generation_config.json must not be an error");
        assert!(defaults.suppress_tokens.is_empty());
        assert!(defaults.begin_suppress_tokens.is_empty());
        assert_eq!(defaults.eos_token_ids, vec![50257]);
        assert_eq!(defaults.max_length, 448);
    }

    #[test]
    fn generation_config_parses_scalar_and_list_eos() {
        let scalar: HfGenerationConfig = serde_json::from_str(
            r#"{"suppress_tokens":[1,2],"begin_suppress_tokens":[220,50257],
                "eos_token_id":50257,"max_length":448,
                "forced_decoder_ids":[[1,null],[2,50359]]}"#,
        )
        .expect("scalar eos_token_id");
        assert_eq!(scalar.suppress_tokens.as_deref(), Some(&[1u32, 2][..]));
        assert_eq!(
            scalar.eos_token_id.map(TokenIdField::into_vec),
            Some(vec![50257])
        );
        assert_eq!(scalar.max_length, Some(448));

        let list: HfGenerationConfig =
            serde_json::from_str(r#"{"eos_token_id":[50257,50362]}"#).expect("list eos_token_id");
        assert_eq!(
            list.eos_token_id.map(TokenIdField::into_vec),
            Some(vec![50257, 50362])
        );
        assert_eq!(list.max_length, None);
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
