//! Load a standalone Whisper checkpoint (HF layout) into a [`WhisperBundle`].
//!
//! Expected directory layout (what `openai/whisper-*` ships on HuggingFace):
//! ```text
//! <dir>/
//!   config.json                    # architecture + vocab_size + hidden sizes
//!   tokenizer.json                 # byte-level BPE vocab + merges
//!   model.safetensors              # weights (all prefixes under `model.*`)
//!                                  # — or, for a sharded checkpoint,
//!                                  # model.safetensors.index.json plus its shards
//!   generation_config.json         # optional — decoding constraints, parsed
//!                                  # into [`WhisperGenerationConfig`]
//!   preprocessor_config.json       # optional — mel params (num_mel_bins, etc.)
//! ```
//!
//! The weights, `config.json` and variant detection are boostr's:
//! [`WhisperModel::from_checkpoint`] returns a [`WhisperCheckpoint`], and this
//! module adds the tokenizer and `generation_config.json` on top.

use std::path::Path;

use serde::Deserialize;
use splintr::{AnyTokenizer, PretrainedVocab, WhisperVariant, from_json_path, from_vocab};

use crate::error::{Error, Result};
use boostr::model::audio::whisper_loader::WhisperCheckpoint;
use boostr::model::audio::whisper_model::{GenerateOptions, WhisperModel};
use boostr::model::config::AudioConfig;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

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

impl<R: Runtime<DType = DType>> WhisperBundle<R> {
    /// Load a bundle from an HF-style Whisper checkpoint directory, keeping
    /// each weight in the dtype the file stores it in.
    ///
    /// `openai/whisper-large-v3` ships **fp16** weights, so this yields an fp16
    /// model. numr's ops require the input and the weight to share a dtype, so
    /// feeding it an f32 mel fails with `conv1d requires same dtype`. Use
    /// [`Self::from_dir_with_dtype`] with `DType::F32` to load such a
    /// checkpoint for f32 compute — at 2x the memory, which is why it is not
    /// the default here.
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
        let dir = dir.as_ref();
        let checkpoint = WhisperModel::<R>::from_checkpoint(dir, device)?;
        Self::wrap(dir, checkpoint)
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
        let dir = dir.as_ref();
        let checkpoint = WhisperModel::<R>::from_checkpoint_with_dtype(dir, device, client, dtype)?;
        Self::wrap(dir, checkpoint)
    }

    /// Add the tokenizer and generation config to an already-loaded checkpoint.
    fn wrap(dir: &Path, checkpoint: WhisperCheckpoint<R>) -> Result<Self> {
        let WhisperCheckpoint {
            model,
            config,
            variant,
            num_mel_bins,
        } = checkpoint;

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

        Ok(Self {
            model,
            tokenizer,
            variant,
            config,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
