//! VoxCPM2 as a [`TtsEngine`]: the clone pipeline behind one `synthesize`.
//!
//! A voice is a reference recording. The engine encodes every recording in a
//! voices directory once at load, so a request pays only its own prefill,
//! generation and decode. Text reaches the model as raw tokens, so a Malay
//! and English code-switched sentence needs no language switch.
//!
//! One render runs at a time: `synthesize` holds a lock for its duration.
//! The model's KV caches and generation state are per call, so the lock
//! serialises device work, not correctness.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateState, LoraAdapterReport, VoxCpm2Model,
};
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;
use boostr::model::audio::voxcpm::{VoxCpm2Weights, VoxCpmClient};
use boostr::quant::traits::DequantOps;

use crate::decode::{decode_audio, extension_hint};
use crate::error::{Error, Result};
use crate::g2p::Lang;
use crate::resample::to_mono_at_rate;
use crate::tts::{TtsEngine, Voice};
use crate::voxcpm::tokenizer::{load_tokenizer, normalize_whitespace, tokenize};

/// Sample rate the reference encoder expects.
const REF_RATE: u32 = 16_000;

/// Hard cap on generated patches for one request, whatever the text length.
const MAX_LEN_CAP: usize = 4096;

/// Containers `decode_audio` probes; a file with any other extension in the
/// voices directory is not a voice.
const VOICE_EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "ogg"];

/// Reserved voice id selecting zero-shot rendering: no reference recording,
/// no reference encode, `ref_feat = None` straight through to `prefill`.
///
/// A fixed literal, not an `Option<String>` layered on top of `render`: an
/// omitted voice must be an explicit, listable choice — present in
/// [`VoxCpm2Engine::voices`] like any other id — never a silent fallback a
/// caller has to infer. A literal also can't collide with a voice file by
/// accident: [`VoxCpm2Engine::load`] refuses any file stemmed `zero-shot`.
pub const ZERO_SHOT_VOICE_ID: &str = "zero-shot";

/// Generation settings applied to every request. Defaults are the clone
/// pipeline's verified values.
#[derive(Debug, Clone)]
pub struct VoxCpm2SynthOptions {
    /// Flow-matching solver steps per patch.
    pub n_timesteps: usize,
    /// Classifier-free guidance scale.
    pub cfg_value: f32,
    /// Patches during which the stop token is ignored.
    pub min_len: usize,
    /// Base seed; every request draws from it, so equal requests render
    /// equal audio on one backend.
    pub seed: u64,
}

impl Default for VoxCpm2SynthOptions {
    fn default() -> Self {
        Self {
            n_timesteps: 10,
            cfg_value: 2.0,
            min_len: 2,
            seed: 0,
        }
    }
}

/// Everything [`VoxCpm2Engine::load`] needs beyond the checkpoint location:
/// the dtype to cast to, per-request generation defaults, and an optional
/// LoRA adapter to fold into the weights. Bundled into one struct because
/// `load` already takes five positional arguments (weights, audiovae,
/// voices_dir, device, client) — three more flat parameters would trip
/// clippy's too-many-arguments limit.
#[derive(Debug, Clone, Default)]
pub struct VoxCpm2LoadOptions {
    /// Casts every transformer-stack tensor; `None` keeps the checkpoint's
    /// own dtype (BF16).
    pub dtype: Option<DType>,
    /// Per-request generation settings — see [`VoxCpm2SynthOptions`].
    pub synth: VoxCpm2SynthOptions,
    /// A LoRA adapter safetensors file, folded into the model's weights
    /// once, at load time. `None` loads the base model, unchanged.
    ///
    /// The adapter is folded in ONCE: an engine serves ONE adapted model for
    /// its lifetime. Per-request adapter switching is not offered — serving
    /// several adapters means loading several engines (a `--tts-model
    /// NAME=DIR` bundle per adapter, in blazr's terms), one adapter each.
    pub adapter: Option<PathBuf>,
}

/// A reference voice, encoded once.
struct EncodedVoice<R: Runtime> {
    /// `[T_ref, feat_dim]` reference patches from `encode_reference`.
    ref_feat: Tensor<R>,
}

/// VoxCPM2 voice-cloning engine on runtime `R`.
pub struct VoxCpm2Engine<R: Runtime<DType = DType>> {
    model: VoxCpm2Model<R>,
    client: Arc<R::Client>,
    tokenizer: AnyTokenizer,
    voices: BTreeMap<String, EncodedVoice<R>>,
    options: VoxCpm2SynthOptions,
    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`] was `Some`. Kept so a caller (e.g.
    /// blazr) can log rank/alpha/targets/counts via [`Self::adapter`]
    /// without reaching into the model.
    adapter: Option<LoraAdapterReport>,
    /// Serialises renders. Holds no data; see the module docs.
    render: Mutex<()>,
}

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType>,
    R::Client: VoxCpmClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + RandomOps<R>
        + DequantOps<R>
        + 'static,
{
    /// Load the model, the tokenizer and every voice under `voices_dir`,
    /// optionally folding a LoRA adapter into the weights first.
    ///
    /// A voice is any wav, flac, mp3 or ogg file in `voices_dir`; its id is
    /// the file stem. `voices_dir` is optional: `None`, a missing directory,
    /// or an empty directory all load fine with zero encoded voices —
    /// `render` still serves [`ZERO_SHOT_VOICE_ID`], which needs none. A
    /// voice file stemmed `zero-shot` is refused at load: it would silently
    /// shadow the reserved id.
    ///
    /// `options.adapter` is applied to the model BEFORE it is wrapped as an
    /// engine — see [`VoxCpm2LoadOptions::adapter`] for the one-adapter-per-
    /// bundle model. `None` loads the base model, unchanged. Either way, the
    /// applied adapter's report (if any) is readable back via
    /// [`Self::adapter`].
    pub fn load(
        weights: &VoxCpm2Weights,
        audiovae: &Path,
        voices_dir: Option<&Path>,
        device: &R::Device,
        client: Arc<R::Client>,
        options: VoxCpm2LoadOptions,
    ) -> Result<Self> {
        let VoxCpm2LoadOptions {
            dtype,
            synth,
            adapter,
        } = options;

        let mut model = match weights {
            VoxCpm2Weights::Checkpoint(dir) => {
                VoxCpm2Model::<R>::from_checkpoint(dir, audiovae, device, dtype)?
            }
            VoxCpm2Weights::Gguf { path, config } => {
                VoxCpm2Model::<R>::from_gguf(path, Some(config.as_path()), audiovae, device, dtype)?
            }
            VoxCpm2Weights::Tcf { path, config } => {
                VoxCpm2Model::<R>::from_tcf(path, config, audiovae, device, dtype)?
            }
        };
        let adapter_report = adapter
            .as_deref()
            .map(|path| model.load_lora_adapter(path, device))
            .transpose()?;
        let tokenizer = load_tokenizer(weights.tokenizer_path()?)?;

        let mut voices = BTreeMap::new();
        if let Some(dir) = voices_dir {
            for (id, path) in list_voice_files(dir)? {
                reject_reserved_voice_id(&id, &path)?;
                let ref_wav = load_reference_16k(&path)?;
                let ref_feat = model.encode_reference(client.as_ref(), &ref_wav)?;
                voices.insert(id, EncodedVoice { ref_feat });
            }
        }

        Ok(Self {
            model,
            client,
            tokenizer,
            voices,
            options: synth,
            adapter: adapter_report,
            render: Mutex::new(()),
        })
    }

    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`] was `Some` — rank, alpha, targets and
    /// tensor counts, for a caller to log without reaching into the model.
    /// `None` when the engine is serving the base (unadapted) model.
    pub fn adapter(&self) -> Option<&LoraAdapterReport> {
        self.adapter.as_ref()
    }

    fn render(&self, text: &str, voice_id: &str) -> Result<Vec<f32>> {
        let ref_feat: Option<&Tensor<R>> = if voice_id == ZERO_SHOT_VOICE_ID {
            None
        } else {
            let voice = self
                .voices
                .get(voice_id)
                .ok_or_else(|| Error::InvalidArgument {
                    arg: "voice",
                    reason: format!(
                        "unknown voice {voice_id:?}; available voices: [{}], or {:?} for \
                         zero-shot rendering",
                        self.voices.keys().cloned().collect::<Vec<_>>().join(", "),
                        ZERO_SHOT_VOICE_ID,
                    ),
                })?;
            Some(&voice.ref_feat)
        };
        let normalized = normalize_whitespace(text);
        if normalized.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let mut text_token_ids = tokenize(&self.tokenizer, &normalized);
        let text_len = text_token_ids.len();
        text_token_ids.push(AUDIO_START_ID);

        // The clone pipeline's budget: six patches per text token plus ten,
        // capped.
        let max_len = (text_len * 6 + 10).min(MAX_LEN_CAP);
        let ref_len = ref_feat.map(|f| f.shape()[0]);
        let max_length = seq_len_for(ref_len, text_token_ids.len()) + max_len;

        let mut options = GenerateOptions::new(max_len, self.options.seed);
        options.cfm.n_timesteps = self.options.n_timesteps;
        options.cfm.cfg_value = self.options.cfg_value;
        options.min_len = self.options.min_len;

        let _render = self.render.lock().map_err(|_| Error::ModelError {
            reason: "VoxCPM2 render lock poisoned by an earlier panic".into(),
        })?;
        let client = self.client.as_ref();
        let prefill = self
            .model
            .prefill(client, ref_feat, &text_token_ids, max_length)?;
        let mut state = GenerateState::start(prefill, self.model.config)?;
        self.model
            .patch_generator()
            .generate(client, &mut state, &options)?;
        let decoded = self.model.decode_patches(client, &state.patches)?;
        Ok(decoded.contiguous()?.to_vec())
    }
}

impl<R> TtsEngine for VoxCpm2Engine<R>
where
    R: Runtime<DType = DType>,
    R::Client: VoxCpmClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + RandomOps<R>
        + DequantOps<R>
        + Send
        + Sync
        + 'static,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        // The model has no rate control; a silently ignored `speed` would
        // return audio the caller did not ask for.
        if speed != 1.0 {
            return Err(Error::InvalidArgument {
                arg: "speed",
                reason: format!("VoxCPM2 renders at its natural pace only; got {speed}"),
            });
        }
        self.render(text, voice)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    /// [`ZERO_SHOT_VOICE_ID`] first, then one entry per reference recording.
    /// VoxCPM2 takes raw text in any language it was trained on, so every
    /// voice carries the product language tag rather than a per-voice one.
    fn voices(&self) -> Vec<Voice> {
        std::iter::once(Voice::new(ZERO_SHOT_VOICE_ID, Lang::Ms, ZERO_SHOT_VOICE_ID))
            .chain(
                self.voices
                    .keys()
                    .map(|id| Voice::new(id.clone(), Lang::Ms, id.clone())),
            )
            .collect()
    }
}

/// Sequence length behind `prefill`'s cache sizing. Mirrors `clone.rs`'s
/// zero-shot/reference branch exactly (see its `SequenceLayout` docs): a
/// reference prefix adds `t_ref + 2` positions before the text; zero-shot
/// carries no reference prefix at all, not a reference of zero patches.
/// `text_len` is the token count AFTER `AUDIO_START_ID` is appended.
fn seq_len_for(ref_len: Option<usize>, text_len: usize) -> usize {
    match ref_len {
        Some(t_ref) => t_ref + 2 + text_len,
        None => text_len,
    }
}

/// Refuse a voice file whose stem is the reserved zero-shot id. Without this
/// check, a file named `zero-shot.wav` would silently shadow
/// [`ZERO_SHOT_VOICE_ID`] and `render` could never reach the no-reference
/// path for that id again.
fn reject_reserved_voice_id(id: &str, path: &Path) -> Result<()> {
    if id == ZERO_SHOT_VOICE_ID {
        return Err(Error::ModelError {
            reason: format!(
                "voice file {} has stem {id:?}, which is reserved for zero-shot rendering \
                 ({ZERO_SHOT_VOICE_ID:?}); rename the file",
                path.display()
            ),
        });
    }
    Ok(())
}

/// `(voice id, path)` for every regular file in `dir` with a recognised
/// audio extension, sorted by id. A missing directory is not an error: it
/// loads as zero voices, same as an empty one — a caller relying on
/// zero-shot rendering need not create `voices/` at all.
fn list_voice_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(Error::ModelError {
                reason: format!("reading voices directory {}: {e}", dir.display()),
            });
        }
    };
    let mut voices = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| Error::ModelError {
                reason: format!("reading voices directory {}: {e}", dir.display()),
            })?
            .path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let is_audio = extension_hint(name)
            .is_some_and(|ext| VOICE_EXTENSIONS.iter().any(|a| ext.eq_ignore_ascii_case(a)));
        if !is_audio {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        voices.push((stem.to_string(), path.clone()));
    }
    voices.sort();
    Ok(voices)
}

/// Decode one reference recording to mono at the encoder's rate.
fn load_reference_16k(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
        reason: format!("reading voice {}: {e}", path.display()),
    })?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    to_mono_at_rate(&data, REF_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_listing_takes_audio_files_only_sorted_by_id() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("zara.wav"), b"").unwrap();
        std::fs::write(dir.join("amir.flac"), b"").unwrap();
        std::fs::write(dir.join("notes.txt"), b"").unwrap();
        std::fs::create_dir_all(dir.join("sub.wav")).unwrap();
        let listed = list_voice_files(&dir).unwrap();
        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["amir", "zara"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_voice_files_on_missing_dir_is_empty_ok() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices_missing");
        let _ = std::fs::remove_dir_all(&dir);
        let listed = list_voice_files(&dir).unwrap();
        assert!(listed.is_empty());
    }

    #[test]
    fn list_voice_files_on_empty_dir_is_empty_ok() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let listed = list_voice_files(&dir).unwrap();
        assert!(listed.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seq_len_for_with_reference_adds_t_ref_plus_two() {
        assert_eq!(seq_len_for(Some(40), 12), 40 + 2 + 12);
    }

    #[test]
    fn seq_len_for_zero_shot_is_text_len_only() {
        assert_eq!(seq_len_for(None, 12), 12);
    }

    #[test]
    fn reserved_voice_id_is_refused_at_load() {
        let path = PathBuf::from("/voices/zero-shot.wav");
        let err = reject_reserved_voice_id(ZERO_SHOT_VOICE_ID, &path).unwrap_err();
        assert!(err.to_string().contains(ZERO_SHOT_VOICE_ID));
        assert!(err.to_string().contains("zero-shot.wav"));
    }

    #[test]
    fn non_reserved_voice_id_is_accepted() {
        let path = PathBuf::from("/voices/zara.wav");
        assert!(reject_reserved_voice_id("zara", &path).is_ok());
    }
}
