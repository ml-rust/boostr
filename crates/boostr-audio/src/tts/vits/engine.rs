//! `VitsEngine` — the ready-to-call VITS (MMS-TTS) synthesis bundle, and
//! [`load_vits_from_dir`], which loads one from an HF-style checkpoint
//! directory and wires it into a [`TtsBundle`].
//!
//! ```text
//! PATH/
//!   config.json              — VitsConfig (model_type: "vits")
//!   vocab.json                — character vocabulary
//!   tokenizer_config.json     — add_blank / normalize / language / pad / unk
//!   special_tokens_map.json   — optional
//!   model.safetensors         — weights (single file; every published
//!                               MMS-TTS checkpoint ships this way)
//! ```
//!
//! Every published MMS-TTS checkpoint is single-speaker and single-language,
//! so the voice catalog this engine reports is exactly one voice, named
//! after `tokenizer_config.json`'s `language` field (an ISO 639-3 code, e.g.
//! `"zlm"` for Standard Malay) — matching how a caller already picks an MMS
//! checkpoint per language, not per voice.

use std::path::Path;
use std::sync::Arc;

use boostr::model::audio::vits::{VitsModel, VitsSynthOptions};
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::lang::Lang;
use crate::tts::bundle::{TtsBundle, Voice};
use crate::tts::vits::tokenizer::VitsTokenizer;

/// A loaded VITS (MMS-TTS) checkpoint: model + tokenizer + the one voice it
/// serves.
pub struct VitsEngine {
    model: VitsModel<CpuRuntime>,
    tokenizer: VitsTokenizer,
    client: Arc<CpuClient>,
    device: Arc<CpuDevice>,
    voice: Voice,
    /// Overrides [`VitsSynthOptions::seed`] when set; `None` keeps the
    /// model's own default (`0`).
    seed: Option<u64>,
}

impl VitsEngine {
    /// Loads `config.json`, the tokenizer files, and `model.safetensors`
    /// from `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config = VitsConfig::from_json_file(&config_path)?;
        config.validate()?;

        let tokenizer = VitsTokenizer::from_dir(dir)?;

        let device = Arc::new(CpuDevice::new());
        let client = Arc::new(CpuClient::new((*device).clone()));

        let weights_path = dir.join("model.safetensors");
        let mut weights = VarMap::<CpuRuntime>::from_safetensors(&weights_path, &device)?;
        let mut vb = VarBuilder::new(&mut weights, &device);
        let model = VitsModel::<CpuRuntime>::from_varbuilder(client.as_ref(), &mut vb, &config)?;

        let language_code = read_language_code(dir)?;
        let lang = Lang::parse(&language_code).map_err(|e| Error::ModelError {
            reason: format!("{}/tokenizer_config.json: {e}", dir.display()),
        })?;
        let voice = Voice::new(language_code.clone(), lang, language_code);

        Ok(Self {
            model,
            tokenizer,
            client,
            device,
            voice,
            seed: None,
        })
    }

    /// Fixes the noise seed every [`VitsEngine::synthesize`] call draws,
    /// instead of the model's default (`0`). Useful for a reproducible test
    /// or demo render; a production server normally leaves this unset so
    /// repeated requests are not identical.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Render `text` in `voice` (must be this engine's one voice id) at
    /// `speed` (`1.0` is the checkpoint's natural pace; maps directly onto
    /// [`VitsSynthOptions::speaking_rate`] — VITS already inverts it
    /// internally when scaling durations).
    pub fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        if voice != self.voice.id {
            return Err(Error::InvalidArgument {
                arg: "voice",
                reason: format!(
                    "unknown voice {voice:?}; this checkpoint serves only {:?}",
                    self.voice.id
                ),
            });
        }
        if !(speed.is_finite() && speed > 0.0) {
            return Err(Error::InvalidArgument {
                arg: "speed",
                reason: format!("expected a finite positive rate, got {speed}"),
            });
        }
        if text.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }

        let ids = self.tokenizer.encode(text)?;
        if ids.is_empty() {
            return Err(Error::DataError {
                reason: "tokenizer produced no input ids for this text (empty or fully \
                          out-of-vocabulary)"
                    .into(),
            });
        }
        let seq_len = ids.len();
        let input_ids: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&input_ids, &[1, seq_len], &self.device)?;
        let attention_mask =
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; seq_len], &[1, seq_len], &self.device)?;

        let mut opts: VitsSynthOptions = self.model.default_synth_options();
        opts.speaking_rate = speed as f64;
        if let Some(seed) = self.seed {
            opts.seed = seed;
        }

        let out =
            self.model
                .synthesize(self.client.as_ref(), &input_ids, &attention_mask, &opts)?;
        Ok(out.waveform.contiguous()?.to_vec())
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.config().sampling_rate as u32
    }

    /// The tokenizer [`VitsEngine::synthesize`] itself uses. Exposed so a
    /// test can lock down that the engine's own tokenization path stays
    /// exactly [`VitsTokenizer`]'s (already oracle-tested against HF in
    /// `vits_tokenizer_oracle.rs`) rather than one that has quietly drifted.
    pub fn tokenizer(&self) -> &VitsTokenizer {
        &self.tokenizer
    }
}

impl crate::tts::TtsEngine for VitsEngine {
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        VitsEngine::synthesize(self, text, voice, speed)
    }

    fn sample_rate(&self) -> u32 {
        VitsEngine::sample_rate(self)
    }

    fn voices(&self) -> Vec<Voice> {
        vec![self.voice.clone()]
    }
}

/// Reads `tokenizer_config.json`'s `language` field (an ISO 639-3 code on
/// every published MMS-TTS checkpoint, e.g. `"zlm"`). Falls back to `"und"`
/// (ISO 639-2's "undetermined") when the field is absent.
fn read_language_code(dir: &Path) -> Result<String> {
    let path = dir.join("tokenizer_config.json");
    let bytes = std::fs::read(&path)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::ModelError {
            reason: format!("parsing {}: {e}", path.display()),
        })?;
    Ok(value
        .get("language")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| "und".to_string()))
}

/// Whether `config_path` names a VITS checkpoint (`"model_type": "vits"`).
pub fn is_vits_config(config_path: &Path) -> Result<bool> {
    let bytes = std::fs::read(config_path)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::ModelError {
            reason: format!("parsing {}: {e}", config_path.display()),
        })?;
    Ok(value.get("model_type").and_then(|v| v.as_str()) == Some("vits"))
}

/// Loads a [`VitsEngine`] from `dir` and wraps it in a [`TtsBundle`], the
/// same shape `blazr`'s loader gives a Kokoro or VoxCPM2 checkpoint: a
/// scaffolding bundle over the engine's own voice catalog, with the engine
/// attached.
pub fn load_vits_from_dir(dir: &Path) -> Result<TtsBundle> {
    let engine = VitsEngine::from_dir(dir)?;
    let sample_rate = engine.sample_rate();
    let voices = crate::tts::TtsEngine::voices(&engine);
    Ok(TtsBundle::scaffolding(voices, sample_rate).with_engine(Arc::new(engine)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_vits_config_reads_the_model_type_field() {
        let tmp = std::env::temp_dir().join("boostr_audio_vits_engine_config_check");
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("config.json");
        std::fs::write(&path, br#"{"model_type": "vits"}"#).unwrap();
        assert!(is_vits_config(&path).unwrap());
        std::fs::write(&path, br#"{"model_type": "kokoro"}"#).unwrap();
        assert!(!is_vits_config(&path).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
