//! Text-to-speech bundle: G2P + neural acoustic model + vocoder + voice
//! catalog.
//!
//! A bundle can run in two modes. Without an attached [`KokoroEngine`] it
//! validates the surrounding pipeline — tokenizer validation, voice lookup,
//! G2P, and WAV encoding — and returns [`TtsError::NotImplemented`] from the
//! synthesis step, which lets callers assert their plumbing is correct without
//! loading model weights. With an engine attached via [`TtsBundle::with_engine`],
//! `synthesize` runs the full `text → waveform` neural path.

use std::sync::Arc;

use thiserror::Error;

use super::g2p::{G2pError, Lang, Phonemizer};
use super::kokoro::KokoroEngine;

/// Errors raised by the TTS pipeline.
#[derive(Debug, Error)]
pub enum TtsError {
    #[error("voice '{0}' not registered on this bundle")]
    UnknownVoice(String),
    #[error("G2P error: {0}")]
    G2p(#[from] G2pError),
    #[error("neural synthesis requires an engine; this bundle was created without one")]
    NotImplemented,
    #[error("load error: {0}")]
    Load(String),
    #[error("engine error: {0}")]
    Engine(String),
}

/// A single voice the bundle supports.
///
/// Kokoro's canonical naming: two-letter prefix (`af` = American Female,
/// `am` = American Male, `bf` = British Female, `bm` = British Male, `jf/jm`
/// = Japanese, etc.) followed by a display name (`alloy`, `nova`, `adam`,
/// `alice`, …).
#[derive(Debug, Clone)]
pub struct Voice {
    pub id: String,
    pub lang: Lang,
    /// Display name (used in API error messages and catalogs).
    pub display_name: String,
}

impl Voice {
    pub fn new(id: impl Into<String>, lang: Lang, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            lang,
            display_name: display_name.into(),
        }
    }
}

/// Configuration controlling a synthesis call.
#[derive(Debug, Clone)]
pub struct SynthesizeOptions {
    /// Playback-rate multiplier applied to the duration predictor. Kokoro
    /// supports ~0.25×..4×. Enforced by the caller (blazr).
    pub speed: f32,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

/// End-to-end TTS bundle.
///
/// Holds the voice catalog and, optionally, a neural synthesis engine. Without
/// an engine, only the G2P + voice-catalog portions are functional and
/// `synthesize` returns [`TtsError::NotImplemented`].
pub struct TtsBundle {
    voices: Vec<Voice>,
    /// Sample rate of generated waveforms (Kokoro is 24 kHz).
    pub sample_rate: u32,
    /// Neural synthesis engine. When `None`, `synthesize` returns
    /// [`TtsError::NotImplemented`] (scaffolding mode). When `Some`, the full
    /// `text → waveform` pipeline runs.
    engine: Option<Arc<KokoroEngine>>,
}

impl TtsBundle {
    /// Create an engine-less bundle with a fixed voice catalog. Useful for
    /// validating the surrounding pipeline (voice lookup, G2P) without loading
    /// model weights; `synthesize` returns [`TtsError::NotImplemented`].
    pub fn scaffolding(voices: Vec<Voice>, sample_rate: u32) -> Self {
        Self {
            voices,
            sample_rate,
            engine: None,
        }
    }

    /// Attach a real synthesis engine. Once attached, `synthesize` runs the
    /// full neural path instead of returning `NotImplemented`.
    pub fn with_engine(mut self, engine: Arc<KokoroEngine>) -> Self {
        self.sample_rate = engine.sample_rate();
        self.engine = Some(engine);
        self
    }

    /// Whether this bundle has a live neural engine.
    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    /// All voices this bundle exposes.
    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    /// Look up a voice by id (e.g. `"af_alloy"`).
    pub fn voice(&self, id: &str) -> Option<&Voice> {
        self.voices.iter().find(|v| v.id == id)
    }

    /// Run G2P for a text string using the language of `voice`.
    ///
    /// This is useful in two places:
    /// 1. As the first step of the neural synthesis path.
    /// 2. To validate the input pipeline end-to-end when no engine is attached.
    pub fn phonemize(&self, text: &str, voice_id: &str) -> Result<Vec<String>, TtsError> {
        let voice = self
            .voice(voice_id)
            .ok_or_else(|| TtsError::UnknownVoice(voice_id.to_string()))?;
        let phonemizer = Phonemizer::new(voice.lang)?;
        Ok(phonemizer.text_to_phonemes(text)?)
    }

    /// Synthesize a waveform.
    ///
    /// If an engine is attached, runs the full neural pipeline end-to-end.
    /// Otherwise returns [`TtsError::NotImplemented`] after validating voice
    /// and G2P wiring — same behavior as before the engine landed, so the
    /// server surfaces configuration issues fast.
    pub fn synthesize(
        &self,
        text: &str,
        voice_id: &str,
        options: &SynthesizeOptions,
    ) -> Result<Vec<f32>, TtsError> {
        match &self.engine {
            Some(engine) => engine
                .synthesize(text, voice_id, options.speed)
                .map_err(|e| TtsError::Engine(e.to_string())),
            None => {
                // Scaffolding path: validate voice + run G2P eagerly, then fail
                // with `NotImplemented`. Exercises the plumbing end-to-end so
                // misconfigured endpoints surface before synthesis is even
                // attempted.
                let _phonemes = self.phonemize(text, voice_id)?;
                let _ = options.speed;
                Err(TtsError::NotImplemented)
            }
        }
    }
}

/// Default Kokoro voice catalog (a representative subset). These match the
/// canonical Kokoro voice ids so configuration files referencing them remain
/// valid once voice packs are loaded.
pub fn default_kokoro_voices() -> Vec<Voice> {
    vec![
        Voice::new("af_alloy", Lang::EnUs, "Alloy"),
        Voice::new("af_nova", Lang::EnUs, "Nova"),
        Voice::new("af_bella", Lang::EnUs, "Bella"),
        Voice::new("am_adam", Lang::EnUs, "Adam"),
        Voice::new("am_michael", Lang::EnUs, "Michael"),
        Voice::new("bf_alice", Lang::EnGb, "Alice"),
        Voice::new("bf_emma", Lang::EnGb, "Emma"),
        Voice::new("bm_daniel", Lang::EnGb, "Daniel"),
        Voice::new("bm_george", Lang::EnGb, "George"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolding_bundle_exposes_voices() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        assert!(bundle.voice("af_alloy").is_some());
        assert!(bundle.voice("not-a-voice").is_none());
        assert_eq!(bundle.sample_rate, 24_000);
    }

    #[test]
    fn synthesize_returns_not_implemented() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        let err = bundle
            .synthesize("hello", "af_alloy", &SynthesizeOptions::default())
            .unwrap_err();
        // Without tts-g2p feature we'd hit FeatureDisabled first; otherwise
        // the G2P step runs and we get NotImplemented from the synth step.
        #[cfg(feature = "tts-g2p")]
        assert!(matches!(err, TtsError::NotImplemented));
        #[cfg(not(feature = "tts-g2p"))]
        assert!(matches!(err, TtsError::G2p(G2pError::FeatureDisabled)));
    }

    #[test]
    fn synthesize_validates_voice_first() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        let err = bundle
            .synthesize("hello", "ghost_voice", &SynthesizeOptions::default())
            .unwrap_err();
        assert!(matches!(err, TtsError::UnknownVoice(_)));
    }

    #[test]
    fn default_catalog_has_known_voices() {
        let voices = default_kokoro_voices();
        assert!(voices.iter().any(|v| v.id == "af_nova"));
        assert!(voices.iter().any(|v| v.id == "bm_george"));
    }
}
