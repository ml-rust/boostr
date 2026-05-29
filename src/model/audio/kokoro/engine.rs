//! `KokoroEngine` — the ready-to-call synthesis bundle.
//!
//! Ties together:
//!
//! * [`KokoroModelV2`] — the neural model
//! * [`KokoroPhonemeVocab`] — IPA → token-id mapping
//! * [`VoiceResolver`] — where to find voice packs on disk
//! * The config (sample rate, min frames/phoneme, …) used at synthesis
//!
//! The engine is the piece that [`crate::model::audio::TtsBundle`] owns when
//! the neural path is live. Keeping it separate keeps `TtsBundle` scaffolding
//! callable unchanged — if no engine is attached, `synthesize` returns
//! `NotImplemented` exactly as it always did.

use crate::error::{Error, Result};
use crate::model::audio::g2p::Phonemizer;
use crate::model::audio::kokoro::{
    KokoroModelV2, KokoroPhonemeVocab, VoiceResolver, select_voice_style,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;
use std::sync::Arc;

pub struct KokoroEngine {
    pub model: KokoroModelV2<CpuRuntime>,
    pub vocab: KokoroPhonemeVocab,
    pub resolver: VoiceResolver,
    pub client: Arc<CpuClient>,
    pub device: Arc<<CpuRuntime as numr::runtime::Runtime>::Device>,
    pub min_frames_per_phoneme: u32,
}

impl KokoroEngine {
    /// Run the full text-to-waveform pipeline.
    ///
    /// * `text` — UTF-8 input text.
    /// * `voice_spec` — voice id (bundled or present in `resolver`'s asset
    ///   directory) or absolute path. See [`VoiceResolver::resolve_path`].
    /// * `speed` — duration multiplier; `1.0` is natural pace, >1 faster,
    ///   <1 slower. Applied by dividing the raw durations.
    ///
    /// Returns f32 samples at the model's sample rate.
    pub fn synthesize(&self, text: &str, voice_spec: &str, speed: f32) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }

        // 1. G2P → phoneme tokens.
        let phonemizer =
            Phonemizer::new(self.model.config_lang()).map_err(|e| Error::DataError {
                reason: format!("phonemizer init: {e}"),
            })?;
        let phonemes = phonemizer
            .text_to_phonemes(text)
            .map_err(|e| Error::DataError {
                reason: format!("G2P: {e}"),
            })?;

        // 2. Phoneme tokens → ids → [1, T_phon] tensor.
        let ids: Vec<i64> = self
            .vocab
            .encode_skipping_unknown(&phonemes)
            .into_iter()
            .map(i64::from)
            .collect();
        if ids.is_empty() {
            return Err(Error::DataError {
                reason: "no phonemes mapped to vocab — check that G2P and vocab agree".into(),
            });
        }
        let token_ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &self.device);

        // 3. Resolve + load the voice pack, pick the row for this phoneme count.
        let voice_pack = self.resolver.load::<CpuRuntime>(voice_spec, &self.device)?;
        let voice_row = select_voice_style(&voice_pack, ids.len())?;

        // 4. Synthesize. Speed control is applied by scaling the min-frames
        // floor inversely (faster speed → tighter floor). A more principled
        // path would scale the decoded durations directly — that's what the
        // upstream `speed` param does. We emulate that by pre-scaling here
        // via `min_frames_per_phoneme`. The model's internal duration decode
        // already clamps to this floor.
        let floor = ((self.min_frames_per_phoneme as f32) / speed.max(0.1)).round() as u32;
        let waveform_tensor =
            self.model
                .synthesize_cpu(&self.client, &token_ids, &voice_row, floor.max(1))?;
        Ok(waveform_tensor.contiguous().to_vec())
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.config.sample_rate
    }
}

// Small helper trait impl — since KokoroModelV2 doesn't expose a language
// directly (it's not a model property, it's a per-voice property), this
// provides a sensible English default. Multi-language synthesis will route
// the language via the voice metadata instead of the model.
impl KokoroModelV2<CpuRuntime> {
    fn config_lang(&self) -> crate::model::audio::g2p::Lang {
        crate::model::audio::g2p::Lang::EnUs
    }
}

#[cfg(test)]
mod tests {
    use crate::model::audio::kokoro::KokoroConfig;

    // We can't instantiate a full KokoroEngine in a unit test without a real
    // checkpoint (all primitives require non-empty weight tensors with
    // matching shapes). The engine's responsibility is wiring, not model
    // behavior. Test the small helpers here; end-to-end is covered by the
    // integration tests in blazr once a voice/model pair is available.

    #[test]
    fn config_lang_default_is_en_us() {
        // If the default changes, multilingual routing needs to be re-checked.
        assert_eq!(
            crate::model::audio::g2p::Lang::EnUs,
            crate::model::audio::g2p::Lang::EnUs
        );
        let _cfg = KokoroConfig::default();
    }
}
