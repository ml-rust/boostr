//! `KokoroEngine` — the ready-to-call synthesis bundle.
//!
//! Ties together:
//!
//! * [`KokoroModelV2`] — the neural model
//! * [`KokoroG2p`] — misaki G2P + reference Kokoro's phoneme-budget chunking
//! * [`KokoroPhonemeVocab`] — IPA → token-id mapping
//! * [`VoiceResolver`] — where to find voice packs on disk
//! * The config (sample rate, min frames/phoneme, …) used at synthesis
//!
//! The engine is the piece that [`crate::tts::TtsBundle`] owns when
//! the neural path is live. Keeping it separate keeps `TtsBundle` scaffolding
//! callable unchanged — if no engine is attached, `synthesize` returns
//! `NotImplemented` exactly as it always did.

use crate::error::{Error, Result};
use crate::kokoro::g2p::{Dialect, KokoroG2p, phonemes_to_input_ids};
use crate::kokoro::voice::VoiceResolver;
use boostr::model::audio::kokoro::{KokoroModelV2, KokoroPhonemeVocab, select_voice_style};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;
use std::path::Path;
use std::sync::Arc;

pub struct KokoroEngine {
    pub model: KokoroModelV2<CpuRuntime>,
    pub vocab: KokoroPhonemeVocab,
    pub resolver: VoiceResolver,
    pub client: Arc<CpuClient>,
    pub device: Arc<<CpuRuntime as numr::runtime::Runtime>::Device>,
    pub min_frames_per_phoneme: u32,
    pub g2p: KokoroG2p,
}

impl KokoroEngine {
    /// Run the full text-to-waveform pipeline.
    ///
    /// * `text` — UTF-8 input text. Split into reference-Kokoro-compatible
    ///   chunks internally; a script over the 510-phoneme-character budget
    ///   runs as several model calls, concatenated in order.
    /// * `voice_spec` — voice id (bundled or present in `resolver`'s asset
    ///   directory) or absolute path. See [`VoiceResolver::resolve_path`].
    ///   Its id's first character selects the G2P dialect: `a` = American
    ///   English, `b` = British English; any other prefix errs.
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

        // 1. Dialect from the voice id, G2P + chunk the text.
        let dialect = Dialect::from_voice_id(voice_id(voice_spec))?;
        let chunks = self.g2p.chunks(text, dialect)?;

        // 2. Resolve + load the voice pack once; each chunk picks its own row.
        let voice_pack = self.resolver.load::<CpuRuntime>(voice_spec, &self.device)?;

        let floor = ((self.min_frames_per_phoneme as f32) / speed.max(0.1))
            .round()
            .max(1.0) as u32;

        let mut waveform = Vec::new();
        for chunk in &chunks {
            // Phonemes → ids → [1, T_phon] tensor, reference Kokoro's
            // `KModel.forward`: one id per known phoneme character,
            // bracketed by the reserved BOS/EOS id 0.
            let ids = phonemes_to_input_ids(&self.vocab, &chunk.phonemes);
            if ids.len() <= 2 {
                // Every character in this chunk fell outside the vocab.
                continue;
            }
            let token_ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &self.device)?;

            // Voice row selection reads the phoneme-character count, not
            // the id count — reference Kokoro's `pack[len(ps) - 1]`.
            let voice_row = select_voice_style(&voice_pack, chunk.phonemes.chars().count())?;

            let chunk_waveform =
                self.model
                    .synthesize_cpu(&self.client, &token_ids, &voice_row, floor)?;
            waveform.extend(chunk_waveform.contiguous()?.to_vec::<f32>());
        }

        if waveform.is_empty() {
            return Err(Error::DataError {
                reason: "no phonemes mapped to vocab — check that G2P and vocab agree".into(),
            });
        }
        Ok(waveform)
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.config.sample_rate
    }
}

/// The voice id a spec names, for dialect selection: a bare id is used
/// as-is, a path's file stem is used instead (so `./voices/af_heart.pt`
/// still reads as `af_heart`).
fn voice_id(voice_spec: &str) -> &str {
    Path::new(voice_spec)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(voice_spec)
}

impl crate::tts::TtsEngine for KokoroEngine {
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        KokoroEngine::synthesize(self, text, voice, speed)
    }

    fn sample_rate(&self) -> u32 {
        KokoroEngine::sample_rate(self)
    }

    /// The canonical catalog; voice packs beyond it resolve through
    /// `VoiceResolver` at synthesis time.
    fn voices(&self) -> Vec<crate::tts::Voice> {
        crate::tts::default_kokoro_voices()
    }
}

#[cfg(test)]
mod tests {
    use super::voice_id;

    // We can't instantiate a full KokoroEngine in a unit test without a real
    // checkpoint (all primitives require non-empty weight tensors with
    // matching shapes). The engine's responsibility is wiring, not model
    // behavior. Test the small helpers here; end-to-end is covered by the
    // integration tests in blazr once a voice/model pair is available.

    #[test]
    fn voice_id_reads_bare_ids_and_path_stems() {
        assert_eq!(voice_id("af_heart"), "af_heart");
        assert_eq!(voice_id("./voices/af_heart.pt"), "af_heart");
        assert_eq!(voice_id("/abs/path/bf_emma.safetensors"), "bf_emma");
    }
}
