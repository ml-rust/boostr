//! Sample-level audio DSP, file codecs, voices, and product pipelines.
//!
//! Placement rule: code that operates on `Tensor<R>` with backend kernels
//! belongs in `boostr::model::audio`. Code that operates on `Vec<f32>`
//! samples, files, directories, voices, or text pipelines belongs here.
//!
//! Always compiled: RIFF/WAVE in and out ([`wav`]), the polyphase
//! [`resample`]r, the reference-take [`enhance`] chain, [`pitch`], [`quality`]
//! and ASR [`eval`] metrics, and the backend-free half of [`g2p`]. Feature
//! gates add the compressed-audio [`decode`]rs, the espeak-ng G2P backend, the
//! [`tts`] surface, the [`kokoro`] and [`voxcpm`] engines, and [`corpus`]
//! preparation.

pub mod enhance;
pub mod error;
pub mod eval;
pub mod g2p;
pub mod pitch;
pub mod quality;
pub mod resample;
pub mod wav;

#[cfg(feature = "corpus")]
pub mod corpus;
#[cfg(feature = "decode")]
pub mod decode;
#[cfg(feature = "kokoro")]
pub mod kokoro;
#[cfg(feature = "tts")]
pub mod tts;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;

pub use error::{Error, Result};

#[cfg(feature = "corpus")]
pub use corpus::{
    CorpusOptions, MAX_UTTERANCE_SECS, PRETRAINED_TOKENIZER_NAMES, SpeechCorpusBuilder,
    TextTokenizer, Utterance, check_max_speech_duration, pack_utterances,
    pack_utterances_with_layout,
};
#[cfg(feature = "decode")]
pub use decode::{decode_audio, decode_audio_file_mono_at, decode_audio_mono_at, extension_hint};
pub use eval::{
    ErrorRate, align, by_group, character_error_rate, grand_total, normalize, total,
    word_error_rate,
};
pub use g2p::{G2pError, Lang, Phonemizer};
pub use pitch::{PitchOptions, PitchTrack, estimate_pitch};
pub use quality::{TakeQuality, measure_quality};
pub use resample::{
    DEFAULT_TAPS_PER_PHASE, MAX_FILTER_TAPS, resample, resample_with_taps, to_mono_at_rate,
};
#[cfg(feature = "tts")]
pub use tts::{SynthesizeOptions, TtsBundle, TtsEngine, TtsError, Voice, default_kokoro_voices};
pub use wav::{WavData, decode_wav, encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16, to_mono};

#[cfg(all(test, feature = "decode"))]
pub(crate) mod test_utils {
    use std::path::PathBuf;

    /// Resolve a real-audio fixture: `$AUDIO_CORPUS_FLAC`, else the first
    /// `.flac` in `$AUDIO_CORPUS_DIR` by sorted name. `None` when neither is
    /// set or the resolved path is absent, so callers skip.
    pub(crate) fn corpus_flac() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AUDIO_CORPUS_FLAC") {
            let path = PathBuf::from(p);
            return path.exists().then_some(path);
        }
        let dir = PathBuf::from(std::env::var("AUDIO_CORPUS_DIR").ok()?);
        let mut flacs: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "flac"))
            .collect();
        flacs.sort();
        flacs.into_iter().next()
    }
}
