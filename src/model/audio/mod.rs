//! Tensor-level audio: signal ops every speech model builds on, and the
//! per-architecture model trees behind their feature gates.
//!
//! Sample-level DSP, file codecs, voices and product pipelines live in the
//! `boostr-audio` crate, which depends on this module and never the reverse.

#[cfg(feature = "kokoro")]
pub mod kokoro;
pub mod mel;
#[cfg(feature = "neucodec")]
pub mod neucodec;
pub mod phoneme_vocab;
pub mod reflection_pad;
pub mod stft;
pub mod vad;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;
#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisper")]
pub mod whisper_decoder;
#[cfg(feature = "whisper")]
pub mod whisper_loader;
#[cfg(feature = "whisper")]
pub mod whisper_model;
#[cfg(feature = "whisper")]
pub mod whisper_transcribe;

pub use mel::{
    LogSpec, MelNorm, MelOptions, MelScale, compute_mel_spectrogram, compute_mel_spectrogram_with,
};
pub use phoneme_vocab::{PhonemeVocab, phonemes_to_ids};
pub use reflection_pad::reflection_pad_1d;
pub use stft::{
    IStftClient, IStftOptions, IStftPadding, StftClient, StftOptions, hann_window, istft, stft,
};
#[cfg(feature = "silero-vad")]
pub use vad::{SileroVad, SileroVadWeights, VadConfig, VadState};
pub use vad::{SpeechSegment, VadSegmentOptions, segments_from_probabilities};
#[cfg(feature = "whisper")]
pub use whisper::WhisperEncoder;
#[cfg(feature = "whisper")]
pub use whisper_decoder::{DecoderCache, DecoderLayerCache, WhisperDecoder, WhisperDecoderLayer};
#[cfg(feature = "whisper")]
pub use whisper_loader::{WhisperBundle, WhisperGenerationConfig};
#[cfg(feature = "whisper")]
pub use whisper_model::{GenerateOptions, WhisperModel};
#[cfg(feature = "whisper")]
pub use whisper_transcribe::{TranscribeOptions, Transcription};
