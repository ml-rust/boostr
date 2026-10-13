//! Tensor-level audio: signal ops every speech model builds on, and the
//! per-architecture model trees behind their feature gates.
//!
//! Sample-level DSP, file codecs, voices and product pipelines live in the
//! `voxr` crate, which depends on this module and never the reverse.

#[cfg(feature = "clap")]
pub mod clap;
#[cfg(feature = "kokoro")]
pub mod kokoro;
pub mod mel;
#[cfg(feature = "neucodec")]
pub mod neucodec;
pub mod phoneme_vocab;
mod reexports;
pub mod stft;
#[cfg(feature = "silero-vad")]
pub mod vad;
#[cfg(feature = "vits")]
pub mod vits;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;
#[cfg(feature = "whisper")]
mod whisper_stack;

pub use reexports::*;
