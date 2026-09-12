//! Engine-agnostic text-to-speech surface: the [`TtsEngine`] trait a model
//! implements and the [`TtsBundle`] that owns one plus its voice catalog.

pub mod bundle;
pub mod engine;

pub use bundle::{SynthesizeOptions, TtsBundle, TtsError, Voice, default_kokoro_voices};
pub use engine::TtsEngine;
