//! Engine-agnostic text-to-speech surface: the [`TtsEngine`] trait a model
//! implements and the [`TtsBundle`] that owns one plus its voice catalog.

pub mod bundle;
pub mod engine;
#[cfg(feature = "vits")]
pub mod vits;

pub use bundle::{SynthesizeOptions, TtsBundle, TtsError, Voice, default_kokoro_voices};
pub use engine::TtsEngine;
#[cfg(feature = "vits")]
pub use vits::{VitsEngine, VitsTokenizer, is_vits_config, load_vits_from_dir};
