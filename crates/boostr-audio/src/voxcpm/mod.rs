//! VoxCPM2 voice-cloning pipeline: reference recordings decoded from a voices
//! directory, text tokenized through splintr, and the [`VoxCpm2Engine`] that
//! drives boostr's `VoxCpm2Model`. Weights-source selection
//! (`VoxCpm2Weights`) stays with the model in boostr.

pub mod engine;
pub mod tokenizer;

pub use engine::{VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions, ZERO_SHOT_VOICE_ID};
pub use tokenizer::{load_tokenizer, normalize_whitespace, tokenize};
