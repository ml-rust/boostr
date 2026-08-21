//! Text-to-speech language model: a causal decoder over one flat vocabulary
//! holding both text tokens and neural-audio-codec tokens.

pub mod vocab;

pub use vocab::{CodecVocab, SpecialToken, SpeechVocab};
