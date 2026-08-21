//! Text-to-speech language model: a causal decoder over one flat vocabulary
//! holding both text tokens and neural-audio-codec tokens.

pub mod codec;
pub mod vocab;

pub use codec::CodecVocab;
pub use vocab::{SpecialToken, SpeechVocab};
