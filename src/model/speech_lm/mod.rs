//! Text-to-speech language model: a causal decoder over one flat vocabulary
//! holding both text tokens and neural-audio-codec tokens.

pub mod codec;
pub mod frame;
pub mod special;
pub mod vocab;

pub use codec::CodecVocab;
pub use special::{ALL_SPECIAL_TOKENS, DEFAULT_CONTROL_REGION, SpecialToken};
pub use vocab::SpeechVocab;
