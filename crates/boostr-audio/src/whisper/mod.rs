//! Whisper as a product pipeline: the checkpoint's weights from boostr plus
//! the tokenizer and decoding constraints that make it transcribe.
//!
//! [`bundle`] loads and holds the pieces; [`transcribe`] runs the front end,
//! encoder, greedy decode and detokenizer in one call.

pub mod bundle;
pub mod transcribe;

pub use bundle::{WhisperBundle, WhisperGenerationConfig};
pub use transcribe::{TranscribeOptions, Transcription};
