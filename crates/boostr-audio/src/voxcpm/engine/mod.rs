//! VoxCPM2 as a [`TtsEngine`](crate::tts::TtsEngine): the clone pipeline
//! behind one `synthesize`, and the same pipeline emitting audio chunk by
//! chunk behind `synthesize_stream`.
//!
//! A voice is a reference recording. The engine encodes every recording in a
//! voices directory once at load, so a request pays only its own prefill,
//! generation and decode. Text reaches the model as raw tokens, so a Malay
//! and English code-switched sentence needs no language switch.
//!
//! One render runs at a time: `synthesize` and `synthesize_stream` hold a
//! lock for their duration. The model's KV caches and generation state are
//! per call, so the lock serialises device work, not correctness.
//!
//! [`types`] holds the engine struct and its constants, [`load`] the loader
//! and voice-directory handling, [`render`] the request path.

mod load;
mod render;
mod types;

pub use types::{VoxCpm2Engine, ZERO_SHOT_VOICE_ID};
