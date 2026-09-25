//! VoxCPM2 as a [`TtsEngine`](crate::tts::TtsEngine): the clone pipeline
//! behind one `synthesize`, and the same pipeline emitting audio chunk by
//! chunk behind `synthesize_stream`.
//!
//! A voice is a reference recording. The engine encodes every recording in a
//! voices directory once at load, so a request pays only its own prefill,
//! generation and decode. Text reaches the model as raw tokens, so a Malay
//! and English code-switched sentence needs no language switch.
//!
//! Concurrent requests share one generation. Both entry points stay `&self`
//! and blocking: they resolve their request, queue it and wait, while the
//! engine's own thread runs whatever is queued at the same time as ONE
//! batched generation. A row's noise draw is independent of the batch width
//! and of its neighbours, so a batched render is bit for bit the render that
//! row would have had alone.
//!
//! [`types`] holds the engine struct, the shared state and the constants,
//! [`load`] the loader and voice-directory handling, [`render`] the caller's
//! half of a request, [`batch`] the queue and the owner thread, [`run`] the
//! batched generation itself, [`continuous`] the persistent batch behind
//! [`VoxCpm2SynthOptions::continuous_batching`](crate::voxcpm::VoxCpm2SynthOptions::continuous_batching).

mod batch;
mod continuous;
mod load;
mod render;
mod run;
mod types;

pub use types::{VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID};
