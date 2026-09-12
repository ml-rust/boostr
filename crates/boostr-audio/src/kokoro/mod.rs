//! Kokoro synthesis pipeline: G2P, voice-pack lookup on disk, and the
//! [`KokoroEngine`] that drives boostr's `KokoroModelV2`.

pub mod engine;
pub mod voice;

pub use engine::KokoroEngine;
pub use voice::{VoiceResolver, resolve_and_load};
