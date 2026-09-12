//! The synthesis engine a [`TtsBundle`](super::bundle::TtsBundle) runs.
//!
//! A bundle holds the voice catalog and the request plumbing; the engine is
//! the model behind it. Kokoro and VoxCPM2 both implement this, so the same
//! `/v1/audio/speech` handler serves either from one `TtsBundle` type.

use crate::error::Result;
use crate::tts::bundle::Voice;

/// A text-to-waveform model.
///
/// `Send + Sync` because a server shares one engine across requests and runs
/// `synthesize` on blocking worker threads. An engine that must serialise
/// device work takes its own lock inside `synthesize`.
pub trait TtsEngine: Send + Sync {
    /// Render `text` in `voice` at `speed` (1.0 is the model's natural pace).
    ///
    /// Returns mono f32 samples at [`TtsEngine::sample_rate`]. An engine with
    /// no rate control refuses any `speed` other than 1.0 rather than
    /// ignoring it.
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>>;

    /// Sample rate of the returned waveform.
    fn sample_rate(&self) -> u32;

    /// The voices this engine can render, for the bundle's catalog.
    fn voices(&self) -> Vec<Voice>;
}
