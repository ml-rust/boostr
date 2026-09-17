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

    /// Render `text` in `voice` at `speed`, handing the waveform to `sink`
    /// in order, chunk by chunk, as it becomes available.
    ///
    /// The concatenation of every chunk equals [`TtsEngine::synthesize`]'s
    /// output for the same request. A `sink` error aborts the render and is
    /// returned as-is, so a caller whose client went away can stop the
    /// engine mid-utterance. The default renders whole and hands over one
    /// chunk; an engine that can decode incrementally overrides it.
    fn synthesize_stream(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let samples = self.synthesize(text, voice, speed)?;
        sink(&samples)
    }

    /// Sample rate of the returned waveform.
    fn sample_rate(&self) -> u32;

    /// The voices this engine can render, for the bundle's catalog.
    fn voices(&self) -> Vec<Voice>;
}
