//! Silero VAD: the streaming voice-activity model.
//!
//! See [`model`] for the forward pass and the 64-sample context contract that
//! silently produces garbage when skipped, [`state`] for the per-stream state,
//! [`config`] for the 8 kHz / 16 kHz geometry, and [`loader`] for checkpoint
//! loading. [`segment`] is the layer above the model: thresholding and
//! duration rules that turn per-chunk probabilities into utterance boundaries.

#[cfg(feature = "silero-vad")]
pub mod config;
#[cfg(feature = "silero-vad")]
pub mod loader;
#[cfg(feature = "silero-vad")]
pub mod model;
pub mod segment;
#[cfg(feature = "silero-vad")]
pub mod state;

#[cfg(feature = "silero-vad")]
pub use config::{ENCODER_KERNEL, ENCODER_STRIDES, HIDDEN_SIZE, STFT_FRAMES, VadConfig};
#[cfg(feature = "silero-vad")]
pub use model::{SileroVad, SileroVadWeights};
pub use segment::{SpeechSegment, VadSegmentOptions, segments_from_probabilities};
#[cfg(feature = "silero-vad")]
pub use state::VadState;

#[cfg(all(test, feature = "silero-vad"))]
mod tests;
