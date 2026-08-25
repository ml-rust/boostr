//! Silero VAD: the streaming voice-activity model.
//!
//! See [`model`] for the forward pass and the 64-sample context contract that
//! silently produces garbage when skipped, [`state`] for the per-stream state,
//! [`config`] for the 8 kHz / 16 kHz geometry, and [`loader`] for checkpoint
//! loading. [`segment`] is the layer above the model: thresholding and
//! duration rules that turn per-chunk probabilities into utterance boundaries.

pub mod config;
pub mod loader;
pub mod model;
pub mod segment;
pub mod state;

pub use config::{ENCODER_KERNEL, ENCODER_STRIDES, HIDDEN_SIZE, STFT_FRAMES, VadConfig};
pub use model::{SileroVad, SileroVadWeights};
pub use segment::{SpeechSegment, VadSegmentOptions, segments_from_probabilities};
pub use state::VadState;

#[cfg(test)]
mod tests;
