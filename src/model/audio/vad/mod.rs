//! Silero VAD: the streaming voice-activity model.
//!
//! See [`model`] for the forward pass and the 64-sample context contract that
//! silently produces garbage when skipped, [`state`] for the per-stream state,
//! [`config`] for the 8 kHz / 16 kHz geometry, and [`loader`] for checkpoint
//! loading. The layer above the model — thresholding and duration rules that
//! turn per-chunk probabilities into utterance boundaries — is
//! `boostr_audio::vad`, which operates on sample buffers rather than tensors.

pub mod config;
mod forward;
pub mod loader;
pub mod model;
pub mod state;

pub use config::{ENCODER_KERNEL, ENCODER_STRIDES, HIDDEN_SIZE, STFT_FRAMES, VadConfig};
pub use model::{SileroVad, SileroVadWeights};
pub use state::VadState;
