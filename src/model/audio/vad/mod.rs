//! Silero VAD: the streaming voice-activity model.
//!
//! See [`model`] for the forward pass and the 64-sample context contract that
//! silently produces garbage when skipped, [`state`] for the per-stream state,
//! [`config`] for the 8 kHz / 16 kHz geometry, and [`loader`] for checkpoint
//! loading. This is the MODEL only — thresholding and duration rules (the
//! segmentation layer) live elsewhere.

pub mod config;
pub mod loader;
pub mod model;
pub mod state;

pub use config::{ENCODER_KERNEL, ENCODER_STRIDES, HIDDEN_SIZE, STFT_FRAMES, VadConfig};
pub use model::{SileroVad, SileroVadWeights};
pub use state::VadState;

#[cfg(test)]
mod tests;
