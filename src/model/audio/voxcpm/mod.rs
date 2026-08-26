//! VoxCPM2 `AudioVAE` decoder: latent `[B, 64, T]` -> waveform `[B, 1, T*1920]`.
//!
//! See [`decoder::AudioVaeDecoder`] for the top-level assembly and geometry.

pub mod causal_conv1d;
pub mod causal_transpose_conv1d;
pub mod client;
pub mod decoder;
pub mod decoder_block;
pub mod loader;
pub mod res_unit;
pub mod snake;

pub use causal_conv1d::CausalConv1d;
pub use causal_transpose_conv1d::CausalTransposeConv1d;
pub use client::VoxCpmClient;
pub use decoder::{AudioVaeDecoder, AudioVaeDecoderWeights, DEFAULT_SR_BUCKET};
pub use decoder_block::{DecoderBlock, DecoderBlockWeights};
pub use loader::DEFAULT_DECODER_PREFIX;
pub use res_unit::ResUnit;
pub use snake::Snake;
