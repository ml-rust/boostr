//! VoxCPM2 TTS components: [`client`] (shared trait bound), [`vae`]
//! (`AudioVAE` codec), [`local_encoder`] (`feat_encoder`, LongRoPE via
//! `crate::nn::RoPE::precompute_freqs`).

pub mod client;
pub mod loader;
pub mod local_encoder;
pub mod vae;

pub use client::VoxCpmClient;
