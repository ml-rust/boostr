//! VoxCPM2 TTS components: [`client`] (shared trait bound), [`vae`]
//! (`AudioVAE` codec), [`local_encoder`] (`feat_encoder`), [`long_rope`]
//! (shared LongRoPE cache builder).

pub mod client;
pub mod loader;
pub mod local_encoder;
pub mod long_rope;
pub mod vae;

pub use client::VoxCpmClient;
