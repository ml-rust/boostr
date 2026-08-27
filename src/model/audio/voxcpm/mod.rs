//! VoxCPM2 TTS components: [`client`] (shared trait bound), [`vae`]
//! (`AudioVAE` codec), [`bidirectional`] (shared MiniCPM4 blocks for
//! `feat_encoder`/`feat_decoder`), [`local_encoder`] (`feat_encoder`),
//! [`minicpm4`] (the `base_lm` decoder), [`fsq`] (the `fsq_layer` bottleneck
//! and its six sibling projections), [`local_dit`] (`feat_decoder`),
//! [`model`] (the end-to-end orchestrator: reference encode, two-LM prefill,
//! per-patch generation loop). Both transformers get LongRoPE via
//! `crate::nn::RoPE::precompute_freqs`.

pub mod bidirectional;
pub mod client;
pub mod fsq;
pub mod loader;
pub mod local_dit;
pub mod local_encoder;
pub mod minicpm4;
pub mod model;
pub mod tokenizer;
pub mod vae;

pub use client::VoxCpmClient;
pub use model::{PrefillState, VoxCpm2Config, VoxCpm2Model};
pub use tokenizer::{load_tokenizer, normalize_whitespace, tokenize};
