//! VoxCPM2 TTS components: [`client`] (shared trait bound), [`vae`]
//! (`AudioVAE` codec), [`bidirectional`] (shared MiniCPM4 blocks for
//! `feat_encoder`/`feat_decoder`), [`local_encoder`] (`feat_encoder`),
//! [`minicpm4`] (the `base_lm` decoder), [`fsq`] (the `fsq_layer` bottleneck
//! and its six sibling projections), [`local_dit`] (`feat_decoder`),
//! [`model`] (the end-to-end orchestrator: reference encode, two-LM prefill,
//! per-patch generation loop), [`weights`] (which files the stack loads from),
//! [`gguf_keys`] (the metadata keys a single-file GGUF carries).
//! Both transformers get LongRoPE via `crate::nn::RoPE::precompute_freqs`.
//! The clone engine, voice directory and tokenizer live in `boostr-audio`.

pub mod bidirectional;
pub mod client;
pub mod fsq;
pub mod gguf_keys;
pub mod loader;
pub mod local_dit;
pub mod local_encoder;
pub mod minicpm4;
pub mod model;
pub mod vae;
pub mod weights;

pub use client::VoxCpmClient;
pub use gguf_keys::{GGUF_CONFIG_JSON_KEY, GGUF_TOKENIZER_JSON_KEY};
pub use model::{LoraAdapterReport, PrefillState, VoxCpm2Config, VoxCpm2Model};
pub use weights::{TokenizerSource, VoxCpm2Weights};
