//! NeuCodec: the full waveform <-> FSQ-code path.
//!
//! See [`encoder`] for the 16 kHz-waveform-to-indices direction (both branches,
//! the prior projection and the quantizer), [`decoder`] for the reverse
//! direction, [`resnet_block`] for the
//! GroupNorm-vs-LayerNorm decision, [`fbank`] for the semantic branch's
//! Kaldi-compatible feature frontend, and [`semantic_encoder`] for the
//! 16-layer Wav2Vec2-BERT conformer that consumes those features.

pub mod acoustic_encoder;
pub mod alias_free;
pub mod client;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod fbank;
pub mod istft_head;
pub mod loader;
mod reexports;
pub mod resnet_block;
pub mod semantic_adapter;
pub mod semantic_encoder;
pub mod transformer_block;

pub use reexports::*;
