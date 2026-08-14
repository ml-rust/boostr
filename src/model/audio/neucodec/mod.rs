//! NeuCodec acoustic decoder — architecture only, no weight loading.
//!
//! See [`decoder`] for the verified pipeline and [`resnet_block`] for the
//! GroupNorm-vs-LayerNorm decision.

pub mod client;
pub mod config;
pub mod decoder;
pub mod istft_head;
pub mod loader;
pub mod resnet_block;
pub mod transformer_block;

pub use client::NeuCodecClient;
pub use config::NeuCodecDecoderConfig;
pub use decoder::{NeuCodecDecoder, NeuCodecDecoderWeights};
pub use istft_head::{IstftHead, IstftHeadWeights};
pub use loader::DEFAULT_DECODER_PREFIX;
pub use resnet_block::{ResnetBlock, ResnetBlockWeights};
pub use transformer_block::{TransformerBlock, TransformerBlockWeights};
