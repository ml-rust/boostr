//! NeuCodec acoustic decoder — architecture only, no weight loading.
//!
//! See [`decoder`] for the verified pipeline and [`resnet_block`] for the
//! GroupNorm-vs-LayerNorm decision.

pub mod alias_free;
pub mod client;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod istft_head;
pub mod loader;
pub mod resnet_block;
pub mod transformer_block;

pub use alias_free::{
    Activation1d, DownSample1d, SnakeBeta, UpSample1d, kaiser_sinc_filter1d, replicate_pad_1d,
};
pub use client::NeuCodecClient;
pub use codec::NeuCodec;
pub use config::NeuCodecDecoderConfig;
pub use decoder::{NeuCodecDecoder, NeuCodecDecoderWeights};
pub use istft_head::{IstftHead, IstftHeadWeights};
pub use loader::{
    DEFAULT_DECODER_PREFIX, DEFAULT_QUANTIZER_PREFIX, NEUCODEC_FSQ_LEVELS, load_fsq_quantizer,
};
pub use resnet_block::{ResnetBlock, ResnetBlockWeights};
pub use transformer_block::{TransformerBlock, TransformerBlockWeights};
