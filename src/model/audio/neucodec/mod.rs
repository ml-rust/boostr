//! NeuCodec acoustic decoder — architecture only, no weight loading.
//!
//! See [`decoder`] for the verified pipeline and [`resnet_block`] for the
//! GroupNorm-vs-LayerNorm decision.

pub mod acoustic_encoder;
pub mod alias_free;
pub mod client;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod istft_head;
pub mod loader;
pub mod resnet_block;
pub mod semantic_adapter;
pub mod transformer_block;

pub use acoustic_encoder::{AcousticEncoder, EncoderBlock, ResidualUnit, encoder_hop_length};
pub use alias_free::{
    Activation1d, DownSample1d, SnakeBeta, UpSample1d, kaiser_sinc_filter1d, replicate_pad_1d,
};
pub use client::NeuCodecClient;
pub use codec::NeuCodec;
pub use config::NeuCodecDecoderConfig;
pub use decoder::{NeuCodecDecoder, NeuCodecDecoderWeights};
pub use istft_head::{IstftHead, IstftHeadWeights};
pub use loader::{
    DEFAULT_ACOUSTIC_ENCODER_PREFIX, DEFAULT_DECODER_PREFIX, DEFAULT_QUANTIZER_PREFIX,
    DEFAULT_SEMANTIC_ADAPTER_PREFIX, NEUCODEC_FSQ_LEVELS, load_acoustic_encoder,
    load_fsq_quantizer, load_semantic_adapter,
};
pub use resnet_block::{ResnetBlock, ResnetBlockWeights};
pub use semantic_adapter::{SemanticAdapter, SemanticAdapterWeights};
pub use transformer_block::{TransformerBlock, TransformerBlockWeights};
