//! NeuCodec acoustic decoder — architecture only, no weight loading.
//!
//! See [`decoder`] for the verified pipeline, [`resnet_block`] for the
//! GroupNorm-vs-LayerNorm decision, [`fbank`] for the semantic branch's
//! Kaldi-compatible feature frontend, and [`semantic_encoder`] for the
//! 16-layer Wav2Vec2-BERT conformer that consumes those features.

pub mod acoustic_encoder;
pub mod alias_free;
pub mod client;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod fbank;
pub mod istft_head;
pub mod loader;
pub mod resnet_block;
pub mod semantic_adapter;
pub mod semantic_encoder;
pub mod transformer_block;

pub use acoustic_encoder::{AcousticEncoder, EncoderBlock, ResidualUnit, encoder_hop_length};
pub use alias_free::{
    Activation1d, DownSample1d, SnakeBeta, UpSample1d, kaiser_sinc_filter1d, replicate_pad_1d,
};
pub use client::NeuCodecClient;
pub use codec::NeuCodec;
pub use config::NeuCodecDecoderConfig;
pub use decoder::{NeuCodecDecoder, NeuCodecDecoderWeights};
pub use fbank::{
    FFT_LENGTH, FRAME_LENGTH, FRAME_SHIFT, HIGH_FREQ, LOW_FREQ, MEL_FLOOR, NUM_FFT_BINS,
    NUM_MEL_BINS, SAMPLE_RATE, STACKED_DIM, hz_to_mel, mel_filterbank, mel_to_hz, num_frames,
    povey_window, seamless_fbank,
};
pub use istft_head::{IstftHead, IstftHeadWeights};
pub use loader::{
    DEFAULT_ACOUSTIC_ENCODER_PREFIX, DEFAULT_DECODER_PREFIX, DEFAULT_QUANTIZER_PREFIX,
    DEFAULT_SEMANTIC_ADAPTER_PREFIX, DEFAULT_SEMANTIC_ENCODER_PREFIX, NEUCODEC_FSQ_LEVELS,
    load_acoustic_encoder, load_fsq_quantizer, load_residual_fsq, load_semantic_adapter,
    load_semantic_encoder, load_semantic_encoder_with,
};
pub use resnet_block::{ResnetBlock, ResnetBlockWeights};
pub use semantic_adapter::{SemanticAdapter, SemanticAdapterWeights};
pub use semantic_encoder::{
    ConvolutionModule, ConvolutionModuleWeights, FFN_RESIDUAL_SCALE, FeatureProjection,
    FeatureProjectionWeights, SemanticEncoder, SemanticEncoderConfig, SemanticEncoderLayer,
    SemanticEncoderLayerWeights, SemanticEncoderWeights, SemanticFeedForward,
    SemanticSelfAttention, SemanticSelfAttentionWeights, causal_padding, relative_distance_indices,
};
pub use transformer_block::{TransformerBlock, TransformerBlockWeights};
