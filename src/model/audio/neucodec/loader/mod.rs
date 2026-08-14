//! Weight loading for the NeuCodec sub-models from a `neuphonic/neucodec`
//! SafeTensors checkpoint.
//!
//! Each entry point reads only its own prefix, so loading one branch never
//! materializes the others: `acoustic_decoder.*` (136 of the checkpoint's 811
//! tensors), `acoustic_encoder.*`, `quantizer.*`, `semantic_adapter.*`, and
//! `semantic_encoder.*` (the Wav2Vec2-BERT branch), and the two-tensor
//! `fc_encoder.*` prior projection. One file per branch, plus
//! [`support`] for the shape-checked tensor fetch they all share.

mod acoustic_encoder;
mod decoder;
mod fc_prior;
mod fsq;
mod semantic_adapter;
mod semantic_encoder;
mod support;

pub use acoustic_encoder::{DEFAULT_ACOUSTIC_ENCODER_PREFIX, load_acoustic_encoder};
pub use decoder::DEFAULT_DECODER_PREFIX;
pub use fc_prior::{DEFAULT_FC_PRIOR_PREFIX, load_fc_prior};
pub use fsq::{
    DEFAULT_QUANTIZER_PREFIX, NEUCODEC_FSQ_LEVELS, load_fsq_quantizer, load_residual_fsq,
};
pub use semantic_adapter::{DEFAULT_SEMANTIC_ADAPTER_PREFIX, load_semantic_adapter};
pub use semantic_encoder::{
    DEFAULT_SEMANTIC_ENCODER_PREFIX, load_semantic_encoder, load_semantic_encoder_with,
};
