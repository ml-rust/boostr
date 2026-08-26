//! Weight loaders for the VoxCPM2 `AudioVAE` encoder and decoder from their
//! `encoder.*`/`decoder.*` SafeTensors checkpoint prefixes.

mod decoder;
mod encoder;

pub use decoder::DEFAULT_DECODER_PREFIX;
pub use encoder::DEFAULT_ENCODER_PREFIX;
