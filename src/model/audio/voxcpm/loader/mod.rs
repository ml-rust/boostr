//! Weight loading for the VoxCPM2 `AudioVAE` decoder/encoder from their
//! `decoder.*`/`encoder.*` SafeTensors checkpoint prefixes.

mod decoder;
mod encoder;
mod support;

pub use decoder::DEFAULT_DECODER_PREFIX;
pub use encoder::DEFAULT_ENCODER_PREFIX;
