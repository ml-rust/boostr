//! Weight loaders for the VoxCPM2 `AudioVAE` encoder and decoder from their
//! `encoder.*`/`decoder.*` checkpoint prefixes, in either container the
//! `AudioVAE` ships in — see [`VaeCheckpoint`].

mod checkpoint;
mod decoder;
mod encoder;

pub use checkpoint::VaeCheckpoint;
pub use decoder::DEFAULT_DECODER_PREFIX;
pub use encoder::DEFAULT_ENCODER_PREFIX;
