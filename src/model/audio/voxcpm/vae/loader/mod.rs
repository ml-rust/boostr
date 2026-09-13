//! Weight loaders for the VoxCPM2 `AudioVAE` encoder and decoder from their
//! `encoder.*`/`decoder.*` checkpoint prefixes, in either container the
//! separate `AudioVAE` ships in — see [`VaeCheckpoint`] — or from the
//! `vae.encoder.*`/`vae.decoder.*` tensors compressr embeds in a VoxCPM2
//! GGUF/TCF, through the `from_source` constructors.

mod checkpoint;
mod decoder;
mod encoder;

pub use checkpoint::VaeCheckpoint;
pub use decoder::{
    DEFAULT_DECODER_PREFIX, VAE_GGUF_DECODER_PREFIX, VAE_GGUF_PROBE_TENSOR, VAE_GGUF_ROOT,
};
pub use encoder::{DEFAULT_ENCODER_PREFIX, VAE_GGUF_ENCODER_PREFIX};
