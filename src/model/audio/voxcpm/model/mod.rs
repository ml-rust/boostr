//! VoxCPM2 end-to-end orchestrator: [`config`] (patch geometry and special token ids),
//! [`loader`] ([`VoxCpm2Model`], every sub-model plus its checkpoint loader), [`gguf_loader`]
//! (the single-file GGUF entry point), [`tcf_loader`] (the single-file TCF entry point),
//! [`sequence`] (prefix layout and mask complementarity),
//! [`patches`] (wav padding and the VAE patch fold), [`prefill`] (reference and prompt
//! encode and the two-LM prefill, one row or a left-padded batch), [`generate`] (the per-patch sampling
//! loop and its stop logic), [`decode`]
//! (unfolding patches back to a latent and VAE-decoding to a waveform), `chunked_decode`
//! (windowed VAE decode so peak memory does not scale with utterance length), [`train`] (the
//! CFM training loss, wiring teacher-forced conditioning into a differentiable `Var`),
//! [`widths`] (the row counts a generation loop is allowed to run at).
//! The wav file wrapper is a separate, later unit and does not live here.

pub(crate) mod chunked_decode;
pub mod config;
pub mod decode;
pub mod generate;
pub mod gguf_loader;
pub mod loader;
pub mod patches;
pub mod prefill;
mod reexports;
pub mod sequence;
pub mod tcf_loader;
pub mod train;
pub mod widths;

pub use reexports::*;
