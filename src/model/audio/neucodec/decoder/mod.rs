//! `NeuCodecDecoder` — the NeuCodec acoustic decoder top-level assembly.
//!
//! Architecture (VERIFIED from `acoustic_decoder.*` in the real
//! `neuphonic/neucodec` `model.safetensors` — NOT `config.json` or the
//! GitHub source, both of which are partly wrong about this decoder):
//!
//! ```text
//! input [B, T, 2048]                      (FSQ project_out features)
//!   -> fc            Linear[1024, 2048]+bias          -> [B, T, 1024]
//!   -> embed         Conv1d(1024->1024, k=7)+bias      -> [B, 1024, T]  (channels-first)
//!   -> prior_net     2x ResnetBlock                    -> [B, 1024, T]
//!   -> (permute)                                       -> [B, T, 1024]  (channels-last)
//!   -> 12x TransformerBlock (RMSNorm/attn/RMSNorm/MLP) -> [B, T, 1024]
//!   -> (permute)                                       -> [B, 1024, T]
//!   -> post_net      2x ResnetBlock                    -> [B, 1024, T]
//!   -> (permute)                                       -> [B, T, 1024]
//!   -> norm          LayerNorm(eps=1e-6, weight+bias)  -> [B, T, 1024]
//!   -> head.linear   Linear[1922, 1024]+bias           -> [B, T, 1922]
//!   -> split/activate -> (mag [B, 961, T], phase [B, 961, T])
//!   -> istft (n_fft=1920, hop=480)                     -> waveform [B, samples]
//! ```
//!
//! `samples == T * hop_length` under Vocos `padding="same"` framing — one hop
//! per latent frame (see [`NeuCodecDecoder::forward`] tests for the exact
//! derivation).
//!
//! This module is architecture-only: no weight loading. Construct with
//! synthetic weights via [`NeuCodecDecoderWeights`]; a loader is a separate
//! unit.

mod core;
mod forward;
mod model;
#[cfg(test)]
mod test_support;

pub use model::{NeuCodecDecoder, NeuCodecDecoderWeights};
