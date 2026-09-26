//! End-to-end Whisper model: encoder + decoder with greedy generation.
//!
//! Weight layout matches HuggingFace `WhisperForConditionalGeneration`:
//! - `model.encoder.*` → `WhisperEncoder`
//! - `model.decoder.*` → `WhisperDecoder`
//! - `proj_out.weight` is tied to `model.decoder.embed_tokens.weight`
//!   (the decoder already takes care of this via its `tied_out_weight`).

mod generate;
mod language;
mod mask;
mod model;
#[cfg(test)]
mod test_support;

pub use generate::{CrossAttentionWeights, GenerateOptions};
pub use model::WhisperModel;
