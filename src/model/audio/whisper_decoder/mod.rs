//! Whisper decoder: causal transformer with cross-attention to encoder hidden states.
//!
//! Weight naming matches the HuggingFace `WhisperForConditionalGeneration` layout:
//! - `decoder.embed_tokens.weight`
//! - `decoder.embed_positions.weight`
//! - `decoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
//! - `decoder.layers.{i}.self_attn_layer_norm.{weight,bias}`
//! - `decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj.{weight,bias}`
//! - `decoder.layers.{i}.encoder_attn_layer_norm.{weight,bias}`
//! - `decoder.layers.{i}.{fc1,fc2}.{weight,bias}`
//! - `decoder.layers.{i}.final_layer_norm.{weight,bias}`
//! - `decoder.layer_norm.{weight,bias}`
//! - `proj_out.weight` is tied to `decoder.embed_tokens.weight`
//!
//! The decoder uses pre-norm everywhere. Self-attention is causal; cross-attention
//! has no mask (encoder outputs are fully visible).

mod cache;
mod decoder;
mod helpers;
mod layer;

pub use cache::{DecoderCache, DecoderLayerCache};
pub use decoder::WhisperDecoder;
pub use layer::WhisperDecoderLayer;
