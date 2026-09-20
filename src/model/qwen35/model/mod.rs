//! The `qwen35` model.
//!
//! - `build`: the [`Qwen35Model`] type, its layer structs, and
//!   [`Qwen35Model::new`]
//! - `forward`: the KV-cache + GDN-state token forward, the hidden-state
//!   forward, and accessors
//! - `forward_embeds`: the same forward over pre-built input embeddings
//!   with explicit IMROPE positions, plus `embed_tokens`
//! - `forward_graph`: the CUDA graph-mode decode forward
//! - `gguf` + `gguf_layers`: [`Qwen35Model::from_varbuilder`] over a
//!   GGUF-filled `VarMap`

mod build;
mod forward;
mod forward_embeds;
mod forward_graph;
mod gguf;
mod gguf_layers;

pub use build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};

#[cfg(test)]
pub(crate) use build::tests::tiny_model;
