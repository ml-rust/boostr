//! The `qwen35` model.
//!
//! - `build`: the [`Qwen35Model`] type, its layer structs, and
//!   [`Qwen35Model::new`]
//! - `forward`: the KV-cache + GDN-state forward, the hidden-state forward,
//!   and accessors
//! - `gguf` + `gguf_layers`: [`Qwen35Model::from_varbuilder`] over a
//!   GGUF-filled `VarMap`

mod build;
mod forward;
mod gguf;
mod gguf_layers;

pub use build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};

#[cfg(test)]
pub(crate) use build::tests::tiny_model;
