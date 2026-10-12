//! The `qwen35` model.
//!
//! - `build`: the [`Qwen35Model`] type, its layer structs, and
//!   [`Qwen35Model::new`]
//! - `forward`: the KV-cache + GDN-state token forward, the hidden-state
//!   forward, and accessors
//! - `forward_embeds`: the same forward over pre-built input embeddings
//!   with explicit IMROPE positions, plus `embed_tokens`
//! - `forward_batched`: one decode step for a batch of sequences, a lane
//!   each, with per-lane positions and per-lane key ranges
//! - `forward_graph`: the CUDA graph-mode decode forward
//! - `gguf` + `gguf_layers`: [`Qwen35Model::from_varbuilder`] over a
//!   `VarMap` filled from a GGUF or a restructured Hugging Face checkpoint
//! - `attach`: the Hadamard rotation attach the loader applies
//! - `adapter`, `adapter_attach` + `adapter_targets`:
//!   [`Qwen35Model::from_varbuilder_with_adapter`], a PEFT LoRA adapter
//!   attached unmerged at load

mod adapter;
mod adapter_attach;
mod adapter_targets;
mod attach;
mod build;
mod forward;
mod forward_batched;
mod forward_embeds;
mod forward_graph;
mod gguf;
mod gguf_layers;

pub use build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};

#[cfg(test)]
pub(crate) use adapter::tests::{ALL_MODULES, factors as adapter_entries, write_adapter};
#[cfg(test)]
pub(crate) use build::tests::{tiny_config, tiny_model};
