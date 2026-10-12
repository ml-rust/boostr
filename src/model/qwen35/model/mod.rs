//! The `qwen35` model: its type and layers in `build`, the forwards in
//! `forward*` (`forward_gather` returns normed hidden rows at chosen
//! positions, `head_rows` F32 logits of chosen `lm_head` rows), the
//! decision backbone over one shared prefix in `decision_backbone`, the
//! loader in `gguf*` and `attach`, and the unmerged LoRA attach in
//! `adapter*`.

mod adapter;
mod adapter_attach;
mod adapter_targets;
mod attach;
mod build;
mod decision_backbone;
mod forward;
mod forward_batched;
mod forward_embeds;
mod forward_gather;
mod forward_graph;
mod forward_layers;
mod gguf;
mod gguf_layers;
mod head_rows;

pub use build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};

#[cfg(test)]
pub(crate) use adapter::tests::{ALL_MODULES, factors as adapter_entries, write_adapter};
#[cfg(test)]
pub(crate) use build::tests::{Head, chunked_model, tiny_config, tiny_model};
