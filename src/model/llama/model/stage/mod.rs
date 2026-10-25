//! A Llama pipeline stage that holds the weights of one layer range only.
//!
//! - `span`: `LayerSpan`, the range and the checkpoint tensors it reads
//! - `model`: `LlamaStage`, built from those tensors
//! - `forward`: the stage forwards: embed, held layers, head

mod forward;
mod model;
mod span;

pub use model::LlamaStage;
pub use span::{LayerSpan, layer_range_refusal};
