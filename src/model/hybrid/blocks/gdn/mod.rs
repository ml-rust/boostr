//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), its
//! state-carrying inference forward (`forward`), and the CUDA graph-mode
//! decode step (`graph_mode`).

mod forward;
mod graph_mode;
mod layer;

pub use layer::{GdnBlock, GdnWeights};
