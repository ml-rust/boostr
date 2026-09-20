//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), its
//! state-carrying inference forward (`forward`, prefill chain in
//! `prefill`), and the CUDA graph-mode decode step (`graph_mode`).

mod forward;
mod graph_mode;
mod layer;
mod prefill;

pub use layer::{GdnBlock, GdnWeights};
