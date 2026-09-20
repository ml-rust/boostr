//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), its
//! state-carrying inference forward (`forward`, prefill chain in
//! `prefill`, fused gate projections in `gate_proj`), and the CUDA
//! graph-mode decode step (`graph_mode`).

mod forward;
mod gate_proj;
mod graph_mode;
mod layer;
mod prefill;

pub use layer::{GdnBlock, GdnWeights};
