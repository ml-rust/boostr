//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), its
//! state-carrying inference forward (`forward`, prefill chain in
//! `prefill`, fused gate projections in `gate_proj`), the CUDA
//! graph-mode decode step (`graph_mode`), and unmerged LoRA adapters
//! (`adapter`, gate pair in `gate_adapter`).

mod adapter;
mod forward;
mod gate_adapter;
mod gate_proj;
mod graph_mode;
mod layer;
mod prefill;

pub use adapter::GdnProjection;
pub use layer::{GdnBlock, GdnWeights};
