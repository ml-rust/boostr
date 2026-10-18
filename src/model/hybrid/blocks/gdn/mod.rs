//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), its
//! state-carrying inference forward (`forward`, prefill chain in
//! `prefill`, fused gate projections in `gate_proj`, tracked gate forward
//! in `gate_proj_var`), the tracked training forward (`train`), the
//! head-order permutation (`heads`), the CUDA graph-mode decode step
//! (`graph_mode`), the trainable switch (`trainable`), parameter access
//! (`params`), and unmerged LoRA adapters (`adapter`, gate pair in
//! `gate_adapter`).

mod adapter;
mod forward;
mod gate_adapter;
mod gate_proj;
mod gate_proj_var;
mod graph_mode;
mod heads;
mod layer;
mod params;
mod prefill;
mod train;
mod trainable;

pub use adapter::GdnProjection;
pub use layer::{GdnBlock, GdnWeights};
