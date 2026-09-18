//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`) and its
//! state-carrying inference forward (`forward`).

mod forward;
mod layer;

pub use layer::{GdnBlock, GdnWeights};
