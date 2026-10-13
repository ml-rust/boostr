//! Tensor-parallel LLaMA model (Megatron-LM style)
//!
//! Separate from `Llama` to keep zero overhead for non-TP users.
//! Uses `ColumnParallelLinear` for Q/K/V/gate/up projections and
//! `RowParallelLinear` for O/down projections (2 all-reduces per block).

mod build;
mod llama_tp;

pub use llama_tp::LlamaTp;
