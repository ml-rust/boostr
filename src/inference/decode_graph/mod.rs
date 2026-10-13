//! CUDA graph decode loop infrastructure.
//!
//! # Overview
//!
//! The decode loop for autoregressive generation makes ~930 individual kernel
//! launches per token. CUDA graphs replace them with a single `cuGraphLaunch`,
//! cutting per-token launch overhead from per-kernel dispatch to one graph
//! replay.
//!
//! `cuda_impl::decode_graph` documents the capture constraints, the
//! stable-address tensors and the stream ordering. `cuda_impl::qwen35_decode_graph`
//! documents the `qwen35` graph's extra inputs and its sampling split.

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl;
