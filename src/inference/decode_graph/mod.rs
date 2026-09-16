//! CUDA graph decode loop infrastructure.
//!
//! # Overview
//!
//! The decode loop for autoregressive generation makes ~930 individual kernel
//! launches per token. CUDA graphs replace them with a single `cuGraphLaunch`
//! (~5µs overhead instead of ~13ms).
//!
//! ## Constraints
//!
//! CUDA graphs freeze kernel arguments at capture time. We work around
//! position-dependent values by reading them from **device memory** inside kernels.
//! Before each graph replay, the CPU updates a small set of device scalars via
//! async H2D copies on the same stream — no CPU–GPU sync required.
//!
//! ## Stable-address tensors
//!
//! All tensors that are read or written from outside the graph MUST be allocated
//! BEFORE `Runtime::capture_graph_into()` is called (before `cuStreamBeginCapture`).
//! Any tensor allocated INSIDE the capture region has a graph-managed address
//! that is only valid within the graph's execution — accessing it from the CPU
//! after the graph has run causes `CUDA_ERROR_ILLEGAL_ADDRESS`.
//!
//! The output of each decode step is therefore written into a pre-allocated
//! `next_token_buf` via a `cuMemcpyAsync` node captured inside the graph.
//! CUDA automatically patches the (graph-internal) source address when replaying;
//! the stable destination address never changes.
//!
//! ## Stream ordering
//!
//! ALL pre-launch copies MUST use stream-ordered async variants so they are
//! serialized on the compute stream before `cuGraphLaunch` executes:
//!
//! - D2D copies: `cuMemcpyDtoDAsync_v2(dst, src, bytes, stream)`
//! - H2D copies: `cuMemcpyHtoDAsync_v2(dst, src, bytes, stream)`
//!
//! `cuMemcpy` (synchronous context-ordered) is NOT serialized with respect to
//! any stream. Using it before `cuGraphLaunch` (which is stream-ordered) creates
//! a race: the graph's kernels can start before the copies finish, causing
//! `CUDA_ERROR_ILLEGAL_ADDRESS` when kernels dereference stale/invalid pointers.
//!
//! | Tensor             | Allocated   | Updated how                          |
//! |--------------------|-------------|--------------------------------------|
//! | `token_buf`        | pre-capture | D2D async (DtoDAsync) from prev step |
//! | `device_scalars`   | pre-capture | H2D async (HtoDAsync) from CPU       |
//! | `cos_slice`        | pre-capture | D2D async (DtoDAsync) from rope cache|
//! | `sin_slice`        | pre-capture | D2D async (DtoDAsync) from rope cache|
//! | `next_token_buf`   | pre-capture | written by graph (argmax→memcpy node)|

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl;
