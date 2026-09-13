//! CUDA graph replay of the Euler loop: the per-`LocalDit` cache (`cache`)
//! and the concrete `CudaRuntime` capture/replay (`capture`). The generic
//! entry that decides graph vs eager is `super::graphed`.

mod cache;
mod capture;

pub use cache::{EulerGraphCache, EulerGraphKey};
pub(super) use capture::solve_euler_cuda;
