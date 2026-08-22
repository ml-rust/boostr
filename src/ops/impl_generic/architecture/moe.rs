//! Generic MoE implementation — split across focused submodules.
//!
//! THE algorithm — same for all backends.

pub mod dispatch;
pub mod grouped_gemm;
pub mod routing;

pub use dispatch::*;
pub use grouped_gemm::*;
pub use routing::*;
