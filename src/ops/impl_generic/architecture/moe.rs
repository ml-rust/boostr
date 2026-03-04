//! Generic MoE implementation — split across focused submodules.
//!
//! THE algorithm — same for all backends.

#[path = "moe/dispatch.rs"]
pub mod dispatch;
#[path = "moe/grouped_gemm.rs"]
pub mod grouped_gemm;
#[path = "moe/routing.rs"]
pub mod routing;

pub use dispatch::*;
pub use grouped_gemm::*;
pub use routing::*;
