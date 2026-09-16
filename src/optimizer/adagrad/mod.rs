//! AdaGrad optimizer
//!
//! Adaptive gradient algorithm (Duchi et al., 2011). Adapts learning rates
//! per-parameter based on accumulated squared gradients. Particularly effective
//! for sparse gradients (e.g., embedding layers).

mod step;
mod types;

pub use types::{AdaGrad, AdaGradConfig};
