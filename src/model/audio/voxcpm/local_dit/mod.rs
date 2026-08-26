//! VoxCPM2 `feat_decoder` local DiT ("locdit"): the CFM estimator backbone,
//! reusing the shared bidirectional MiniCPM4 stack in
//! `crate::model::audio::voxcpm::bidirectional`. This module loads weights.
//! Two units live here: weight loading (`loader`) and the estimator forward
//! pass (`dit`). The CFM sampler is separate and is NOT in this module.

pub mod config;
pub mod dit;
pub mod loader;

#[cfg(test)]
mod tests;

pub use config::LocalDitConfig;
pub use loader::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit};
