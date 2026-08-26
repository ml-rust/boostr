//! VoxCPM2 `feat_decoder` local DiT ("locdit"): the CFM estimator backbone,
//! reusing the shared bidirectional MiniCPM4 stack in
//! `crate::model::audio::voxcpm::bidirectional`. This module loads weights.
//! Three units live here: weight loading (`loader`), the estimator forward
//! pass (`dit`), and the CFM sampler (`sampler`) that integrates it.

pub mod config;
pub mod dit;
pub mod loader;
pub mod sampler;

#[cfg(test)]
pub(crate) mod tests;

pub use config::LocalDitConfig;
pub use loader::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit};
pub use sampler::{CfmOptions, cfm_time_span};
