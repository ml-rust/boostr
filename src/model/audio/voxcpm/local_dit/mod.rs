//! VoxCPM2 `feat_decoder` local DiT ("locdit"): the CFM estimator backbone,
//! reusing the shared bidirectional MiniCPM4 stack in
//! `crate::model::audio::voxcpm::bidirectional`. This module loads weights
//! only — the estimator forward pass and CFM sampler are separate units.

pub mod config;
pub mod loader;

pub use config::LocalDitConfig;
pub use loader::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit};
