//! Conditional flow-matching (CFM) sampler around VoxCPM2's local DiT
//! estimator ([`LocalDit::forward`]).
//!
//! Reference: `voxcpm/modules/locdit/unified_cfm.py`. The estimator is a
//! velocity field; this module integrates it backwards from `t = 1` to `t = 0`
//! with an explicit Euler scheme, classifier-free guidance (CFG), and the
//! "CFG-zero-star" rescale. `euler` documents the loop and the traps it is
//! pinned against. `schedule` documents where the time schedule is computed.
//!
//! - `schedule`: [`CfmOptions`], the sway-corrected time schedule, and the
//!   per-step `(t, dt)` plan both integrators share
//! - `guidance`: the CFG-zero-star rescale and the guidance combine
//! - `euler`: `LocalDit::solve_euler` (eager) and `LocalDit::sample`
//! - `graphed`: `LocalDit::solve_euler_graphed`, the inference entry that
//!   replays the loop as one CUDA graph per patch and falls back to eager
//! - `graph` (`cuda` only): the captured-graph cache and the capture/replay
//!
//! [`LocalDit::forward`]: crate::model::audio::voxcpm::local_dit::LocalDit::forward

mod euler;
#[cfg(feature = "cuda")]
pub(crate) mod graph;
mod graphed;
mod guidance;
mod schedule;

pub use schedule::{CfmOptions, cfm_time_span};
