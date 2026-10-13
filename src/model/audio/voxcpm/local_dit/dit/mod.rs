//! Estimator forward pass for VoxCPM2's `feat_decoder` local DiT ("locdit").
//!
//! This is the CFM *estimator* only — it evaluates the DiT once for a given
//! `(x, mu, t, cond, dt)`. Sampling (noise, Euler stepping, classifier-free
//! guidance) is a separate unit and lives nowhere in this file.
//!
//! `estimator` documents the reference this pass ports and the traps it is
//! pinned against.

mod estimator;
mod forward;
mod validate;
