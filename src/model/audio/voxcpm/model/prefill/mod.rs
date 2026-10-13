//! Prefill stage of the VoxCPM2 end-to-end orchestrator: reference and prompt
//! audio encoding and the deterministic two-LM prefill, for one row or a
//! left-padded batch.
//!
//! The per-patch sampling loop, the stop logic and the VAE decode path live
//! in `generate` and `decode`. `rows` documents the left padding. `run`
//! documents the pipeline and the traps this module exists to get right.
//!
//! # Layout
//!
//! - `state`: [`PrefillState`], [`PrefillIntermediates`], [`PrefillRow`]
//! - `rows`: [`PaddedBatch`], the host-side padded ids and masks
//! - `encode`: [`VoxCpm2Model::encode_reference`], [`VoxCpm2Model::encode_prompt`]
//! - `entry`: the prefill entry points
//! - `run`: the shared body
//! - `tensors`: the body's tensor helpers
//!
//! [`VoxCpm2Model::encode_reference`]: super::VoxCpm2Model::encode_reference
//! [`VoxCpm2Model::encode_prompt`]: super::VoxCpm2Model::encode_prompt

mod encode;
mod entry;
mod rows;
mod run;
mod state;
mod tensors;

pub use rows::PaddedBatch;
pub use state::{PrefillIntermediates, PrefillRow, PrefillState};
