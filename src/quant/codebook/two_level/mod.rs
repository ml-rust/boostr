//! Two-level super-scale probe: isolates ONE variable — how the per-256
//! super-scale is stored and whether it is pre-divided — from TCF's
//! `Q6S16D_T64` geometry (16-element groups, 8-bit sub-scale, one
//! super-scale per 256 elements, symmetric 6-bit codes). Everything else
//! (group count, sub-scale width, code range) is held fixed across the
//! three [`SuperPrecision`] arms so a difference in reconstruction error
//! traces to the super-scale storage alone, never to geometry. `block`
//! documents why the probe exists and what it reuses.
//!
//! Arms: [`SuperPrecision::Bf16`] (TCF's `Q6S16D_T64` design) vs
//! [`SuperPrecision::F16`] (Q6_K's storage design) vs [`SuperPrecision::F32`]
//! (the ceiling). A separate geometry from the uniform/NF4 codebook probes;
//! isolates super-scale STORAGE FORMAT rather than reconstruction-level shape.
//!
//! `fit` holds per-group scale fitting. `block` holds super-block
//! quantization, the public entry point, and its tests. Two distinct
//! concerns: scale search versus the quantize/round-trip driver.

mod block;
mod fit;

pub use block::{SuperPrecision, two_level_codebook_round_trip};
pub(in crate::quant::codebook) use block::{round_bf16, round_f16};
