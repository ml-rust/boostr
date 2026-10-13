//! Two-level super-scale probe, asymmetric variant: mirrors
//! [`super::two_level`]'s super-scale-storage probe on `Q4AS32D_T64`'s
//! geometry instead of `Q6S16D_T64`'s (both retired TCF native encodings) — 4-bit
//! UNSIGNED codes `0..=15`, 32-element groups, 8 groups per 256-element
//! super-block, a 6-bit sub-scale AND a 6-bit signed sub-minimum per group,
//! and ONE super-scale plus ONE super-minimum per super-block. `block`
//! documents why the probe exists and what it mirrors.
//!
//! The same three [`SuperPrecision`](super::SuperPrecision) arms as the
//! symmetric probe. Isolates whether the symmetric probe's storage-format
//! finding generalizes to a design that stores TWO bf16 supers instead of one.
//!
//! `fit` holds per-group pair fitting. `refine` holds sub-level
//! refinement and reconstruction. `block` holds super-block quantization,
//! the public entry point, and its tests. Three distinct passes of the
//! same search.

mod block;
mod fit;
mod refine;

pub use block::two_level_asymmetric_round_trip;
