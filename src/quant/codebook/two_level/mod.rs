//! Two-level super-scale probe: isolates ONE variable — how the per-256
//! super-scale is stored and whether it is pre-divided — from TCF's
//! `Q6S16D_T64` geometry (16-element groups, 8-bit sub-scale, one
//! super-scale per 256 elements, symmetric 6-bit codes). Everything else
//! (group count, sub-scale width, code range) is held fixed across the
//! three [`SuperPrecision`] arms so a difference in reconstruction error
//! traces to the super-scale storage alone, never to geometry.
//!
//! Why this probe exists: `Q6S16D_T64` measurably loses to GGUF's `q6_k` at
//! the same 6.5 bpw, and every OTHER structural knob (group size, sub-scale
//! width) already matches. The one difference left is how the super-scale
//! is stored — TCF: bfloat16, pre-divided by the sub-scale range (255) so a
//! group decodes in one multiply; Q6_K: f16, NOT pre-divided, with an int8
//! sub-scale. bf16 has 8 mantissa bits, f16 has 11 — trading exponent range
//! for precision. [`SuperPrecision::F32`] is the ceiling neither storage
//! format can beat: no rounding at all above the per-group `f32` fit.
//!
//! Reuses [`super::quantize::candidate_multipliers`] and
//! [`super::quantize::nearest_level`]'s tie rule (first-on-tie, ascending)
//! unchanged — this arm differs from the codebook probes ONLY in level
//! shape (uniform 6-bit integers, not a 16-entry table) and in adding a
//! second scale tier.
//!
//! Split into `fit` (per-group scale fitting) and `block` (super-block
//! quantization, the public entry point, and its tests) to stay under this
//! repo's 500-line file limit.

mod block;
mod fit;

pub use block::{SuperPrecision, two_level_codebook_round_trip};
pub(in crate::quant::codebook) use block::{round_bf16, round_f16};
