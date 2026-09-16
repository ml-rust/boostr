//! Two-level super-scale probe, asymmetric variant: mirrors
//! [`super::two_level`]'s super-scale-storage probe on `Q4AS32D_T64`'s
//! geometry instead of `Q6S16D_T64`'s (both retired TCF native encodings) — 4-bit
//! UNSIGNED codes `0..=15`, 32-element groups, 8 groups per 256-element
//! super-block, a 6-bit sub-scale AND a 6-bit signed sub-minimum per group,
//! and ONE super-scale plus ONE super-minimum per super-block.
//!
//! Why this probe exists: `Q6S16D_T64`'s super-scale probe found bf16 worth
//! only a little over f16 at 6.5 bpw. `Q4AS32D_T64` — TCF's best 4-bit
//! encoding, at parity with GGUF's `q4_k` — stores the same bf16-super
//! design TWICE: a super-scale and a super-minimum. This probe isolates
//! whether f16 helps there too, holding every other structural knob (group
//! count, sub-level width, code range) fixed across the three
//! [`SuperPrecision`] arms, exactly as `two_level.rs` does for the
//! symmetric case.
//!
//! [`SuperPrecision::Bf16Reserved`] is not applicable: it retired a reserved
//! CODE on `two_level.rs`'s SIGNED 6-bit grid, and this geometry's codes are
//! UNSIGNED with no reserved pattern. [`two_level_asymmetric_round_trip`]
//! refuses it outright rather than silently mapping it onto [`Bf16`].
//!
//! Mirrors the retired TCF native quantizer's asymmetric super-block
//! search: same three-pass shape (fit every group's `f32` pair, derive the
//! super pair, round and refine each group's sub-levels against it), same
//! weighted least-squares refit closing each candidate, same `±2 x ±2`
//! sub-level search. It differs only where `two_level.rs` already differs:
//! no binary16 rounding of the per-group intermediate (kept exact `f32`),
//! and a local candidate-multiplier sweep in place of that quantizer's
//! search-effort ladder — see `fit::asymmetric_candidate_multipliers`.
//!
//! Split into `fit` (per-group pair fitting), `refine` (sub-level
//! refinement and reconstruction) and `block` (super-block quantization,
//! the public entry point, and its tests) to stay under this repo's
//! 500-line file limit.
//!
//! [`Bf16`]: super::two_level::SuperPrecision::Bf16

mod block;
mod fit;
mod refine;

pub use block::two_level_asymmetric_round_trip;
