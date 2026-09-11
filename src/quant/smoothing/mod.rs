//! Per-input-channel weight smoothing (AWQ-style): two scale sources — one
//! derived from an [`ImportanceMatrix`](crate::quant::ImportanceMatrix) entry
//! and the weight ([`scale::smoothing_scale`]), one calibration-free, derived
//! from the weight alone ([`weight_only::weight_only_smoothing_scale`]).
//! `normalize` holds the geometric-mean normalization and degenerate-channel
//! rule shared by both, so it lives in exactly one place.
//!
//! Encoder-side policy over plain slices, like `quant::imatrix` beside it —
//! no `Runtime`, no tensor op, no per-backend kernel.

mod normalize;
pub mod scale;
pub mod weight_only;

pub use scale::smoothing_scale;
pub use weight_only::weight_only_smoothing_scale;
