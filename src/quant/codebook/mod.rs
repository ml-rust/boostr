//! Self-contained 4-bit block quantizer with a swappable reconstruction
//! codebook, at IDENTICAL geometry and byte cost across two independent
//! axes: group size 32, one `f32` scale per group (plus one `f32` minimum
//! for the affine arm).
//!
//! - SYMMETRIC (`d * level`): [`Codebook::Uniform`] (control) vs
//!   [`Codebook::Nf4`] — see `levels.rs`, `quantize.rs`, `roundtrip.rs`.
//! - AFFINE (`m + d * level`): [`AffineCodebook::Uniform`] (control) vs
//!   [`AffineCodebook::Nf4Shifted`] — see `affine.rs`.
//!
//! Together the four combinations isolate level PLACEMENT (uniform vs
//! non-uniform) from AFFINE-vs-symmetric geometry, the cell no GGUF format
//! occupies on its own (K-quants are affine+uniform, IQ formats are
//! non-uniform+symmetric).
//!
//! - TWO-LEVEL SUPER-SCALE, symmetric (6-bit codes, 16-element groups, one
//!   super-scale per 256): [`SuperPrecision::Bf16`] (TCF's `Q6S16D_T64`
//!   design) vs [`SuperPrecision::F16`] (Q6_K's storage design) vs
//!   [`SuperPrecision::F32`] (the ceiling) — see `two_level.rs`. A separate
//!   geometry from the two probes above; isolates super-scale STORAGE
//!   FORMAT rather than reconstruction-level shape.
//! - TWO-LEVEL SUPER-SCALE, asymmetric (4-bit unsigned codes, 32-element
//!   groups, one super-scale AND one super-minimum per 256): the same three
//!   [`SuperPrecision`] arms applied to `Q4AS32D_T64`'s geometry instead of
//!   `Q6S16D_T64`'s — see `two_level_asymmetric.rs`. Isolates whether the
//!   symmetric probe's storage-format finding generalizes to a design that
//!   stores TWO bf16 supers instead of one.
//!
//! Library code: plain slices, no [`numr::runtime::Runtime`], sibling of
//! `quant::smoothing`.

mod affine;
mod levels;
mod quantize;
mod roundtrip;
mod two_level;
mod two_level_asymmetric;

pub use affine::{
    AFFINE_UNIFORM_LEVELS, AffineCodebook, NF4_SHIFTED_LEVELS, affine_codebook_round_trip,
};
pub use levels::{Codebook, NF4_LEVELS, UNIFORM_LEVELS};
pub use roundtrip::codebook_round_trip;
pub use two_level::{SuperPrecision, two_level_codebook_round_trip};
pub use two_level_asymmetric::two_level_asymmetric_round_trip;
