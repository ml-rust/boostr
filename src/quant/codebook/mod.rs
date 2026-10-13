//! Self-contained 4-bit block quantizer with a swappable reconstruction
//! codebook, at IDENTICAL geometry and byte cost across two independent
//! axes: group size 32, one `f32` scale per group (plus one `f32` minimum
//! for the affine arm).
//!
//! - SYMMETRIC (`d * level`): [`Codebook::Uniform`] (control) vs
//!   [`Codebook::Nf4`] — see `levels.rs`, `quantize.rs`, `roundtrip.rs`.
//! - AFFINE (`m + d * level`): [`AffineCodebook::Uniform`] (control) vs
//!   [`AffineCodebook::Nf4Shifted`] — see `affine.rs`, which also documents
//!   what the four combinations isolate.
//! - TWO-LEVEL SUPER-SCALE, symmetric and asymmetric — see `two_level` and
//!   `two_level_asymmetric`, which document their arms and geometry.
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
