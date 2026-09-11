//! Self-contained 4-bit block quantizer with a swappable reconstruction
//! codebook — [`Codebook::Uniform`] (control) vs [`Codebook::Nf4`] — at
//! IDENTICAL geometry and byte cost: symmetric, group size 32, one `f32`
//! scale per group. Isolates whether moving the 16 levels off a uniform
//! grid reduces task damage, before any TCF format work.
//!
//! Library code: plain slices, no [`numr::runtime::Runtime`], sibling of
//! `quant::smoothing`.

mod levels;
mod quantize;
mod roundtrip;

pub use levels::{Codebook, NF4_LEVELS, UNIFORM_LEVELS};
pub use roundtrip::codebook_round_trip;
