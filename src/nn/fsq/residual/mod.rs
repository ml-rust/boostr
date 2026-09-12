//! Residual Finite Scalar Quantizer — see `layer` (the type) and `codec` (encode/decode).

mod codec;
mod layer;

pub use layer::{ResidualFsq, ResidualFsqWeights};
