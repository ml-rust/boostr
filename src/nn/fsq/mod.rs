//! Finite Scalar Quantizer (FSQ).

pub mod codes;
pub mod config;
pub mod quantizer;
pub mod residual;

pub use config::{FsqConfig, ResidualFsqConfig};
pub use quantizer::Fsq;
pub use residual::{ResidualFsq, ResidualFsqWeights};
