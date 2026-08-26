//! VoxCPM2's `fsq_layer` finite-scalar-quantization bottleneck and the six
//! auxiliary projections around it (loading only — see [`layer`] for why the
//! training branch is absent, and [`loader`] for the checkpoint key layout).

pub mod config;
pub mod layer;
pub mod loader;

pub use config::FsqConfig;
pub use layer::{AuxProjections, ScalarQuantization};
