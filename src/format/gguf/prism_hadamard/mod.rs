//! PrismML activation-rotation ("Hadamard") contract, parsed from
//! `prism.hadamard.*` GGUF metadata.

pub mod block;
pub mod config;
pub mod signs;
pub mod weights;

pub use config::{PrismHadamardConfig, SignMode};
