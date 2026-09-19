//! Activation-rotation ("Hadamard") contract, parsed from
//! the `prism.hadamard.*` GGUF metadata keys.

pub mod block;
pub mod config;
pub mod signs;
pub mod weights;

pub use config::{HadamardContract, SignMode};
