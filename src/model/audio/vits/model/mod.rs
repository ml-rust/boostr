//! `VitsModel`: exact port of HF `VitsModel.forward` (inference path only).
//! See `core` for construction and `synthesize` for the forward pass.

mod client;
mod core;
#[cfg(test)]
mod fixtures;
mod options;
mod synthesize;

pub use client::VitsModelClient;
pub use core::VitsModel;
pub use options::{VitsSynthOptions, VitsSynthesis};
