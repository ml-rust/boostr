//! Initialization strategies for new tensors.
//!
//! - `strategy`: the [`Init`] enum and PyTorch's `fan_in` convention
//! - `unseeded`: `Init::init_tensor`
//! - `seeded`: `Init::init_tensor_seeded`

mod seeded;
mod strategy;
mod unseeded;

pub use strategy::Init;
