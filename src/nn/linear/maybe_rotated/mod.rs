//! [`MaybeRotatedLinear`]: a projection that is plain, Hadamard-rotated, or
//! LoRA-adapted, its adapter attachment, and its batched forward over
//! projections sharing one input.

pub mod adapter;
pub mod batch;
pub mod layer;

pub use layer::MaybeRotatedLinear;
