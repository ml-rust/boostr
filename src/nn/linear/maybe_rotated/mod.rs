//! [`MaybeRotatedLinear`]: a projection that is plain, Hadamard-rotated, or
//! LoRA-adapted, its adapter attachment, its batched forward over
//! projections sharing one input, its F32 outputs at chosen weight rows,
//! and its chosen effective weight rows in F32.

pub mod adapter;
pub mod batch;
pub mod layer;
pub(crate) mod row_checks;
pub mod rows;
pub mod weight_rows;

pub use layer::MaybeRotatedLinear;
