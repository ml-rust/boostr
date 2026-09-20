//! Linear and quantized linear layers

pub mod dense;
pub mod maybe_quant_linear;
pub mod maybe_rotated;
pub mod quant_linear;
pub mod regroup_rows;
pub mod rotated_linear;

pub use dense::Linear;
pub use maybe_quant_linear::MaybeQuantLinear;
pub use maybe_rotated::MaybeRotatedLinear;
pub use quant_linear::QuantLinear;
pub use regroup_rows::regroup_leading_axis;
pub use rotated_linear::RotatedLinear;
