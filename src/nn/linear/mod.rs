//! Linear and quantized linear layers

pub mod dense;
pub mod maybe_quant_linear;
pub mod quant_linear;

#[cfg(test)]
mod tests;

pub use dense::Linear;
pub use maybe_quant_linear::MaybeQuantLinear;
pub use quant_linear::QuantLinear;
