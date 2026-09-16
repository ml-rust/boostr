//! Gradient clipping utilities
//!
//! Clip gradients by global norm to prevent exploding gradients during training.

mod norm;
mod value;

#[cfg(test)]
mod test_support;

pub use norm::{clip_grad_norm, clip_grad_norm_per_param};
pub use value::clip_grad_value;
