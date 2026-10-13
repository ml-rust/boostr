//! Regularization layers: dropout and stochastic depth.

pub mod dropout;
pub mod stochastic_depth;

pub use dropout::Dropout;
pub use stochastic_depth::StochasticDepth;
