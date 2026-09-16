//! SGD optimizer with momentum
//!
//! Implements stochastic gradient descent with optional momentum and weight decay.
//! Follows PyTorch's SGD semantics with Nesterov momentum support.

mod apply;
mod step;
mod types;

pub use types::{Sgd, SgdConfig};
