//! Optimizer trait abstraction
//!
//! Defines a common interface for all optimizers so trainers can be optimizer-agnostic.

use crate::error::Result;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Trait for parameter optimizers.
///
/// All optimizers (AdamW, SGD, etc.) implement this trait so trainers
/// can work with any optimizer without hardcoding a specific one.
pub trait Optimizer<R: Runtime<DType = DType>> {
    /// Perform one optimization step.
    ///
    /// Updates all parameters in `params` using gradients from `grads`.
    /// Parameters without gradients are skipped.
    fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f64);

    /// Get the current learning rate.
    fn lr(&self) -> f64;

    /// Reset all optimizer state (moments, velocities, timestep).
    fn reset(&mut self);
}
