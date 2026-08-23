//! Optimizer trait abstraction
//!
//! Defines a common interface for all optimizers so trainers can be optimizer-agnostic.

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
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
    ///
    /// `TypeConversionOps` is required because an optimizer that reads a scalar
    /// back to the host (LAMB's trust ratio) must cast it to F32 first: reading
    /// a BF16/F16 scalar at a fixed f32 width over-runs the device buffer.
    fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + TypeConversionOps<R>
            + FusedOptimizerOps<R>;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f64);

    /// Get the current learning rate.
    fn lr(&self) -> f64;

    /// Reset all optimizer state (moments, velocities, timestep).
    fn reset(&mut self);
}
