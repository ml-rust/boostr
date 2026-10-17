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

    /// Export the optimizer state, keyed by parameter name.
    ///
    /// `names` maps each parameter id to its name. Each key is `<name>.<field>`.
    /// AdamW and Lamb also write `__step`, the global step.
    /// A parameter with no state writes no key.
    /// Tensors keep their dtype and are deep copies on the optimizer's device.
    ///
    /// # Errors
    /// Returns an error when a parameter with state has no entry in `names`,
    /// or when two parameters share one name.
    fn state_dict(&self, names: &HashMap<TensorId, String>) -> Result<HashMap<String, Tensor<R>>>;

    /// Replace the optimizer state with `state`, as written by [`Self::state_dict`].
    ///
    /// `ids` maps each parameter name to its id. The load checks every key
    /// before it modifies the optimizer. After an error, the optimizer is unchanged.
    /// The first step after a load continues from the loaded state.
    ///
    /// # Errors
    /// Returns an error that names the key when:
    /// - a key names a parameter that is not in `ids`, or has an unknown field
    /// - a shape or dtype does not match the other tensors of that parameter
    /// - a parameter has some of its required keys but not all
    /// - `__step` is missing while per-parameter keys are present
    fn load_state_dict(
        &mut self,
        ids: &HashMap<String, TensorId>,
        state: &HashMap<String, Tensor<R>>,
    ) -> Result<()>;
}
