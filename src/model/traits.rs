//! Core model trait

use crate::error::Result;
use crate::model::config::ModelConfig;
use numr::autograd::Var;
use numr::ops::{IndexingOps, ReduceOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Trait alias for the full set of client bounds required by model forward passes.
pub trait ModelClient<R: Runtime>:
    RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>
{
}

impl<R, C> ModelClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
{
}

/// Core trait for all model architectures.
///
/// Models take token IDs and produce logits. All internal computation
/// uses `Var<R>` for autograd compatibility.
pub trait Model<R: Runtime>: Sized {
    /// Create model from configuration with zero-initialized weights.
    fn from_config(config: &ModelConfig, device: &R::Device) -> Result<Self>;

    /// Forward pass: token_ids [B, S] -> logits [B, S, vocab_size]
    fn forward<C>(&self, client: &C, input_ids: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>;

    /// Get the model configuration
    fn config(&self) -> &ModelConfig;
}
