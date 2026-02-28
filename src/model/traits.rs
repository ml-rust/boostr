//! Core model trait

use crate::error::Result;
use crate::model::config::ModelConfig;
use crate::nn::VarBuilder;
use crate::ops::traits::{FlashAttentionOps, RoPEOps};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

/// Trait alias for the full set of client bounds required by model forward passes.
pub trait ModelClient<R: Runtime>:
    RuntimeClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + ReduceOps<R>
    + IndexingOps<R>
    + ShapeOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + UnaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + RoPEOps<R>
    + FlashAttentionOps<R>
    + QuantMatmulOps<R>
    + NormalizationOps<R>
{
}

impl<R, C> ModelClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + RoPEOps<R>
        + FlashAttentionOps<R>
        + QuantMatmulOps<R>
        + NormalizationOps<R>,
{
}

/// Core trait for all model architectures.
///
/// Models take token IDs and produce logits. All internal computation
/// uses `Var<R>` for autograd compatibility.
pub trait Model<R: Runtime>: Sized {
    /// Create model from configuration with zero-initialized weights.
    fn from_config(config: &ModelConfig, device: &R::Device) -> Result<Self>;

    /// Create model from a VarBuilder (loads real weights).
    ///
    /// Takes `&mut` because tensors are moved out of the VarMap (zero-copy).
    fn from_varbuilder(vb: &mut VarBuilder<R>, config: &ModelConfig) -> Result<Self>
    where
        R::Client: DequantOps<R>,
    {
        let _ = (vb, config);
        Err(crate::error::Error::ModelError {
            reason: "from_varbuilder not implemented for this model".into(),
        })
    }

    /// Forward pass: token_ids [B, S] -> logits [B, S, vocab_size]
    fn forward<C>(&self, client: &C, input_ids: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>;

    /// Get the model configuration
    fn config(&self) -> &ModelConfig;
}
