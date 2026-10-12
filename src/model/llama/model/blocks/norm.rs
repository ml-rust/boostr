//! The norm of a Llama-path block: RMSNorm or LayerNorm.

use crate::error::{Error, Result};
use crate::model::config::{BlockConfig, NormKind};
use crate::model::traits::ModelClient;
use crate::nn::{LayerNorm, RmsNorm};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// RMSNorm (Llama, Gemma) or LayerNorm (GPT-NeoX, Falcon, StarCoder2,
/// Cohere, DBRX).
pub enum BlockNorm<R: Runtime> {
    Rms(RmsNorm<R>),
    Layer(LayerNorm<R>),
}

impl<R: Runtime<DType = DType>> BlockNorm<R> {
    /// The norm `block` asks for over `weight` and an optional `bias`.
    ///
    /// A LayerNorm without a bias tensor gets a zero bias. The weight is
    /// taken as stored: `restructure_hf_checkpoint` applies
    /// `block.norm_weight_offset` (Gemma) beforehand.
    ///
    /// # Errors
    ///
    /// A bias on an RMSNorm, or a tensor op error.
    pub fn build(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        eps: f32,
        block: &BlockConfig,
        trainable: bool,
    ) -> Result<Self> {
        Ok(match block.norm {
            NormKind::Rms => {
                if bias.is_some() {
                    return Err(Error::ModelError {
                        reason: "an RMSNorm takes no bias, but the checkpoint has one; set \
                                 block.norm to layer"
                            .into(),
                    });
                }
                Self::Rms(RmsNorm::new(weight, eps, trainable))
            }
            NormKind::Layer => {
                let bias = match bias {
                    Some(b) => b,
                    None => Tensor::<R>::zeros(weight.shape(), weight.dtype(), weight.device())?,
                };
                Self::Layer(LayerNorm::new(weight, bias, eps, trainable))
            }
        })
    }

    /// Normalize `x` over its last axis.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Rms(n) => n.forward(client, x),
            Self::Layer(n) => n.forward(client, x),
        }
    }

    /// `(norm(x + residual), x + residual)`, fused for RMSNorm. Bypasses
    /// autograd, like [`RmsNorm::fused_add_forward`].
    pub fn fused_add_forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        residual: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Rms(n) => n.fused_add_forward(client, x, residual),
            Self::Layer(n) => {
                let sum = var_add(x, residual, client).map_err(Error::Numr)?;
                let normed = n.forward(client, &sum)?;
                Ok((normed, sum))
            }
        }
    }

    /// [`fused_add_forward`](Self::fused_add_forward) that keeps autograd
    /// when either operand needs a gradient.
    pub fn residual_norm<C>(
        &self,
        client: &C,
        x: &Var<R>,
        delta: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Rms(n) => n.residual_norm(client, x, delta),
            Self::Layer(_) => self.fused_add_forward(client, x, delta),
        }
    }

    /// The weight, whose dtype is the block's compute dtype.
    pub fn weight(&self) -> &Var<R> {
        match self {
            Self::Rms(n) => n.weight(),
            Self::Layer(n) => n.weight(),
        }
    }
}
