//! Multimodal projector: maps vision encoder output to LLM embedding space.

use crate::error::{Error, Result};
use crate::model::config::VisionConfig;
use crate::nn::{Activation, Linear, VarBuilder};
use numr::autograd::Var;
use numr::ops::{ActivationOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Projects vision encoder hidden states to the LLM's hidden dimension.
///
/// Supports two modes:
/// - `Linear`: single linear projection (vision_hidden -> llm_hidden)
/// - `Mlp`: two-layer MLP with GELU activation
pub enum MultimodalProjector<R: Runtime> {
    /// Single linear projection
    Linear(Box<Linear<R>>),
    /// Two-layer MLP: linear1 -> activation -> linear2
    Mlp(Box<ProjectorMlp<R>>),
}

/// Inner struct for the MLP projector variant.
pub struct ProjectorMlp<R: Runtime> {
    pub linear1: Linear<R>,
    pub act: Activation,
    pub linear2: Linear<R>,
}

impl<R: Runtime> MultimodalProjector<R> {
    /// Load projector from a VarBuilder.
    ///
    /// `vision_hidden`: output dimension of the vision encoder
    /// `llm_hidden`: input dimension of the LLM
    pub fn from_varbuilder(
        vb: &mut VarBuilder<R>,
        _vision_hidden: usize,
        _llm_hidden: usize,
        config: &VisionConfig,
    ) -> Result<Self> {
        match config.projector_type.as_str() {
            "linear" => {
                let weight = vb.take_tensor("weight")?;
                let bias = vb.take_tensor_optional("bias")?;
                Ok(Self::Linear(Box::new(Linear::new(weight, bias, false))))
            }
            "mlp" => {
                let mut l1_vb = vb.pp("linear_1");
                let linear1 = Linear::new(
                    l1_vb.take_tensor("weight")?,
                    l1_vb.take_tensor_optional("bias")?,
                    false,
                );

                let mut l2_vb = vb.pp("linear_2");
                let linear2 = Linear::new(
                    l2_vb.take_tensor("weight")?,
                    l2_vb.take_tensor_optional("bias")?,
                    false,
                );

                Ok(Self::Mlp(Box::new(ProjectorMlp {
                    linear1,
                    act: Activation::Gelu,
                    linear2,
                })))
            }
            other => Err(Error::ModelError {
                reason: format!("unknown projector type: '{other}', expected 'linear' or 'mlp'"),
            }),
        }
    }

    /// Forward: project vision features to LLM embedding space.
    ///
    /// input: [B, num_patches, vision_hidden]
    /// output: [B, num_patches, llm_hidden]
    pub fn forward_inference<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let input_var = Var::new(input.clone(), false);
        match self {
            Self::Linear(linear) => {
                let out = linear.forward(client, &input_var)?;
                Ok(out.tensor().clone())
            }
            Self::Mlp(mlp) => {
                let h = mlp.linear1.forward(client, &input_var)?;
                let h_act = mlp.act.forward(client, h.tensor()).map_err(Error::Numr)?;
                let h_var = Var::new(h_act, false);
                let out = mlp.linear2.forward(client, &h_var)?;
                Ok(out.tensor().clone())
            }
        }
    }
}
