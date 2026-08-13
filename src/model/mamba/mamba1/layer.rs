//! Mamba1 layer: weights, struct definition, construction, and loading.

use super::config::Mamba1Config;
use crate::error::Result;
use crate::nn::{Conv1d, Linear, VarBuilder};
use numr::autograd::Var;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Mamba1 layer implementing the original selective SSM block.
pub struct Mamba1<R: Runtime> {
    pub(in crate::model::mamba::mamba1) config: Mamba1Config,
    pub(in crate::model::mamba::mamba1) in_proj: Linear<R>,
    pub(in crate::model::mamba::mamba1) conv1d: Conv1d<R>,
    pub(in crate::model::mamba::mamba1) x_proj: Linear<R>,
    pub(in crate::model::mamba::mamba1) dt_proj: Linear<R>,
    pub(in crate::model::mamba::mamba1) out_proj: Linear<R>,
    pub(in crate::model::mamba::mamba1) a_log: Var<R>,
    pub(in crate::model::mamba::mamba1) d_param: Option<Var<R>>,
}

/// Bundled weight tensors for Mamba1 construction (reduces parameter count).
pub struct Mamba1Weights<R: Runtime> {
    pub in_proj: Linear<R>,
    pub conv1d: Conv1d<R>,
    pub x_proj: Linear<R>,
    pub dt_proj: Linear<R>,
    pub out_proj: Linear<R>,
    pub a_log: Tensor<R>,
    pub d_param: Option<Tensor<R>>,
}

impl<R: Runtime> Mamba1<R> {
    /// Create a new Mamba1 layer from config and weights.
    pub fn new(config: Mamba1Config, weights: Mamba1Weights<R>, trainable: bool) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            conv1d: weights.conv1d,
            x_proj: weights.x_proj,
            dt_proj: weights.dt_proj,
            out_proj: weights.out_proj,
            a_log: Var::new(weights.a_log, trainable),
            d_param: weights.d_param.map(|t| Var::new(t, trainable)),
        }
    }

    /// Load from a VarBuilder using Mamba1 mixer naming.
    pub fn from_varbuilder(
        config: &Mamba1Config,
        vb: &mut VarBuilder<R>,
        trainable: bool,
    ) -> Result<Self> {
        config.validate()?;
        let in_proj = take_linear(vb, "in_proj.weight", "in_proj.bias", trainable)?;
        let conv_weight = vb.take_tensor("conv1d.weight")?;
        let conv_bias = vb.take_tensor_optional("conv1d.bias")?;
        let conv1d = Conv1d::new(
            conv_weight,
            conv_bias,
            1,
            PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
            1,
            config.conv_channels(),
            trainable,
        );
        let x_proj = take_linear(vb, "x_proj.weight", "x_proj.bias", trainable)?;
        let dt_proj = take_linear(vb, "dt_proj.weight", "dt_proj.bias", trainable)?;
        let out_proj = take_linear(vb, "out_proj.weight", "out_proj.bias", trainable)?;
        let a_log = take_tensor_any(vb, &["a_log", "A_log"])?;
        let d_param = if config.use_d {
            Some(take_tensor_any(vb, &["d", "D"])?)
        } else {
            None
        };

        let weights = Mamba1Weights {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            out_proj,
            a_log,
            d_param,
        };
        Ok(Self::new(config.clone(), weights, trainable))
    }

    pub fn config(&self) -> &Mamba1Config {
        &self.config
    }
}

fn take_linear<R: Runtime>(
    vb: &mut VarBuilder<R>,
    weight_name: &str,
    bias_name: &str,
    trainable: bool,
) -> Result<Linear<R>> {
    let weight = vb.take_tensor(weight_name)?;
    let bias = vb.take_tensor_optional(bias_name)?;
    Ok(Linear::new(weight, bias, trainable))
}

fn take_tensor_any<R: Runtime>(vb: &mut VarBuilder<R>, names: &[&str]) -> Result<Tensor<R>> {
    for name in names {
        if vb.contains(name) {
            return vb.take_tensor(name);
        }
    }
    Err(crate::error::Error::ModelError {
        reason: format!("missing required tensor; tried {}", names.join(" or ")),
    })
}
