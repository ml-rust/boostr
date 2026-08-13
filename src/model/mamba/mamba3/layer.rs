//! Mamba3 layer: weights, struct definition, construction, and loading.

use super::config::Mamba3Config;
use crate::error::{Error, Result};
use crate::nn::{Conv1d, Linear, RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Mamba3 layer implementing trapezoidal discretization, optional complex RoPE,
/// and optional MIMO projections.
pub struct Mamba3<R: Runtime> {
    pub(in crate::model::mamba::mamba3) config: Mamba3Config,
    pub(in crate::model::mamba::mamba3) in_proj: Linear<R>,
    pub(in crate::model::mamba::mamba3) out_proj: Linear<R>,
    pub(in crate::model::mamba::mamba3) lambda_proj: Linear<R>,
    pub(in crate::model::mamba::mamba3) theta_proj: Option<Linear<R>>,
    pub(in crate::model::mamba::mamba3) b_bias: Var<R>,
    pub(in crate::model::mamba::mamba3) c_bias: Var<R>,
    pub(in crate::model::mamba::mamba3) dt_bias: Option<Var<R>>,
    pub(in crate::model::mamba::mamba3) a_log: Var<R>,
    pub(in crate::model::mamba::mamba3) d_param: Option<Var<R>>,
    pub(in crate::model::mamba::mamba3) bc_norm: RmsNorm<R>,
    pub(in crate::model::mamba::mamba3) norm: RmsNorm<R>,
    pub(in crate::model::mamba::mamba3) conv1d: Option<Conv1d<R>>,
    pub(in crate::model::mamba::mamba3) mimo_x_up: Option<Linear<R>>,
    pub(in crate::model::mamba::mamba3) mimo_x_down: Option<Linear<R>>,
}

/// Bundled weight tensors for Mamba3 construction (reduces parameter count).
pub struct Mamba3Weights<R: Runtime> {
    pub in_proj: Linear<R>,
    pub out_proj: Linear<R>,
    pub lambda_proj: Linear<R>,
    pub theta_proj: Option<Linear<R>>,
    pub b_bias: Tensor<R>,
    pub c_bias: Tensor<R>,
    pub dt_bias: Option<Tensor<R>>,
    pub a_log: Tensor<R>,
    pub d_param: Option<Tensor<R>>,
    pub bc_norm: RmsNorm<R>,
    pub norm: RmsNorm<R>,
    pub conv1d: Option<Conv1d<R>>,
    pub mimo_x_up: Option<Linear<R>>,
    pub mimo_x_down: Option<Linear<R>>,
}

impl<R: Runtime> Mamba3<R> {
    /// Create a new Mamba3 layer from config and weights.
    pub fn new(config: Mamba3Config, weights: Mamba3Weights<R>, trainable: bool) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            out_proj: weights.out_proj,
            lambda_proj: weights.lambda_proj,
            theta_proj: weights.theta_proj,
            b_bias: Var::new(weights.b_bias, trainable),
            c_bias: Var::new(weights.c_bias, trainable),
            dt_bias: weights.dt_bias.map(|t| Var::new(t, trainable)),
            a_log: Var::new(weights.a_log, trainable),
            d_param: weights.d_param.map(|t| Var::new(t, trainable)),
            bc_norm: weights.bc_norm,
            norm: weights.norm,
            conv1d: weights.conv1d,
            mimo_x_up: weights.mimo_x_up,
            mimo_x_down: weights.mimo_x_down,
        }
    }

    /// Load from a VarBuilder using Mamba3 mixer naming.
    pub fn from_varbuilder(
        config: &Mamba3Config,
        vb: &mut VarBuilder<R>,
        trainable: bool,
    ) -> Result<Self> {
        config.validate()?;
        let in_proj = take_linear(vb, "in_proj.weight", "in_proj.bias", trainable)?;
        let out_proj = take_linear(vb, "out_proj.weight", "out_proj.bias", trainable)?;
        let lambda_proj = take_linear(vb, "lambda_proj.weight", "lambda_proj.bias", trainable)?;

        let theta_proj = if config.use_complex_rope {
            if !vb.contains("theta_proj.weight") {
                return Err(Error::ModelError {
                    reason: "Mamba3 complex_rope requires theta_proj.weight".into(),
                });
            }
            Some(take_linear(
                vb,
                "theta_proj.weight",
                "theta_proj.bias",
                trainable,
            )?)
        } else if vb.contains("theta_proj.weight") {
            Some(take_linear(
                vb,
                "theta_proj.weight",
                "theta_proj.bias",
                trainable,
            )?)
        } else {
            None
        };

        let b_bias = vb.take_tensor("b_bias")?;
        let c_bias = vb.take_tensor("c_bias")?;
        let dt_bias = if config.use_dt_bias {
            Some(vb.take_tensor("dt_bias")?)
        } else {
            None
        };
        let a_log = vb.take_tensor("A_log")?;
        let d_param = if config.use_d {
            Some(vb.take_tensor("D")?)
        } else {
            None
        };

        let bc_norm = RmsNorm::new(vb.take_tensor("bc_norm.weight")?, 1e-6, trainable);
        let norm = RmsNorm::new(vb.take_tensor("norm.weight")?, 1e-6, trainable);

        let conv1d = if config.use_conv {
            let conv_weight = vb.take_tensor("conv1d.weight")?;
            let conv_bias = if vb.contains("conv1d.bias") {
                Some(vb.take_tensor("conv1d.bias")?)
            } else {
                None
            };
            let causal_pad = config.d_conv - 1;
            Some(Conv1d::new(
                conv_weight,
                conv_bias,
                1,
                PaddingMode::Custom(causal_pad, 0, 0, 0),
                1,
                config.conv_channels(),
                trainable,
            ))
        } else {
            None
        };

        let (mimo_x_up, mimo_x_down) = if config.mimo_rank > 0 {
            let up = take_linear(vb, "mimo_x_up.weight", "mimo_x_up.bias", trainable)?;
            let down = take_linear(vb, "mimo_x_down.weight", "mimo_x_down.bias", trainable)?;
            (Some(up), Some(down))
        } else {
            (None, None)
        };

        let weights = Mamba3Weights {
            in_proj,
            out_proj,
            lambda_proj,
            theta_proj,
            b_bias,
            c_bias,
            dt_bias,
            a_log,
            d_param,
            bc_norm,
            norm,
            conv1d,
            mimo_x_up,
            mimo_x_down,
        };
        Ok(Self::new(config.clone(), weights, trainable))
    }

    pub fn config(&self) -> &Mamba3Config {
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
    let bias = if vb.contains(bias_name) {
        Some(vb.take_tensor(bias_name)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, trainable))
}
