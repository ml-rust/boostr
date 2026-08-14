//! Mamba3 layer: weights, struct definition, construction, and loading.

use super::config::Mamba3Config;
use crate::error::{Error, Result};
use crate::nn::{Conv1d, Init, Linear, Module, RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, PaddingMode, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

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

/// Bundled Mamba3 weights with explicit stable IDs for raw `Var` fields.
pub struct Mamba3WeightsWithIds<R: Runtime> {
    pub in_proj: Linear<R>,
    pub out_proj: Linear<R>,
    pub lambda_proj: Linear<R>,
    pub theta_proj: Option<Linear<R>>,
    pub b_bias: (Tensor<R>, TensorId),
    pub c_bias: (Tensor<R>, TensorId),
    pub dt_bias: Option<(Tensor<R>, TensorId)>,
    pub a_log: (Tensor<R>, TensorId),
    pub d_param: Option<(Tensor<R>, TensorId)>,
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

    /// Create a Mamba3 layer while preserving explicit IDs for raw parameters.
    pub fn with_ids(
        config: Mamba3Config,
        weights: Mamba3WeightsWithIds<R>,
        trainable: bool,
    ) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            out_proj: weights.out_proj,
            lambda_proj: weights.lambda_proj,
            theta_proj: weights.theta_proj,
            b_bias: Var::with_id(weights.b_bias.0, weights.b_bias.1, trainable),
            c_bias: Var::with_id(weights.c_bias.0, weights.c_bias.1, trainable),
            dt_bias: weights
                .dt_bias
                .map(|(tensor, id)| Var::with_id(tensor, id, trainable)),
            a_log: Var::with_id(weights.a_log.0, weights.a_log.1, trainable),
            d_param: weights
                .d_param
                .map(|(tensor, id)| Var::with_id(tensor, id, trainable)),
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

    /// Build from a VarBuilder, initializing missing tensors for fresh training.
    pub fn init<C>(
        config: &Mamba3Config,
        vb: &mut VarBuilder<R>,
        dtype: DType,
        client: &C,
        trainable: bool,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + RandomOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + TensorOps<R>,
    {
        macro_rules! init_tensor {
            ($name:literal, $shape:expr, $init:expr) => {
                vb.take_or_init_tensor($name, $shape, dtype, $init, client)?
            };
        }
        macro_rules! init_linear_layer {
            ($weight:literal, $weight_shape:expr, $bias:literal, $bias_shape:expr) => {
                init_linear(
                    vb,
                    $weight,
                    $weight_shape,
                    $bias,
                    $bias_shape,
                    dtype,
                    client,
                    trainable,
                )?
            };
        }

        config.validate()?;
        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let theta_dim = config.nheads * (config.d_state / 2);

        let in_proj = init_linear_layer!(
            "in_proj.weight",
            &[config.proj_dim(), config.d_model],
            "in_proj.bias",
            &[config.proj_dim()]
        );
        let out_proj = init_linear_layer!(
            "out_proj.weight",
            &[config.d_model, d_inner],
            "out_proj.bias",
            &[config.d_model]
        );
        let lambda_proj = init_linear_layer!(
            "lambda_proj.weight",
            &[config.nheads, config.d_model],
            "lambda_proj.bias",
            &[config.nheads]
        );
        let theta_proj = if config.use_complex_rope || vb.contains("theta_proj.weight") {
            Some(init_linear_layer!(
                "theta_proj.weight",
                &[theta_dim, config.d_model],
                "theta_proj.bias",
                &[theta_dim]
            ))
        } else {
            None
        };

        let b_bias = init_tensor!("b_bias", &[config.nheads, config.d_state], Init::Zeros);
        let c_bias = init_tensor!("c_bias", &[config.nheads, config.d_state], Init::Zeros);
        let dt_bias = if config.use_dt_bias {
            Some(init_tensor!("dt_bias", &[config.nheads], Init::Zeros))
        } else {
            None
        };
        let a_log = init_tensor!("A_log", &[config.nheads], Init::Zeros);
        let d_param = if config.use_d {
            Some(init_tensor!("D", &[config.nheads], Init::Ones))
        } else {
            None
        };

        let bc_norm = RmsNorm::new(
            init_tensor!("bc_norm.weight", &[config.d_state], Init::Ones),
            1e-6,
            trainable,
        );
        let norm = RmsNorm::new(
            init_tensor!("norm.weight", &[d_inner], Init::Ones),
            1e-6,
            trainable,
        );

        let conv1d = if config.use_conv {
            let conv_weight = init_tensor!(
                "conv1d.weight",
                &[conv_channels, 1, config.d_conv],
                Init::PyTorchLinear
            );
            let conv_bias = if vb.contains("conv1d.bias") {
                Some(init_tensor!("conv1d.bias", &[conv_channels], Init::Zeros))
            } else {
                None
            };
            Some(Conv1d::new(
                conv_weight,
                conv_bias,
                1,
                PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
                1,
                conv_channels,
                trainable,
            ))
        } else {
            None
        };

        let (mimo_x_up, mimo_x_down) = if config.mimo_rank > 0 {
            let up = init_linear_layer!(
                "mimo_x_up.weight",
                &[config.headdim * config.mimo_rank, config.headdim],
                "mimo_x_up.bias",
                &[config.headdim * config.mimo_rank]
            );
            let down = init_linear_layer!(
                "mimo_x_down.weight",
                &[config.headdim, config.headdim * config.mimo_rank],
                "mimo_x_down.bias",
                &[config.headdim]
            );
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

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.in_proj.parameters());
        params.extend(self.out_proj.parameters());
        params.extend(self.lambda_proj.parameters());
        if let Some(theta_proj) = &self.theta_proj {
            params.extend(theta_proj.parameters());
        }
        params.push((self.b_bias.id(), &self.b_bias));
        params.push((self.c_bias.id(), &self.c_bias));
        if let Some(dt_bias) = &self.dt_bias {
            params.push((dt_bias.id(), dt_bias));
        }
        params.push((self.a_log.id(), &self.a_log));
        if let Some(d_param) = &self.d_param {
            params.push((d_param.id(), d_param));
        }
        params.extend(self.bc_norm.parameters());
        params.extend(self.norm.parameters());
        if let Some(conv1d) = &self.conv1d {
            params.extend(conv1d.parameters());
        }
        if let Some(mimo_x_up) = &self.mimo_x_up {
            params.extend(mimo_x_up.parameters());
        }
        if let Some(mimo_x_down) = &self.mimo_x_down {
            params.extend(mimo_x_down.parameters());
        }
        params
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }
}

impl<R: Runtime> Module<R> for Mamba3<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Mamba3::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "in_proj", self.in_proj.named_parameters());
        extend_named(&mut params, "out_proj", self.out_proj.named_parameters());
        extend_named(
            &mut params,
            "lambda_proj",
            self.lambda_proj.named_parameters(),
        );
        if let Some(theta_proj) = &self.theta_proj {
            extend_named(&mut params, "theta_proj", theta_proj.named_parameters());
        }
        params.push(("b_bias".to_string(), &self.b_bias));
        params.push(("c_bias".to_string(), &self.c_bias));
        if let Some(dt_bias) = &self.dt_bias {
            params.push(("dt_bias".to_string(), dt_bias));
        }
        params.push(("a_log".to_string(), &self.a_log));
        if let Some(d_param) = &self.d_param {
            params.push(("d_param".to_string(), d_param));
        }
        extend_named(&mut params, "bc_norm", self.bc_norm.named_parameters());
        extend_named(&mut params, "norm", self.norm.named_parameters());
        if let Some(conv1d) = &self.conv1d {
            extend_named(&mut params, "conv1d", conv1d.named_parameters());
        }
        if let Some(mimo_x_up) = &self.mimo_x_up {
            extend_named(&mut params, "mimo_x_up", mimo_x_up.named_parameters());
        }
        if let Some(mimo_x_down) = &self.mimo_x_down {
            extend_named(&mut params, "mimo_x_down", mimo_x_down.named_parameters());
        }
        params
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Mamba3::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        Mamba3::trainable_parameters(self)
    }
}

fn extend_named<'a, R: Runtime>(
    params: &mut Vec<(String, &'a Var<R>)>,
    prefix: &str,
    child: Vec<(String, &'a Var<R>)>,
) {
    params.extend(
        child
            .into_iter()
            .map(|(name, var)| (format!("{prefix}.{name}"), var)),
    );
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

#[allow(clippy::too_many_arguments)]
fn init_linear<R, C>(
    vb: &mut VarBuilder<R>,
    weight_name: &str,
    weight_shape: &[usize],
    bias_name: &str,
    bias_shape: &[usize],
    dtype: DType,
    client: &C,
    trainable: bool,
) -> Result<Linear<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + RandomOps<R> + ScalarOps<R> + BinaryOps<R> + CompareOps<R> + TensorOps<R>,
{
    let weight = vb.take_or_init_tensor(
        weight_name,
        weight_shape,
        dtype,
        Init::PyTorchLinear,
        client,
    )?;
    let bias = if vb.contains(bias_name) {
        Some(vb.take_or_init_tensor(bias_name, bias_shape, dtype, Init::Zeros, client)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, trainable))
}
