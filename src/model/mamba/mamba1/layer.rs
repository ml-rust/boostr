//! Mamba1 layer: weights, struct definition, construction, and loading.

use super::config::Mamba1Config;
use crate::error::Result;
use crate::nn::{Conv1d, Init, Linear, Module, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, PaddingMode, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

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

/// Bundled Mamba1 weights with explicit stable IDs for raw `Var` fields.
pub struct Mamba1WeightsWithIds<R: Runtime> {
    pub in_proj: Linear<R>,
    pub conv1d: Conv1d<R>,
    pub x_proj: Linear<R>,
    pub dt_proj: Linear<R>,
    pub out_proj: Linear<R>,
    pub a_log: (Tensor<R>, TensorId),
    pub d_param: Option<(Tensor<R>, TensorId)>,
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

    /// Create a Mamba1 layer while preserving explicit IDs for raw parameters.
    pub fn with_ids(
        config: Mamba1Config,
        weights: Mamba1WeightsWithIds<R>,
        trainable: bool,
    ) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            conv1d: weights.conv1d,
            x_proj: weights.x_proj,
            dt_proj: weights.dt_proj,
            out_proj: weights.out_proj,
            a_log: Var::with_id(weights.a_log.0, weights.a_log.1, trainable),
            d_param: weights
                .d_param
                .map(|(tensor, id)| Var::with_id(tensor, id, trainable)),
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

    /// Build from a VarBuilder, initializing missing tensors for fresh training.
    pub fn init<C>(
        config: &Mamba1Config,
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
        config.validate()?;
        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();

        let in_proj = init_linear(
            vb,
            "in_proj.weight",
            &[config.in_proj_dim(), config.d_model],
            "in_proj.bias",
            &[config.in_proj_dim()],
            dtype,
            client,
            trainable,
        )?;
        let conv_weight = vb.take_or_init_tensor(
            "conv1d.weight",
            &[conv_channels, 1, config.d_conv],
            dtype,
            Init::PyTorchLinear,
            client,
        )?;
        let conv_bias = if vb.contains("conv1d.bias") {
            Some(vb.take_or_init_tensor(
                "conv1d.bias",
                &[conv_channels],
                dtype,
                Init::Zeros,
                client,
            )?)
        } else {
            None
        };
        let conv1d = Conv1d::new(
            conv_weight,
            conv_bias,
            1,
            PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
            1,
            conv_channels,
            trainable,
        );
        let x_proj = init_linear(
            vb,
            "x_proj.weight",
            &[config.x_proj_dim(), d_inner],
            "x_proj.bias",
            &[config.x_proj_dim()],
            dtype,
            client,
            trainable,
        )?;
        let dt_proj = init_linear(
            vb,
            "dt_proj.weight",
            &[d_inner, d_inner],
            "dt_proj.bias",
            &[d_inner],
            dtype,
            client,
            trainable,
        )?;
        let out_proj = init_linear(
            vb,
            "out_proj.weight",
            &[config.d_model, d_inner],
            "out_proj.bias",
            &[config.d_model],
            dtype,
            client,
            trainable,
        )?;
        let a_log = take_or_init_tensor_any(
            vb,
            &["a_log", "A_log"],
            &[d_inner, config.d_state],
            dtype,
            Init::Zeros,
            client,
        )?;
        let d_param = if config.use_d {
            Some(take_or_init_tensor_any(
                vb,
                &["d", "D"],
                &[d_inner],
                dtype,
                Init::Ones,
                client,
            )?)
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

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = Vec::new();
        params.extend(self.in_proj.parameters());
        params.extend(self.conv1d.parameters());
        params.extend(self.x_proj.parameters());
        params.extend(self.dt_proj.parameters());
        params.extend(self.out_proj.parameters());
        params.push((self.a_log.id(), &self.a_log));
        if let Some(d_param) = &self.d_param {
            params.push((d_param.id(), d_param));
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

impl<R: Runtime> Module<R> for Mamba1<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Mamba1::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "in_proj", self.in_proj.named_parameters());
        extend_named(&mut params, "conv1d", self.conv1d.named_parameters());
        extend_named(&mut params, "x_proj", self.x_proj.named_parameters());
        extend_named(&mut params, "dt_proj", self.dt_proj.named_parameters());
        extend_named(&mut params, "out_proj", self.out_proj.named_parameters());
        params.push(("a_log".to_string(), &self.a_log));
        if let Some(d_param) = &self.d_param {
            params.push(("d_param".to_string(), d_param));
        }
        params
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Mamba1::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        Mamba1::trainable_parameters(self)
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
    let bias = vb.take_tensor_optional(bias_name)?;
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

fn take_or_init_tensor_any<R, C>(
    vb: &mut VarBuilder<R>,
    names: &[&str],
    shape: &[usize],
    dtype: DType,
    init: Init,
    client: &C,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + RandomOps<R> + ScalarOps<R> + BinaryOps<R> + CompareOps<R> + TensorOps<R>,
{
    for name in names {
        if vb.contains(name) {
            return vb.take_or_init_tensor(name, shape, dtype, init, client);
        }
    }
    let Some(name) = names.first() else {
        return Err(crate::error::Error::ModelError {
            reason: "missing tensor name for initialization".into(),
        });
    };
    vb.take_or_init_tensor(name, shape, dtype, init, client)
}
