//! [`Mamba3::init`]: build from a VarBuilder, initializing every tensor the
//! builder lacks, for fresh training.

use super::config::Mamba3Config;
use super::layer::{Mamba3, Mamba3Weights};
use crate::error::Result;
use crate::nn::{Conv1d, Init, Linear, RmsNorm, VarBuilder};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, PaddingMode, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba3<R> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Module, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;

    fn assert_named_shape<R: Runtime>(params: &[(String, &Var<R>)], name: &str, shape: &[usize]) {
        let actual = params
            .iter()
            .find(|(param_name, _)| param_name == name)
            .map(|(_, param)| param.shape().to_vec())
            .unwrap_or_else(|| panic!("missing parameter {name}"));
        assert_eq!(actual, shape, "shape mismatch for {name}");
    }

    #[test]
    fn test_mamba3_init_from_empty_varmap_shapes_and_ssm_defaults() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(4)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_complex_rope(true)
            .with_mimo_rank(2)
            .with_use_conv(true)
            .with_dt_softplus(false)
            .with_use_dt_bias(true)
            .with_use_d(true);
        let mut varmap = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);

        let mamba = Mamba3::init(&config, &mut vb, DType::F32, &client, true).unwrap();
        let params = mamba.named_parameters();
        let mut names: Vec<&str> = params.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "a_log",
                "b_bias",
                "bc_norm.weight",
                "c_bias",
                "conv1d.weight",
                "d_param",
                "dt_bias",
                "in_proj.weight",
                "lambda_proj.weight",
                "mimo_x_down.weight",
                "mimo_x_up.weight",
                "norm.weight",
                "out_proj.weight",
                "theta_proj.weight",
            ]
        );
        assert_named_shape(
            &params,
            "in_proj.weight",
            &[config.proj_dim(), config.d_model],
        );
        assert_named_shape(
            &params,
            "out_proj.weight",
            &[config.d_model, config.d_inner()],
        );
        assert_named_shape(
            &params,
            "lambda_proj.weight",
            &[config.nheads, config.d_model],
        );
        assert_named_shape(
            &params,
            "theta_proj.weight",
            &[config.nheads * (config.d_state / 2), config.d_model],
        );
        assert_named_shape(&params, "b_bias", &[config.nheads, config.d_state]);
        assert_named_shape(&params, "c_bias", &[config.nheads, config.d_state]);
        assert_named_shape(&params, "dt_bias", &[config.nheads]);
        assert_named_shape(&params, "a_log", &[config.nheads]);
        assert_named_shape(&params, "d_param", &[config.nheads]);
        assert_named_shape(&params, "bc_norm.weight", &[config.d_state]);
        assert_named_shape(&params, "norm.weight", &[config.d_inner()]);
        assert_named_shape(
            &params,
            "conv1d.weight",
            &[config.conv_channels(), 1, config.d_conv],
        );
        assert_named_shape(
            &params,
            "mimo_x_up.weight",
            &[config.headdim * config.mimo_rank, config.headdim],
        );
        assert_named_shape(
            &params,
            "mimo_x_down.weight",
            &[config.headdim, config.headdim * config.mimo_rank],
        );

        let a_log: Vec<f32> = mamba.a_log.tensor().contiguous().unwrap().to_vec();
        assert!(a_log.iter().all(|&value| value == 0.0));
        let d_param: Vec<f32> = mamba
            .d_param
            .as_ref()
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec();
        assert!(d_param.iter().all(|&value| value == 1.0));

        let mut strict_varmap = VarMap::<CpuRuntime>::new();
        let mut strict_vb = VarBuilder::new(&mut strict_varmap, &device);
        assert!(Mamba3::from_varbuilder(&config, &mut strict_vb, false).is_err());
    }
}
