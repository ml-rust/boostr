//! Mamba1 training/full-sequence forward pass.

use super::layer::Mamba1;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::{SsmInput, ssm_forward_sequential, var_contiguous};
use numr::autograd::{
    Var, var_exp, var_mul, var_narrow, var_neg, var_reshape, var_silu, var_softplus, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba1<R> {
    /// Full-sequence Var forward pass.
    ///
    /// x: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + BinaryOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + CompareOps<R>,
    {
        self.config.validate()?;
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Error::ModelError {
                reason: format!("expected [batch, seq_len, d_model], got shape {:?}", shape),
            });
        }
        let seq_len = shape[1];
        if shape[2] != self.config.d_model {
            return Err(Error::ModelError {
                reason: format!(
                    "d_model mismatch: expected {}, got {}",
                    self.config.d_model, shape[2]
                ),
            });
        }

        let d_inner = self.config.d_inner();

        // 1. Input projection and split into SSM path and residual/gate path.
        let projected = self.in_proj.forward(client, x)?;
        let x_part = var_contiguous(&var_narrow(&projected, 2, 0, d_inner).map_err(Error::Numr)?)?;
        let residual =
            var_contiguous(&var_narrow(&projected, 2, d_inner, d_inner).map_err(Error::Numr)?)?;

        // 2. Depthwise causal Conv1D over the SSM path.
        let x_ncl = var_contiguous(&var_transpose(&x_part).map_err(Error::Numr)?)?;
        let conv_out = self.conv1d.forward(client, &x_ncl)?;
        let conv_out = var_contiguous(&var_narrow(&conv_out, 2, 0, seq_len).map_err(Error::Numr)?)?;
        let conv_out = var_contiguous(&var_transpose(&conv_out).map_err(Error::Numr)?)?;
        let conv_out = var_silu(&conv_out, client).map_err(Error::Numr)?;

        // 3. Sequential selective scan (never chunked/parallel for Mamba1 training).
        let y = self.selective_scan_sequential(client, &conv_out)?;

        // 4. Gate with the residual branch and project back to d_model.
        let gate = var_silu(&residual, client).map_err(Error::Numr)?;
        let gated = var_mul(&y, &gate, client).map_err(Error::Numr)?;
        self.out_proj.forward(client, &gated)
    }

    pub(in crate::model::mamba::mamba1) fn selective_scan_sequential<C>(
        &self,
        client: &C,
        u: &Var<R>,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        let shape = u.shape();
        if shape.len() != 3 {
            return Err(Error::ModelError {
                reason: format!("expected [batch, seq_len, d_inner], got shape {:?}", shape),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let d_inner = self.config.d_inner();
        if shape[2] != d_inner {
            return Err(Error::ModelError {
                reason: format!("d_inner mismatch: expected {}, got {}", d_inner, shape[2]),
            });
        }

        let x_dbl = var_contiguous(&self.x_proj.forward(client, u)?)?;
        let delta = var_contiguous(&var_narrow(&x_dbl, 2, 0, d_inner).map_err(Error::Numr)?)?;
        let b_proj = var_contiguous(
            &var_narrow(&x_dbl, 2, d_inner, self.config.d_state).map_err(Error::Numr)?,
        )?;
        let c_proj = var_contiguous(
            &var_narrow(
                &x_dbl,
                2,
                d_inner + self.config.d_state,
                self.config.d_state,
            )
            .map_err(Error::Numr)?,
        )?;

        let mut delta = self.dt_proj.forward(client, &delta)?;
        if self.config.dt_softplus {
            delta = var_softplus(&delta, client).map_err(Error::Numr)?;
        }

        // Preserve the oxidizr Mamba1 reference transform: A = exp(-a_log).
        let a_neg = var_neg(&self.a_log, client).map_err(Error::Numr)?;
        let a = var_exp(&a_neg, client).map_err(Error::Numr)?;
        self.selective_scan_from_parts(client, u, &delta, &a, &b_proj, &c_proj, batch, seq_len)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::model::mamba::mamba1) fn selective_scan_from_parts<C>(
        &self,
        client: &C,
        u: &Var<R>,
        delta: &Var<R>,
        a: &Var<R>,
        b: &Var<R>,
        c: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        let d_inner = self.config.d_inner();
        let x_ssm = var_reshape(u, &[batch, seq_len, d_inner, 1]).map_err(Error::Numr)?;
        let b_proj =
            var_reshape(b, &[batch, seq_len, 1, self.config.d_state]).map_err(Error::Numr)?;
        let c_proj =
            var_reshape(c, &[batch, seq_len, 1, self.config.d_state]).map_err(Error::Numr)?;
        let scan_config = self.config.scan_config();
        let ssm_input = SsmInput {
            x: &x_ssm,
            a,
            b: &b_proj,
            c: &c_proj,
            d_param: self.d_param.as_ref(),
            dt: delta,
            config: &scan_config,
            hidden_state_clamp: self.config.hidden_state_clamp,
        };
        let out = ssm_forward_sequential(client, &ssm_input)?;
        var_reshape(&out, &[batch, seq_len, d_inner]).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mamba::mamba1::config::Mamba1Config;
    use crate::model::mamba::mamba1::layer::Mamba1Weights;
    use crate::nn::{Conv1d, Linear};
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn linear_with_value(rows: usize, cols: usize, value: f32) -> Linear<CpuRuntime> {
        let (_, device) = cpu_setup();
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&vec![value; rows * cols], &[rows, cols], &device)
                .unwrap(),
            None,
            false,
        )
    }

    fn tiny_mamba1(config: Mamba1Config) -> Mamba1<CpuRuntime> {
        let (_, device) = cpu_setup();
        let d_inner = config.d_inner();
        let in_proj = linear_with_value(config.in_proj_dim(), config.d_model, 0.0);
        let conv1d = Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.conv_channels() * config.d_conv],
                &[config.conv_channels(), 1, config.d_conv],
                &device,
            )
            .unwrap(),
            None,
            1,
            PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
            1,
            config.conv_channels(),
            false,
        );
        let x_proj = linear_with_value(config.x_proj_dim(), d_inner, 0.0);
        let dt_proj = linear_with_value(d_inner, d_inner, 0.0);
        let out_proj = linear_with_value(config.d_model, d_inner, 0.0);
        let weights = Mamba1Weights {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            out_proj,
            a_log: Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; d_inner * config.d_state],
                &[d_inner, config.d_state],
                &device,
            )
            .unwrap(),
            d_param: if config.use_d {
                Some(
                    Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; d_inner], &[d_inner], &device)
                        .unwrap(),
                )
            } else {
                None
            },
        };
        Mamba1::new(config, weights, false)
    }

    #[test]
    fn test_mamba1_forward_shape() {
        let (client, device) = cpu_setup();
        let config = Mamba1Config::new(4)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_d(false);
        let mamba = tiny_mamba1(config);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 3, 4], &device).unwrap(),
            false,
        );

        let out = mamba.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 3, 4]);
    }

    #[test]
    fn test_mamba1_forward_invalid_input() {
        let (client, device) = cpu_setup();
        let config = Mamba1Config::new(4)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_d(false);
        let mamba = tiny_mamba1(config);

        let x_2d = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4], &[1, 4], &device).unwrap(),
            false,
        );
        assert!(mamba.forward(&client, &x_2d).is_err());

        let x_wrong = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[1, 3, 2], &device).unwrap(),
            false,
        );
        assert!(mamba.forward(&client, &x_wrong).is_err());
    }

    #[test]
    fn test_selective_scan_matches_f64_reference() {
        let (client, device) = cpu_setup();
        let config = Mamba1Config::new(2)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_d(true)
            .with_hidden_state_clamp(Some(30.0));
        let mut mamba = tiny_mamba1(config);
        mamba.d_param = Some(Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, -0.2], &[2], &device).unwrap(),
            false,
        ));

        let u_data = [1.0f32, 0.5, -0.25, 2.0, 0.75, -1.0];
        let delta_data = [0.2f32, 0.1, 0.3, 0.25, 0.15, 0.35];
        let a_data = [-0.5f32, -1.0, -0.25, -0.75];
        let b_data = [0.5f32, -0.2, 0.1, 0.3, -0.4, 0.25];
        let c_data = [1.0f32, 0.5, -0.25, 0.75, 0.6, -0.1];
        let d_data = [0.1f32, -0.2];

        let u = Var::new(
            Tensor::<CpuRuntime>::from_slice(&u_data, &[1, 3, 2], &device).unwrap(),
            false,
        );
        let delta = Var::new(
            Tensor::<CpuRuntime>::from_slice(&delta_data, &[1, 3, 2], &device).unwrap(),
            false,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device).unwrap(),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 3, 2], &device).unwrap(),
            false,
        );
        let c = Var::new(
            Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 3, 2], &device).unwrap(),
            false,
        );

        let out = mamba
            .selective_scan_from_parts(&client, &u, &delta, &a, &b, &c, 1, 3)
            .unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        let expected = selective_scan_reference(
            &u_data,
            &delta_data,
            &a_data,
            &b_data,
            &c_data,
            Some(&d_data),
            Some(30.0),
            3,
            2,
            2,
        );

        for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 2e-5,
                "idx={i}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_selective_scan_hidden_state_clamp_matches_f64_reference() {
        let (client, device) = cpu_setup();
        let config = Mamba1Config::new(1)
            .with_expand(1)
            .with_d_state(1)
            .with_dt_softplus(false)
            .with_use_d(false)
            .with_hidden_state_clamp(Some(1.0));
        let mamba = tiny_mamba1(config);

        let u_data = [10.0f32, 10.0];
        let delta_data = [1.0f32, 1.0];
        let a_data = [1.0f32];
        let b_data = [1.0f32, 1.0];
        let c_data = [1.0f32, 1.0];

        let u = Var::new(
            Tensor::<CpuRuntime>::from_slice(&u_data, &[1, 2, 1], &device).unwrap(),
            false,
        );
        let delta = Var::new(
            Tensor::<CpuRuntime>::from_slice(&delta_data, &[1, 2, 1], &device).unwrap(),
            false,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&a_data, &[1, 1], &device).unwrap(),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 2, 1], &device).unwrap(),
            false,
        );
        let c = Var::new(
            Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 2, 1], &device).unwrap(),
            false,
        );

        let out = mamba
            .selective_scan_from_parts(&client, &u, &delta, &a, &b, &c, 1, 2)
            .unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        let expected = selective_scan_reference(
            &u_data,
            &delta_data,
            &a_data,
            &b_data,
            &c_data,
            None,
            Some(1.0),
            2,
            1,
            1,
        );

        for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 1e-6,
                "idx={i}: actual={actual}, expected={expected}"
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn selective_scan_reference(
        u: &[f32],
        delta: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: Option<&[f32]>,
        clamp: Option<f64>,
        seq_len: usize,
        d_inner: usize,
        d_state: usize,
    ) -> Vec<f64> {
        let mut h = vec![0.0f64; d_inner * d_state];
        let mut out = Vec::with_capacity(seq_len * d_inner);

        for t in 0..seq_len {
            for dim in 0..d_inner {
                let u_t = u[t * d_inner + dim] as f64;
                let delta_t = delta[t * d_inner + dim] as f64;
                for state in 0..d_state {
                    let idx = dim * d_state + state;
                    let b_t = b[t * d_state + state] as f64;
                    h[idx] = (delta_t * a[idx] as f64).exp() * h[idx] + delta_t * b_t * u_t;
                    if let Some(limit) = clamp {
                        h[idx] = h[idx].clamp(-limit, limit);
                    }
                }
            }

            for dim in 0..d_inner {
                let mut y = 0.0f64;
                for state in 0..d_state {
                    y += h[dim * d_state + state] * c[t * d_state + state] as f64;
                }
                if let Some(d_param) = d {
                    y += d_param[dim] as f64 * u[t * d_inner + dim] as f64;
                }
                out.push(y);
            }
        }

        out
    }
}
