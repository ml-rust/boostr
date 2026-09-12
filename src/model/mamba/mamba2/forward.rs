//! Mamba2 training forward pass. The inference pass is in `inference`.

use super::layer::Mamba2;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::{SsmInput, var_contiguous};
use numr::autograd::{
    Var, var_add, var_exp, var_mul, var_narrow, var_neg, var_reshape, var_silu, var_softplus,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba2<R> {
    /// Training forward pass.
    ///
    /// x: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
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
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Error::ModelError {
                reason: format!("expected [batch, seq_len, d_model], got shape {:?}", shape),
            });
        }
        let batch = shape[0];
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
        let n_groups_d_state = self.config.ngroups * self.config.d_state;

        // 1. Input projection: [B, S, d_model] -> [B, S, proj_dim]
        let projected = self.in_proj.forward(client, x)?;

        // 2. Split into z, xBC, dt
        let xbc_len = d_inner + 2 * n_groups_d_state;
        let z = var_contiguous(&var_narrow(&projected, 2, 0, d_inner).map_err(Error::Numr)?)?;
        let xbc =
            var_contiguous(&var_narrow(&projected, 2, d_inner, xbc_len).map_err(Error::Numr)?)?;
        let dt = var_contiguous(
            &var_narrow(&projected, 2, d_inner + xbc_len, self.config.nheads)
                .map_err(Error::Numr)?,
        )?;

        // 3. Causal conv1d on xBC
        let xbc_ncl = var_contiguous(&var_transpose(&xbc).map_err(Error::Numr)?)?;
        let xbc_conv = self.conv1d.forward(client, &xbc_ncl)?;
        let xbc = var_contiguous(&var_transpose(&xbc_conv).map_err(Error::Numr)?)?;

        // 4. SiLU activation
        let xbc = var_silu(&xbc, client).map_err(Error::Numr)?;

        // 5. Split xBC into x_ssm, B, C
        let x_ssm = var_contiguous(&var_narrow(&xbc, 2, 0, d_inner).map_err(Error::Numr)?)?;
        let b_proj =
            var_contiguous(&var_narrow(&xbc, 2, d_inner, n_groups_d_state).map_err(Error::Numr)?)?;
        let c_proj = var_contiguous(
            &var_narrow(&xbc, 2, d_inner + n_groups_d_state, n_groups_d_state)
                .map_err(Error::Numr)?,
        )?;

        // 6. Reshape for SSM
        let x_ssm = var_reshape(
            &x_ssm,
            &[batch, seq_len, self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)?;
        let b_proj = var_reshape(
            &b_proj,
            &[batch, seq_len, self.config.ngroups, self.config.d_state],
        )
        .map_err(Error::Numr)?;
        let c_proj = var_reshape(
            &c_proj,
            &[batch, seq_len, self.config.ngroups, self.config.d_state],
        )
        .map_err(Error::Numr)?;

        // 7. Compute A = -exp(A_log)
        let a = var_neg(&var_exp(&self.a_log, client).map_err(Error::Numr)?, client)
            .map_err(Error::Numr)?;

        // 8. dt = softplus(dt + dt_bias).
        // The bias goes INSIDE softplus (matching reference Mamba2 and this
        // crate's Mamba3): softplus(dt) + bias can be negative, which flips the
        // sign of the decay exponent exp(dt * A) and makes the recurrence
        // diverge instead of decay.
        let mut dt = dt;
        if let Some(ref bias) = self.dt_bias {
            dt = var_add(&dt, bias, client).map_err(Error::Numr)?;
        }
        if self.config.dt_softplus {
            dt = var_softplus(&dt, client).map_err(Error::Numr)?;
        }

        // 9. SSM forward
        let ssm_input = SsmInput {
            x: &x_ssm,
            a: &a,
            b: &b_proj,
            c: &c_proj,
            d_param: self.d_param.as_ref(),
            dt: &dt,
            config: &self.config,
            hidden_state_clamp: None,
        };
        let out = crate::model::mamba::ssm::ssm_forward_sequential(client, &ssm_input)?;

        // 10. Reshape back: [B, S, nheads, headdim] -> [B, S, d_inner]
        let out = var_reshape(&out, &[batch, seq_len, d_inner]).map_err(Error::Numr)?;

        // 11. Gate: out = out * silu(z)
        let z_gate = var_silu(&z, client).map_err(Error::Numr)?;
        let out = var_mul(&out, &z_gate, client).map_err(Error::Numr)?;

        // 12. Optional norm
        let out = if let Some(ref norm) = self.norm {
            norm.forward(client, &out)?
        } else {
            out
        };

        // 13. Output projection
        self.out_proj.forward(client, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mamba::mamba2::config::Mamba2Config;
    use crate::model::mamba::mamba2::layer::Mamba2Weights;
    use crate::nn::{Conv1d, Linear};
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Same tiny layer as [`tiny_mamba2`] but with softplus enabled and an explicit
    /// `dt_bias` filled with `bias_value`, for exercising the dt ordering.
    fn mamba2_with_dt_bias(bias_value: f32) -> (Mamba2<CpuRuntime>, Mamba2Config) {
        let (_, device) = cpu_setup();
        let config = Mamba2Config::new(8)
            .with_nheads(1)
            .with_d_state(4)
            .with_expand(2)
            .with_dt_softplus(true)
            .with_use_dt_bias(true)
            .with_use_d(false);

        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let proj_dim = config.proj_dim();

        let in_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 328], &[proj_dim, 8], &device).unwrap(),
            None,
            false,
        );
        let conv1d = Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device)
                .unwrap(),
            None,
            1,
            PaddingMode::Custom(3, 0, 0, 0),
            1,
            conv_channels,
            false,
        );
        let out_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 128], &[8, d_inner], &device).unwrap(),
            None,
            false,
        );
        let a_log =
            Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[config.nheads], &device).unwrap();
        let dt_bias =
            Tensor::<CpuRuntime>::from_slice(&[bias_value], &[config.nheads], &device).unwrap();

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias: Some(dt_bias),
            d_param: None,
            norm: None,
        };
        let mamba = Mamba2::new(config.clone(), weights, false);
        (mamba, config)
    }

    fn tiny_mamba2() -> (Mamba2<CpuRuntime>, Mamba2Config) {
        let (_, device) = cpu_setup();
        let config = Mamba2Config::new(8)
            .with_nheads(1)
            .with_d_state(4)
            .with_expand(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);

        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let proj_dim = config.proj_dim();

        let in_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 328], &[proj_dim, 8], &device).unwrap(),
            None,
            false,
        );
        let conv1d = Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device)
                .unwrap(),
            None,
            1,
            PaddingMode::Custom(3, 0, 0, 0),
            1,
            conv_channels,
            false,
        );
        let out_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 128], &[8, d_inner], &device).unwrap(),
            None,
            false,
        );
        let a_log =
            Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[config.nheads], &device).unwrap();

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias: None,
            d_param: None,
            norm: None,
        };
        let mamba = Mamba2::new(config.clone(), weights, false);
        (mamba, config)
    }

    #[test]
    fn test_mamba2_forward_shape() {
        let (client, device) = cpu_setup();
        let (mamba, _) = tiny_mamba2();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[1, 4, 8], &device).unwrap(),
            false,
        );

        let out = mamba.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_mamba2_forward_invalid_input() {
        let (client, device) = cpu_setup();
        let (mamba, _) = tiny_mamba2();

        // 2D input should fail
        let x_2d = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[1, 8], &device).unwrap(),
            false,
        );
        assert!(mamba.forward(&client, &x_2d).is_err());

        // Wrong d_model should fail
        let x_wrong = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 4, 3], &device).unwrap(),
            false,
        );
        assert!(mamba.forward(&client, &x_wrong).is_err());
    }

    /// `dt_bias` must be added INSIDE softplus: `softplus(dt + bias)`.
    ///
    /// Regression: this computed `softplus(dt) + bias`. With the default zero-init
    /// bias the two are identical, so the bug is invisible until the bias trains
    /// away from zero — at which point a sufficiently negative bias makes dt
    /// negative, flipping the sign of the decay exponent `exp(dt * A)` so the
    /// recurrence diverges instead of decaying.
    ///
    /// A strongly negative bias separates the two orderings:
    ///   softplus(dt + bias) > 0 always
    ///   softplus(dt) + bias < 0 for bias below -softplus(dt)
    #[test]
    fn test_mamba2_dt_bias_is_applied_inside_softplus() {
        use numr::autograd::{var_add, var_softplus};

        let (client, device) = cpu_setup();

        // dt values around zero => softplus(dt) ~ 0.69; a -5.0 bias flips the sign
        // under the WRONG ordering but never under the correct one.
        let dt = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.25, -0.25, 0.5], &[4], &device).unwrap(),
            false,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[-5.0f32; 4], &[4], &device).unwrap(),
            false,
        );

        // Correct: bias inside.
        let inside = var_softplus(&var_add(&dt, &bias, &client).unwrap(), &client).unwrap();
        let inside_vals: Vec<f32> = inside.tensor().contiguous().unwrap().to_vec();

        // Wrong: bias outside.
        let outside = var_add(&var_softplus(&dt, &client).unwrap(), &bias, &client).unwrap();
        let outside_vals: Vec<f32> = outside.tensor().contiguous().unwrap().to_vec();

        assert!(
            inside_vals.iter().all(|v| *v > 0.0),
            "softplus(dt + bias) must stay positive, got {inside_vals:?}"
        );
        assert!(
            outside_vals.iter().all(|v| *v < 0.0),
            "test setup is degenerate: the wrong ordering should go negative here, got {outside_vals:?}"
        );

        // The arithmetic above only pins the semantics; now prove the LAYER uses it.
        //
        // Compare a strongly negative bias against a zero bias. dt scales the SSM
        // input term, so the two orderings move the output in OPPOSITE directions:
        //   correct  softplus(dt_raw - 5) ~= 0.007  -> much SMALLER than softplus(dt_raw) ~= 0.69
        //   wrong    softplus(dt_raw) - 5 ~= -4.31  -> |dt| much LARGER, and the decay
        //                                              exponent dt*A flips sign
        // Asserting the direction is robust; asserting a magnitude threshold is not,
        // because these tiny fixture weights never actually overflow.
        let magnitude = |bias: f32| -> f32 {
            let (mamba, _) = mamba2_with_dt_bias(bias);
            let x = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[0.05f32; 8 * 6], &[1, 6, 8], &device).unwrap(),
                false,
            );
            let out = mamba.forward(&client, &x).expect("forward must succeed");
            let vals: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "dt_bias={bias} produced non-finite output"
            );
            vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        };

        let neutral = magnitude(0.0);
        let suppressed = magnitude(-5.0);
        assert!(
            neutral > 0.0,
            "test setup is degenerate: zero-bias output is exactly zero"
        );
        assert!(
            suppressed < neutral * 0.5,
            "a strongly negative dt_bias must SHRINK the output (dt -> 0); \
             got {suppressed} vs {neutral} at zero bias — dt_bias is being added \
             outside softplus"
        );
    }
}
