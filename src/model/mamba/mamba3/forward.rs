//! Mamba3 training/full-sequence forward pass. The per-stage helpers it calls
//! live beside it: `trapezoidal` (the SSM scan, group repetition, B/C norm),
//! `rope` (complex RoPE on B/C), `mimo` (the MIMO up/down projections).

use super::layer::Mamba3;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::var_contiguous;
use numr::autograd::{
    Var, var_add, var_cat, var_clamp, var_cumsum, var_mul, var_narrow, var_reshape, var_sigmoid,
    var_silu, var_softplus, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, CumulativeOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba3<R> {
    /// Full-sequence Var forward pass.
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
            + CumulativeOps<R>
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
        let bc_size = self.config.bc_size();

        // 1. Input projection: [B, S, d_model] -> [B, S, proj_dim]
        let projected = self.in_proj.forward(client, x)?;

        // 2. Split into gate, x, B, C, dt.
        let gate = var_contiguous(&var_narrow(&projected, 2, 0, d_inner).map_err(Error::Numr)?)?;
        let x_part =
            var_contiguous(&var_narrow(&projected, 2, d_inner, d_inner).map_err(Error::Numr)?)?;
        let mut b_proj =
            var_contiguous(&var_narrow(&projected, 2, 2 * d_inner, bc_size).map_err(Error::Numr)?)?;
        let mut c_proj = var_contiguous(
            &var_narrow(&projected, 2, 2 * d_inner + bc_size, bc_size).map_err(Error::Numr)?,
        )?;
        let dt = var_contiguous(
            &var_narrow(&projected, 2, 2 * d_inner + 2 * bc_size, self.config.nheads)
                .map_err(Error::Numr)?,
        )?;

        // 3. Optional causal Conv1D over concatenated [x, B, C], otherwise SiLU(x).
        let x_part = if let Some(ref conv1d) = self.conv1d {
            let xbc = var_cat(&[&x_part, &b_proj, &c_proj], 2, client).map_err(Error::Numr)?;
            let xbc_ncl = var_contiguous(&var_transpose(&xbc).map_err(Error::Numr)?)?;
            let conv_out = conv1d.forward(client, &xbc_ncl)?;
            let conv_out =
                var_contiguous(&var_narrow(&conv_out, 2, 0, seq_len).map_err(Error::Numr)?)?;
            let conv_out = var_contiguous(&var_transpose(&conv_out).map_err(Error::Numr)?)?;
            let conv_out = var_silu(&conv_out, client).map_err(Error::Numr)?;
            let x_conv =
                var_contiguous(&var_narrow(&conv_out, 2, 0, d_inner).map_err(Error::Numr)?)?;
            b_proj =
                var_contiguous(&var_narrow(&conv_out, 2, d_inner, bc_size).map_err(Error::Numr)?)?;
            c_proj = var_contiguous(
                &var_narrow(&conv_out, 2, d_inner + bc_size, bc_size).map_err(Error::Numr)?,
            )?;
            x_conv
        } else {
            var_silu(&x_part, client).map_err(Error::Numr)?
        };

        // 4. Trapezoidal mixing parameter lambda: [B, S, H].
        let lambda =
            var_sigmoid(&self.lambda_proj.forward(client, x)?, client).map_err(Error::Numr)?;

        // 5. B/C projection reshape, repeat from groups to heads, RMS-normalize, and bias.
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
        let b_proj = self.repeat_groups_to_heads(&b_proj, batch, seq_len)?;
        let c_proj = self.repeat_groups_to_heads(&c_proj, batch, seq_len)?;
        let b_proj = self.apply_bc_norm(client, &b_proj, batch, seq_len)?;
        let c_proj = self.apply_bc_norm(client, &c_proj, batch, seq_len)?;
        let b_bias = var_reshape(
            &self.b_bias,
            &[1, 1, self.config.nheads, self.config.d_state],
        )
        .map_err(Error::Numr)?;
        let c_bias = var_reshape(
            &self.c_bias,
            &[1, 1, self.config.nheads, self.config.d_state],
        )
        .map_err(Error::Numr)?;
        let mut b_proj = var_add(&b_proj, &b_bias, client).map_err(Error::Numr)?;
        let mut c_proj = var_add(&c_proj, &c_bias, client).map_err(Error::Numr)?;

        // 6. dt = clamp(softplus(dt + dt_bias), min, max).
        let mut dt = if let Some(ref bias) = self.dt_bias {
            var_add(&dt, bias, client).map_err(Error::Numr)?
        } else {
            dt
        };
        if self.config.dt_softplus {
            dt = var_softplus(&dt, client).map_err(Error::Numr)?;
        }
        let dt = var_clamp(
            &dt,
            self.config.time_step_min,
            self.config.time_step_max,
            client,
        )
        .map_err(Error::Numr)?;

        // 7. Optional complex RoPE over B/C state channels.
        if self.config.use_complex_rope {
            let theta_proj = self.theta_proj.as_ref().ok_or_else(|| Error::ModelError {
                reason: "Mamba3 complex_rope requires theta_proj".into(),
            })?;
            let theta = theta_proj.forward(client, x)?;
            let theta = var_reshape(
                &theta,
                &[batch, seq_len, self.config.nheads, self.config.d_state / 2],
            )
            .map_err(Error::Numr)?;
            let dt_expanded =
                var_reshape(&dt, &[batch, seq_len, self.config.nheads, 1]).map_err(Error::Numr)?;
            let theta_scaled = var_mul(&theta, &dt_expanded, client).map_err(Error::Numr)?;
            let angles = var_cumsum(&theta_scaled, 1, client).map_err(Error::Numr)?;
            b_proj = self.apply_rope(client, &b_proj, &angles)?;
            c_proj = self.apply_rope(client, &c_proj, &angles)?;
        }

        // 8. Reshape x to heads and optionally apply MIMO up-projection.
        let x_ssm = var_reshape(
            &x_part,
            &[batch, seq_len, self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)?;
        let x_for_ssm = self.apply_mimo_up(client, &x_ssm, batch, seq_len)?;

        // 9. Trapezoidal SSM recurrence.
        let y = self.trapezoidal_ssm_forward(client, &x_for_ssm, &b_proj, &c_proj, &dt, &lambda)?;

        // 10. Optional MIMO down-projection.
        let y = self.apply_mimo_down(client, &y, batch, seq_len)?;

        // 11. Gated RMS normalization followed by output projection.
        let y = var_reshape(&y, &[batch, seq_len, d_inner]).map_err(Error::Numr)?;
        let gate = var_silu(&gate, client).map_err(Error::Numr)?;
        let gated = var_mul(&y, &gate, client).map_err(Error::Numr)?;
        let scan_output = self.norm.forward(client, &gated)?;
        self.out_proj.forward(client, &scan_output)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::model::mamba::mamba3::config::Mamba3Config;
    use crate::model::mamba::mamba3::layer::Mamba3Weights;
    use crate::nn::{Linear, RmsNorm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    pub(in super::super) fn linear_with_value(
        rows: usize,
        cols: usize,
        value: f32,
    ) -> Linear<CpuRuntime> {
        let (_, device) = cpu_setup();
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&vec![value; rows * cols], &[rows, cols], &device)
                .unwrap(),
            None,
            false,
        )
    }

    pub(in super::super) fn tiny_mamba3(config: Mamba3Config) -> Mamba3<CpuRuntime> {
        let (_, device) = cpu_setup();
        let in_proj = linear_with_value(config.proj_dim(), config.d_model, 0.0);
        let out_proj = linear_with_value(config.d_model, config.d_inner(), 0.0);
        let lambda_proj = linear_with_value(config.nheads, config.d_model, 0.0);
        let theta_proj = if config.use_complex_rope {
            Some(linear_with_value(
                config.nheads * (config.d_state / 2),
                config.d_model,
                0.0,
            ))
        } else {
            None
        };
        let mimo_x_up = if config.mimo_rank > 0 {
            Some(linear_with_value(
                config.headdim * config.mimo_rank,
                config.headdim,
                0.0,
            ))
        } else {
            None
        };
        let mimo_x_down = if config.mimo_rank > 0 {
            Some(linear_with_value(
                config.headdim,
                config.headdim * config.mimo_rank,
                0.0,
            ))
        } else {
            None
        };

        let weights = Mamba3Weights {
            in_proj,
            out_proj,
            lambda_proj,
            theta_proj,
            b_bias: Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.nheads * config.d_state],
                &[config.nheads, config.d_state],
                &device,
            )
            .unwrap(),
            c_bias: Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.nheads * config.d_state],
                &[config.nheads, config.d_state],
                &device,
            )
            .unwrap(),
            dt_bias: if config.use_dt_bias {
                Some(
                    Tensor::<CpuRuntime>::from_slice(
                        &vec![0.0f32; config.nheads],
                        &[config.nheads],
                        &device,
                    )
                    .unwrap(),
                )
            } else {
                None
            },
            a_log: Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.nheads],
                &[config.nheads],
                &device,
            )
            .unwrap(),
            d_param: if config.use_d {
                Some(
                    Tensor::<CpuRuntime>::from_slice(
                        &vec![0.0f32; config.nheads],
                        &[config.nheads],
                        &device,
                    )
                    .unwrap(),
                )
            } else {
                None
            },
            bc_norm: RmsNorm::new(
                Tensor::<CpuRuntime>::from_slice(
                    &vec![1.0f32; config.d_state],
                    &[config.d_state],
                    &device,
                )
                .unwrap(),
                1e-6,
                false,
            ),
            norm: RmsNorm::new(
                Tensor::<CpuRuntime>::from_slice(
                    &vec![1.0f32; config.d_inner()],
                    &[config.d_inner()],
                    &device,
                )
                .unwrap(),
                1e-6,
                false,
            ),
            conv1d: None,
            mimo_x_up,
            mimo_x_down,
        };
        Mamba3::new(config, weights, false)
    }

    #[test]
    fn test_mamba3_forward_shape() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(4)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);
        let mamba = tiny_mamba3(config);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 3, 4], &device).unwrap(),
            false,
        );

        let out = mamba.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 3, 4]);
    }

    #[test]
    fn test_mamba3_forward_invalid_input() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(4)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);
        let mamba = tiny_mamba3(config);

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
}
