//! Mamba3 trapezoidal SSM scan, with the group-to-head repetition and the
//! shared B/C RMS norm the scan's inputs pass through.

use super::layer::Mamba3;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::var_contiguous;
use numr::autograd::{
    Var, var_add, var_add_scalar, var_broadcast_to, var_cat, var_clamp, var_exp, var_matmul,
    var_mul, var_mul_scalar, var_narrow, var_neg, var_reshape,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, CompareOps, NormalizationOps, ReduceOps, ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

const A_MIN: f64 = 1e-6;
const A_MAX: f64 = 1e6;

impl<R: Runtime> Mamba3<R> {
    pub(in crate::model::mamba::mamba3) fn repeat_groups_to_heads(
        &self,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + ReduceOps<R>,
    {
        if self.config.ngroups == self.config.nheads {
            // `alias`, not `clone` — `Var::clone` mints a fresh TensorId.
            return Ok(input.alias());
        }
        if self.config.ngroups == 1 {
            let repeated = var_broadcast_to(
                input,
                &[batch, seq_len, self.config.nheads, self.config.d_state],
            )
            .map_err(Error::Numr)?;
            return var_contiguous(&repeated);
        }
        Err(Error::ModelError {
            reason: format!(
                "Mamba3 only supports ngroups=1 or ngroups=nheads, got {}",
                self.config.ngroups
            ),
        })
    }

    pub(in crate::model::mamba::mamba3) fn apply_bc_norm<C>(
        &self,
        client: &C,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let flat = var_reshape(
            input,
            &[batch * seq_len * self.config.nheads, self.config.d_state],
        )
        .map_err(Error::Numr)?;
        let normed = self.bc_norm.forward(client, &flat)?;
        var_reshape(
            &normed,
            &[batch, seq_len, self.config.nheads, self.config.d_state],
        )
        .map_err(Error::Numr)
    }

    pub(in crate::model::mamba::mamba3) fn trapezoidal_ssm_forward<C>(
        &self,
        client: &C,
        hidden_states: &Var<R>,
        b: &Var<R>,
        c: &Var<R>,
        dt: &Var<R>,
        lambda: &Var<R>,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        let shape = hidden_states.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let nheads = self.config.nheads;
        let headdim = shape[3];
        let d_state = self.config.d_state;
        let device = hidden_states.tensor().device();
        let dtype = hidden_states.tensor().dtype();

        let a_pos = var_exp(&self.a_log, client).map_err(Error::Numr)?;
        let a_pos = var_clamp(&a_pos, A_MIN, A_MAX, client).map_err(Error::Numr)?;
        let a = var_neg(&a_pos, client).map_err(Error::Numr)?;
        let a_broad = var_reshape(&a, &[1, nheads, 1, 1]).map_err(Error::Numr)?;

        let h_tensor = Tensor::<R>::zeros(&[batch, nheads, headdim, d_state], dtype, device)?;
        let mut h = Var::new(h_tensor, false);
        let prev_x_tensor = Tensor::<R>::zeros(&[batch, nheads, headdim], dtype, device)?;
        let prev_b_tensor = Tensor::<R>::zeros(&[batch, nheads, d_state], dtype, device)?;
        let mut prev_x = Var::new(prev_x_tensor, false);
        let mut prev_b = Var::new(prev_b_tensor, false);
        let mut outputs: Vec<Var<R>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = var_contiguous(
                &var_reshape(
                    &var_narrow(hidden_states, 1, t, 1).map_err(Error::Numr)?,
                    &[batch, nheads, headdim],
                )
                .map_err(Error::Numr)?,
            )?;
            let b_t = var_contiguous(
                &var_reshape(
                    &var_narrow(b, 1, t, 1).map_err(Error::Numr)?,
                    &[batch, nheads, d_state],
                )
                .map_err(Error::Numr)?,
            )?;
            let c_t = var_contiguous(
                &var_reshape(
                    &var_narrow(c, 1, t, 1).map_err(Error::Numr)?,
                    &[batch, nheads, d_state],
                )
                .map_err(Error::Numr)?,
            )?;
            let dt_t = var_contiguous(
                &var_reshape(
                    &var_narrow(dt, 1, t, 1).map_err(Error::Numr)?,
                    &[batch, nheads, 1, 1],
                )
                .map_err(Error::Numr)?,
            )?;
            let lambda_t = var_contiguous(
                &var_reshape(
                    &var_narrow(lambda, 1, t, 1).map_err(Error::Numr)?,
                    &[batch, nheads, 1, 1],
                )
                .map_err(Error::Numr)?,
            )?;

            let dt_a = var_mul(&dt_t, &a_broad, client).map_err(Error::Numr)?;
            let alpha = var_exp(&dt_a, client).map_err(Error::Numr)?;
            let one_minus_lambda = var_add_scalar(
                &var_mul_scalar(&lambda_t, -1.0, client).map_err(Error::Numr)?,
                1.0,
                client,
            )
            .map_err(Error::Numr)?;
            let beta = var_mul(&one_minus_lambda, &dt_t, client).map_err(Error::Numr)?;
            let beta = var_mul(&beta, &alpha, client).map_err(Error::Numr)?;
            let gamma = var_mul(&lambda_t, &dt_t, client).map_err(Error::Numr)?;

            h = var_mul(&alpha, &h, client).map_err(Error::Numr)?;

            let prev_x_col =
                var_reshape(&prev_x, &[batch, nheads, headdim, 1]).map_err(Error::Numr)?;
            let prev_b_row =
                var_reshape(&prev_b, &[batch, nheads, 1, d_state]).map_err(Error::Numr)?;
            let prev_term = var_mul(&prev_x_col, &prev_b_row, client).map_err(Error::Numr)?;
            let prev_term = var_mul(&beta, &prev_term, client).map_err(Error::Numr)?;

            let x_col = var_reshape(&x_t, &[batch, nheads, headdim, 1]).map_err(Error::Numr)?;
            let b_row = var_reshape(&b_t, &[batch, nheads, 1, d_state]).map_err(Error::Numr)?;
            let current_term = var_mul(&x_col, &b_row, client).map_err(Error::Numr)?;
            let current_term = var_mul(&gamma, &current_term, client).map_err(Error::Numr)?;

            h = var_add(&h, &prev_term, client).map_err(Error::Numr)?;
            h = var_add(&h, &current_term, client).map_err(Error::Numr)?;

            let c_col = var_reshape(&c_t, &[batch, nheads, d_state, 1]).map_err(Error::Numr)?;
            let y_t = var_matmul(&h, &c_col, client).map_err(Error::Numr)?;
            let mut y_t = var_reshape(&y_t, &[batch, nheads, headdim]).map_err(Error::Numr)?;

            if let Some(d_param) = self.d_param.as_ref() {
                let d_broad = var_reshape(d_param, &[1, nheads, 1]).map_err(Error::Numr)?;
                let d_x = var_mul(&d_broad, &x_t, client).map_err(Error::Numr)?;
                y_t = var_add(&y_t, &d_x, client).map_err(Error::Numr)?;
            }

            let y_t = var_reshape(&y_t, &[batch, 1, nheads, headdim]).map_err(Error::Numr)?;
            outputs.push(y_t);
            prev_x = x_t;
            prev_b = b_t;
        }

        let output_refs: Vec<&Var<R>> = outputs.iter().collect();
        var_cat(&output_refs, 1, client).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny_mamba3;
    use super::*;
    use crate::model::mamba::mamba3::config::Mamba3Config;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_trapezoidal_discretization_matches_f64_reference() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(2)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(true);
        let mut mamba = tiny_mamba3(config);
        mamba.a_log = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32.ln()], &[1], &device).unwrap(),
            false,
        );
        mamba.d_param = Some(Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[1], &device).unwrap(),
            false,
        ));

        let x_data = [1.0f32, 0.5, -0.25, 2.0, 0.75, -1.0];
        let b_data = [0.5f32, -0.2, 0.1, 0.3, -0.4, 0.25];
        let c_data = [1.0f32, 0.5, -0.25, 0.75, 0.6, -0.1];
        let dt_data = [0.2f32, 0.4, 0.3];
        let lambda_data = [0.7f32, 0.2, 0.9];

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 3, 1, 2], &device).unwrap(),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 3, 1, 2], &device).unwrap(),
            false,
        );
        let c = Var::new(
            Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 3, 1, 2], &device).unwrap(),
            false,
        );
        let dt = Var::new(
            Tensor::<CpuRuntime>::from_slice(&dt_data, &[1, 3, 1], &device).unwrap(),
            false,
        );
        let lambda = Var::new(
            Tensor::<CpuRuntime>::from_slice(&lambda_data, &[1, 3, 1], &device).unwrap(),
            false,
        );

        let out = mamba
            .trapezoidal_ssm_forward(&client, &x, &b, &c, &dt, &lambda)
            .unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        let expected =
            trapezoidal_reference(&x_data, &b_data, &c_data, &dt_data, &lambda_data, -0.5, 0.1);
        for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 2e-5,
                "idx={i}: actual={actual}, expected={expected}"
            );
        }
    }

    fn trapezoidal_reference(
        x: &[f32; 6],
        b: &[f32; 6],
        c: &[f32; 6],
        dt: &[f32; 3],
        lambda: &[f32; 3],
        a: f64,
        d: f64,
    ) -> Vec<f64> {
        let mut h = [[0.0f64; 2]; 2];
        let mut prev_x = [0.0f64; 2];
        let mut prev_b = [0.0f64; 2];
        let mut out = Vec::with_capacity(6);

        for t in 0..3 {
            let x_t = [x[2 * t] as f64, x[2 * t + 1] as f64];
            let b_t = [b[2 * t] as f64, b[2 * t + 1] as f64];
            let c_t = [c[2 * t] as f64, c[2 * t + 1] as f64];
            let dt_t = dt[t] as f64;
            let lambda_t = lambda[t] as f64;
            let alpha = (dt_t * a).exp();
            let beta = (1.0 - lambda_t) * dt_t * alpha;
            let gamma = lambda_t * dt_t;

            for dim in 0..2 {
                for state in 0..2 {
                    h[dim][state] = alpha * h[dim][state]
                        + beta * prev_x[dim] * prev_b[state]
                        + gamma * x_t[dim] * b_t[state];
                }
            }

            for dim in 0..2 {
                out.push(h[dim][0] * c_t[0] + h[dim][1] * c_t[1] + d * x_t[dim]);
            }
            prev_x = x_t;
            prev_b = b_t;
        }

        out
    }
}
