//! Mamba3 training/full-sequence forward pass.

use super::layer::Mamba3;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::var_contiguous;
use numr::autograd::{
    Var, var_add, var_add_scalar, var_broadcast_to, var_cat, var_clamp, var_cos, var_cumsum,
    var_exp, var_matmul, var_mul, var_mul_scalar, var_narrow, var_neg, var_reshape, var_sigmoid,
    var_silu, var_sin, var_softplus, var_sub, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, CumulativeOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

const A_MIN: f64 = 1e-6;
const A_MAX: f64 = 1e6;

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

    pub(in crate::model::mamba::mamba3) fn apply_rope<C>(
        &self,
        client: &C,
        input: &Var<R>,
        angles: &Var<R>,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let nheads = shape[2];
        let d_state = shape[3];
        if !d_state.is_multiple_of(2) {
            return Err(Error::ModelError {
                reason: format!("complex RoPE requires even state dimension, got {d_state}"),
            });
        }
        let half = d_state / 2;
        let pairs = var_reshape(input, &[batch, seq_len, nheads, half, 2]).map_err(Error::Numr)?;
        let real_pair = var_contiguous(&var_narrow(&pairs, 4, 0, 1).map_err(Error::Numr)?)?;
        let real = var_reshape(&real_pair, &[batch, seq_len, nheads, half]).map_err(Error::Numr)?;
        let imag_pair = var_contiguous(&var_narrow(&pairs, 4, 1, 1).map_err(Error::Numr)?)?;
        let imag = var_reshape(&imag_pair, &[batch, seq_len, nheads, half]).map_err(Error::Numr)?;
        let cos = var_cos(angles, client).map_err(Error::Numr)?;
        let sin = var_sin(angles, client).map_err(Error::Numr)?;

        let real_cos = var_mul(&real, &cos, client).map_err(Error::Numr)?;
        let imag_sin = var_mul(&imag, &sin, client).map_err(Error::Numr)?;
        let real_new = var_sub(&real_cos, &imag_sin, client).map_err(Error::Numr)?;

        let real_sin = var_mul(&real, &sin, client).map_err(Error::Numr)?;
        let imag_cos = var_mul(&imag, &cos, client).map_err(Error::Numr)?;
        let imag_new = var_add(&real_sin, &imag_cos, client).map_err(Error::Numr)?;

        let real_new =
            var_reshape(&real_new, &[batch, seq_len, nheads, half, 1]).map_err(Error::Numr)?;
        let imag_new =
            var_reshape(&imag_new, &[batch, seq_len, nheads, half, 1]).map_err(Error::Numr)?;
        let rotated = var_cat(&[&real_new, &imag_new], 4, client).map_err(Error::Numr)?;
        var_reshape(&rotated, &[batch, seq_len, nheads, d_state]).map_err(Error::Numr)
    }

    pub(in crate::model::mamba::mamba3) fn apply_mimo_up<C>(
        &self,
        client: &C,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        if self.config.mimo_rank == 0 {
            // `alias`, not `clone` — `Var::clone` mints a fresh TensorId.
            return Ok(input.alias());
        }
        let up_proj = self.mimo_x_up.as_ref().ok_or_else(|| Error::ModelError {
            reason: "Mamba3 mimo_rank > 0 requires mimo_x_up".into(),
        })?;
        let flat = var_reshape(
            input,
            &[batch * seq_len * self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)?;
        let up = up_proj.forward(client, &flat)?;
        var_reshape(
            &up,
            &[
                batch,
                seq_len,
                self.config.nheads,
                self.config.headdim * self.config.mimo_rank,
            ],
        )
        .map_err(Error::Numr)
    }

    pub(in crate::model::mamba::mamba3) fn apply_mimo_down<C>(
        &self,
        client: &C,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        if self.config.mimo_rank == 0 {
            // `alias`, not `clone` — `Var::clone` mints a fresh TensorId.
            return Ok(input.alias());
        }
        let down_proj = self.mimo_x_down.as_ref().ok_or_else(|| Error::ModelError {
            reason: "Mamba3 mimo_rank > 0 requires mimo_x_down".into(),
        })?;
        let mimo_dim = self.config.headdim * self.config.mimo_rank;
        let flat = var_reshape(input, &[batch * seq_len * self.config.nheads, mimo_dim])
            .map_err(Error::Numr)?;
        let down = down_proj.forward(client, &flat)?;
        var_reshape(
            &down,
            &[batch, seq_len, self.config.nheads, self.config.headdim],
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
