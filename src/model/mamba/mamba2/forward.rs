//! Mamba2 training and inference forward passes.

use super::layer::Mamba2;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::{SsmInput, var_contiguous};
use numr::autograd::{
    Var, var_add, var_exp, var_mul, var_narrow, var_neg, var_reshape, var_silu, var_softplus,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
            + ShapeOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>,
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
        let z = var_contiguous(&var_narrow(&projected, 2, 0, d_inner).map_err(Error::Numr)?);
        let xbc =
            var_contiguous(&var_narrow(&projected, 2, d_inner, xbc_len).map_err(Error::Numr)?);
        let dt = var_contiguous(
            &var_narrow(&projected, 2, d_inner + xbc_len, self.config.nheads)
                .map_err(Error::Numr)?,
        );

        // 3. Causal conv1d on xBC
        let xbc_ncl = var_contiguous(&var_transpose(&xbc).map_err(Error::Numr)?);
        let xbc_conv = self.conv1d.forward(client, &xbc_ncl)?;
        let xbc = var_contiguous(&var_transpose(&xbc_conv).map_err(Error::Numr)?);

        // 4. SiLU activation
        let xbc = var_silu(&xbc, client).map_err(Error::Numr)?;

        // 5. Split xBC into x_ssm, B, C
        let x_ssm = var_contiguous(&var_narrow(&xbc, 2, 0, d_inner).map_err(Error::Numr)?);
        let b_proj =
            var_contiguous(&var_narrow(&xbc, 2, d_inner, n_groups_d_state).map_err(Error::Numr)?);
        let c_proj = var_contiguous(
            &var_narrow(&xbc, 2, d_inner + n_groups_d_state, n_groups_d_state)
                .map_err(Error::Numr)?,
        );

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

        // 8. Process dt
        let mut dt = dt;
        if self.config.dt_softplus {
            dt = var_softplus(&dt, client).map_err(Error::Numr)?;
        }
        if let Some(ref bias) = self.dt_bias {
            dt = var_add(&dt, bias, client).map_err(Error::Numr)?;
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

    /// Inference forward pass on raw tensors (no autograd overhead).
    ///
    /// Takes and updates per-layer SSM state (hidden + conv buffer).
    /// For prefill (seq_len > 1): processes full sequence via conv1d + sequential scan.
    /// For decode (seq_len = 1): uses cached conv state for conv step + single SSM step.
    ///
    /// x: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
    pub fn forward_inference<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
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
            + ShapeOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.config.d_model {
            return Err(Error::ModelError {
                reason: format!(
                    "expected [batch, seq_len, {}], got shape {:?}",
                    self.config.d_model, shape
                ),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let d_inner = self.config.d_inner();
        let n_groups_d_state = self.config.ngroups * self.config.d_state;

        // 1. Input projection: [B, S, d_model] -> [B, S, proj_dim]
        let x_var = Var::new(x.clone(), false);
        let projected = self.in_proj.forward(client, &x_var)?;
        let projected = projected.tensor().clone().contiguous();

        // 2. Split into z, xBC, dt
        let xbc_len = d_inner + 2 * n_groups_d_state;
        let z = projected
            .narrow(2, 0, d_inner)
            .map_err(Error::Numr)?
            .contiguous();
        let xbc = projected
            .narrow(2, d_inner, xbc_len)
            .map_err(Error::Numr)?
            .contiguous();
        let dt = projected
            .narrow(2, d_inner + xbc_len, self.config.nheads)
            .map_err(Error::Numr)?
            .contiguous();

        // 3. Causal conv1d on xBC — transpose NLC -> NCL
        let xbc_ncl = xbc.transpose(-1, -2).map_err(Error::Numr)?.contiguous();

        let xbc_conv = if seq_len > 1 {
            self.prefill_conv(client, &xbc_ncl, seq_len, batch, x, state)?
        } else {
            self.decode_conv(&xbc_ncl, batch, x, state)?
        };

        // Back to NLC
        let xbc = xbc_conv
            .transpose(-1, -2)
            .map_err(Error::Numr)?
            .contiguous();

        // 4. SiLU activation
        let xbc = xbc.silu().map_err(Error::Numr)?;

        // 5. Split xBC into x_ssm, B, C
        let x_ssm = xbc.narrow(2, 0, d_inner).map_err(Error::Numr)?.contiguous();
        let b_proj = xbc
            .narrow(2, d_inner, n_groups_d_state)
            .map_err(Error::Numr)?
            .contiguous();
        let c_proj = xbc
            .narrow(2, d_inner + n_groups_d_state, n_groups_d_state)
            .map_err(Error::Numr)?
            .contiguous();

        // 6. Reshape for SSM
        let x_ssm = x_ssm
            .reshape(&[batch, seq_len, self.config.nheads, self.config.headdim])
            .map_err(Error::Numr)?;
        let b_proj = b_proj
            .reshape(&[batch, seq_len, self.config.ngroups, self.config.d_state])
            .map_err(Error::Numr)?;
        let c_proj = c_proj
            .reshape(&[batch, seq_len, self.config.ngroups, self.config.d_state])
            .map_err(Error::Numr)?;

        // 7. Compute A = -exp(A_log)
        let a = self.a_log.tensor().exp().map_err(Error::Numr)?;
        let neg_one = Tensor::<R>::from_slice(&[-1.0f32], &[1], x.device());
        let a = a.mul(&neg_one).map_err(Error::Numr)?;

        // 8. Process dt
        let mut dt = dt;
        if self.config.dt_softplus {
            dt = client.softplus(&dt).map_err(Error::Numr)?;
        }
        if let Some(ref bias) = self.dt_bias {
            dt = dt.add(bias.tensor()).map_err(Error::Numr)?;
        }

        // 9. SSM forward
        let d_tensor = self.d_param.as_ref().map(|d| d.tensor().clone());
        let ssm_input = crate::model::mamba::ssm::SsmInferenceInput {
            x: &x_ssm,
            a: &a,
            b: &b_proj,
            c: &c_proj,
            d_param: d_tensor.as_ref(),
            dt: &dt,
            config: &self.config,
        };

        let (out, final_h) = crate::model::mamba::ssm::ssm_forward_sequential_inference(
            client,
            &ssm_input,
            state.h(),
        )?;
        state.update_h(final_h);

        // 10. Reshape back: [B, S, nheads, headdim] -> [B, S, d_inner]
        let out = out
            .reshape(&[batch, seq_len, d_inner])
            .map_err(Error::Numr)?;

        // 11. Gate: out = out * silu(z)
        let z_gate = z.silu().map_err(Error::Numr)?;
        let out = out.mul(&z_gate).map_err(Error::Numr)?;

        // 12. Optional norm (use Var path since RmsNorm takes Var)
        let out_var = Var::new(out, false);
        let out_var = if let Some(ref norm) = self.norm {
            norm.forward(client, &out_var)?
        } else {
            out_var
        };

        // 13. Output projection
        let out_var = self.out_proj.forward(client, &out_var)?;
        Ok(out_var.tensor().clone())
    }
}
