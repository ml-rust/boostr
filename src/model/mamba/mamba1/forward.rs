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
