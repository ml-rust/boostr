//! Mamba2 layer: weights, construction, and forward passes.

use super::config::Mamba2Config;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::{SsmInput, var_contiguous};
use crate::nn::{Conv1d, Linear, RmsNorm, VarBuilder};
use numr::autograd::{
    Var, var_add, var_exp, var_mul, var_narrow, var_neg, var_reshape, var_silu, var_softplus,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, PaddingMode, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Mamba2 layer implementing the SSD algorithm.
///
/// Forward: in_proj → split(z, xBC, dt) → conv1d(xBC) → silu → split(x,B,C) →
///          SSM(x, A, B, C, dt, D) → gate(silu(z)) → norm? → out_proj
pub struct Mamba2<R: Runtime> {
    config: Mamba2Config,
    in_proj: Linear<R>,
    conv1d: Conv1d<R>,
    out_proj: Linear<R>,
    a_log: Var<R>,
    dt_bias: Option<Var<R>>,
    d_param: Option<Var<R>>,
    norm: Option<RmsNorm<R>>,
}

/// Bundled weight tensors for Mamba2 construction (reduces parameter count).
pub struct Mamba2Weights<R: Runtime> {
    pub in_proj: Linear<R>,
    pub conv1d: Conv1d<R>,
    pub out_proj: Linear<R>,
    pub a_log: Tensor<R>,
    pub dt_bias: Option<Tensor<R>>,
    pub d_param: Option<Tensor<R>>,
    pub norm: Option<RmsNorm<R>>,
}

impl<R: Runtime> Mamba2<R> {
    /// Create a new Mamba2 layer from config and weights.
    pub fn new(config: Mamba2Config, weights: Mamba2Weights<R>, trainable: bool) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            conv1d: weights.conv1d,
            out_proj: weights.out_proj,
            a_log: Var::new(weights.a_log, trainable),
            dt_bias: weights.dt_bias.map(|t| Var::new(t, trainable)),
            d_param: weights.d_param.map(|t| Var::new(t, trainable)),
            norm: weights.norm,
        }
    }

    /// Load from a VarBuilder (HuggingFace Mamba2 naming).
    pub fn from_varbuilder(
        config: &Mamba2Config,
        vb: &mut VarBuilder<R>,
        trainable: bool,
    ) -> Result<Self> {
        config.validate()?;
        let conv_channels = config.conv_channels();

        let in_proj_weight = vb.take_tensor("in_proj.weight")?;
        let in_proj_bias = if vb.contains("in_proj.bias") {
            Some(vb.take_tensor("in_proj.bias")?)
        } else {
            None
        };
        let in_proj = Linear::new(in_proj_weight, in_proj_bias, trainable);

        let conv_weight = vb.take_tensor("conv1d.weight")?;
        let conv_bias = if vb.contains("conv1d.bias") {
            Some(vb.take_tensor("conv1d.bias")?)
        } else {
            None
        };
        let causal_pad = config.d_conv - 1;
        let conv1d = Conv1d::new(
            conv_weight,
            conv_bias,
            1,
            PaddingMode::Custom(causal_pad, 0, 0, 0),
            1,
            conv_channels,
            trainable,
        );

        let out_proj_weight = vb.take_tensor("out_proj.weight")?;
        let out_proj_bias = if vb.contains("out_proj.bias") {
            Some(vb.take_tensor("out_proj.bias")?)
        } else {
            None
        };
        let out_proj = Linear::new(out_proj_weight, out_proj_bias, trainable);

        let a_log = vb.take_tensor("A_log")?;
        let dt_bias = if config.use_dt_bias && vb.contains("dt_bias") {
            Some(vb.take_tensor("dt_bias")?)
        } else {
            None
        };
        let d_param = if config.use_d && vb.contains("D") {
            Some(vb.take_tensor("D")?)
        } else {
            None
        };
        let norm = if vb.contains("norm.weight") {
            let norm_weight = vb.take_tensor("norm.weight")?;
            Some(RmsNorm::new(norm_weight, 1e-5, trainable))
        } else {
            None
        };

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias,
            d_param,
            norm,
        };
        Ok(Self::new(config.clone(), weights, trainable))
    }

    pub fn config(&self) -> &Mamba2Config {
        &self.config
    }

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
        let neg_one = Tensor::<R>::from_slice(&[-1.0f32], &[1], &x.device());
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

    /// Prefill: full conv1d with causal padding, saves conv state.
    fn prefill_conv<C>(
        &self,
        client: &C,
        xbc_ncl: &Tensor<R>,
        seq_len: usize,
        batch: usize,
        x: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ConvOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R>,
    {
        let conv_out = self.conv1d.forward_inference(client, xbc_ncl)?;
        let conv_out = conv_out
            .narrow(2, 0, seq_len)
            .map_err(Error::Numr)?
            .contiguous();

        let conv_window = self.config.d_conv - 1;
        if seq_len >= conv_window {
            let tail = xbc_ncl
                .narrow(2, seq_len - conv_window, conv_window)
                .map_err(Error::Numr)?
                .contiguous();
            state.update_conv_state(tail);
        } else {
            let conv_channels = self.config.conv_channels();
            let mut new_conv =
                Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), &x.device());
            let offset = conv_window - seq_len;
            if state.is_initialized() && offset > 0 {
                let old_tail = state
                    .conv_state()
                    .narrow(2, conv_window - offset, offset)
                    .map_err(Error::Numr)?
                    .contiguous();
                new_conv = new_conv
                    .slice_assign(&old_tail, 2, 0)
                    .map_err(Error::Numr)?;
            }
            new_conv = new_conv
                .slice_assign(xbc_ncl, 2, offset)
                .map_err(Error::Numr)?;
            state.update_conv_state(new_conv);
        }

        Ok(conv_out)
    }

    /// Decode (seq_len=1): manual conv step using cached state.
    fn decode_conv(
        &self,
        xbc_ncl: &Tensor<R>,
        batch: usize,
        x: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R>,
    {
        let conv_window = self.config.d_conv - 1;
        let conv_channels = self.config.conv_channels();

        let old_state = if conv_window > 1 {
            state
                .conv_state()
                .narrow(2, 1, conv_window - 1)
                .map_err(Error::Numr)?
                .contiguous()
        } else {
            Tensor::<R>::zeros(&[batch, conv_channels, 0], x.dtype(), &x.device())
        };

        let mut new_state =
            Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), &x.device());
        if conv_window > 1 {
            new_state = new_state
                .slice_assign(&old_state, 2, 0)
                .map_err(Error::Numr)?;
        }
        new_state = new_state
            .slice_assign(xbc_ncl, 2, conv_window - 1)
            .map_err(Error::Numr)?;
        state.update_conv_state(new_state.clone());

        let conv_input_refs = [&new_state, xbc_ncl];
        let conv_input = Tensor::cat(&conv_input_refs, 2).map_err(Error::Numr)?;

        let conv_weight = self.conv1d.weight().tensor();
        let conv_bias = self.conv1d.bias().map(|b| b.tensor());
        // The full conv input is [B, C, d_conv], which with valid padding gives [B, C, 1]
        conv_input
            .conv1d(
                conv_weight,
                conv_bias,
                1,
                PaddingMode::Valid,
                1,
                self.config.conv_channels(),
            )
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 328], &[proj_dim, 8], &device),
            None,
            false,
        );
        let conv1d = Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device),
            None,
            1,
            PaddingMode::Custom(3, 0, 0, 0),
            1,
            conv_channels,
            false,
        );
        let out_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 128], &[8, d_inner], &device),
            None,
            false,
        );
        let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[config.nheads], &device);

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
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[1, 4, 8], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[1, 8], &device),
            false,
        );
        assert!(mamba.forward(&client, &x_2d).is_err());

        // Wrong d_model should fail
        let x_wrong = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 4, 3], &device),
            false,
        );
        assert!(mamba.forward(&client, &x_wrong).is_err());
    }
}
