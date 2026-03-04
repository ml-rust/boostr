//! Mamba2 layer: weights, struct definition, construction, and conv helpers.

use super::config::Mamba2Config;
use crate::error::{Error, Result};
use crate::nn::{Conv1d, Linear, RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Mamba2 layer implementing the SSD algorithm.
///
/// Forward: in_proj → split(z, xBC, dt) → conv1d(xBC) → silu → split(x,B,C) →
///          SSM(x, A, B, C, dt, D) → gate(silu(z)) → norm? → out_proj
pub struct Mamba2<R: Runtime> {
    pub(in crate::model::mamba::mamba2) config: Mamba2Config,
    pub(in crate::model::mamba::mamba2) in_proj: Linear<R>,
    pub(in crate::model::mamba::mamba2) conv1d: Conv1d<R>,
    pub(in crate::model::mamba::mamba2) out_proj: Linear<R>,
    pub(in crate::model::mamba::mamba2) a_log: Var<R>,
    pub(in crate::model::mamba::mamba2) dt_bias: Option<Var<R>>,
    pub(in crate::model::mamba::mamba2) d_param: Option<Var<R>>,
    pub(in crate::model::mamba::mamba2) norm: Option<RmsNorm<R>>,
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

    /// Prefill: full conv1d with causal padding, saves conv state.
    pub(in crate::model::mamba::mamba2) fn prefill_conv<C>(
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
                Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), x.device());
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
    pub(in crate::model::mamba::mamba2) fn decode_conv(
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
            Tensor::<R>::zeros(&[batch, conv_channels, 0], x.dtype(), x.device())
        };

        let mut new_state =
            Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), x.device());
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
