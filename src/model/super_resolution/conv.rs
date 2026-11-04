//! [`SrConv`]: one stride-1 conv with bias and an optional PReLU slope,
//! run through numr's fused conv.

use crate::error::{Error, Result};
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::layout::{SrExecution, weight_for_layout};
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{Conv2dConfig, ConvActivation, ConvResidual, FusedConvOps, PaddingMode};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A stride-1 conv with `Same` padding, in the layout of the
/// [`SrExecution`] it loads with.
///
/// The weight sits in the layout's order: OIHW for `Nchw`, OHWI for `Nhwc`.
/// The bias is `[C_out]`. A PReLU slope, when the network has one after
/// this conv, is `[C_out]`. Every tensor has the execution dtype.
pub struct SrConv<R: Runtime> {
    name: String,
    weight: Tensor<R>,
    bias: Tensor<R>,
    prelu: Option<Tensor<R>>,
    config: Conv2dConfig,
    in_channels: usize,
    out_channels: usize,
}

impl<R: Runtime<DType = DType>> SrConv<R> {
    /// Take `{conv}.weight` `[out, in, k, k]` and `{conv}.bias` `[out]`
    /// under `vb`'s prefix. With `prelu = Some(key)`, also take
    /// `{key}.weight` `[out]`.
    ///
    /// Each tensor is cast to `exec.dtype`. The weight is then permuted to
    /// the layout's order. `kernel` must be odd, so `Same` padding is
    /// `kernel / 2` on every side, as PyTorch's `padding=kernel // 2`.
    ///
    /// # Errors
    ///
    /// - A missing key. The error names the full key.
    /// - A shape other than the expected one. The error names the key and
    ///   both shapes.
    /// - An even `kernel`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_varbuilder<C: SrClient<R>>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        conv: &str,
        prelu: Option<&str>,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        exec: &SrExecution,
    ) -> Result<Self> {
        let name = vb.full_name(conv);
        if kernel.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "kernel",
                reason: format!(
                    "conv '{name}' has even kernel {kernel}; `Same` padding matches PyTorch only for an odd kernel"
                ),
            });
        }
        let weight = take_checked(
            client,
            vb,
            &format!("{conv}.weight"),
            &[out_channels, in_channels, kernel, kernel],
            exec.dtype,
        )?;
        let bias = take_checked(
            client,
            vb,
            &format!("{conv}.bias"),
            &[out_channels],
            exec.dtype,
        )?;
        let prelu = prelu
            .map(|key| {
                take_checked(
                    client,
                    vb,
                    &format!("{key}.weight"),
                    &[out_channels],
                    exec.dtype,
                )
            })
            .transpose()?;
        Ok(Self {
            name,
            weight: weight_for_layout(&weight, exec.layout)?,
            bias,
            prelu,
            config: exec.conv_config(PaddingMode::Same),
            in_channels,
            out_channels,
        })
    }

    /// Run the conv and return a new tensor.
    ///
    /// `act` runs after the bias add. Pass [`Self::activation`] for the
    /// stored PReLU. `residual` adds last: `residual + scale * act(conv)`.
    pub fn forward<C: FusedConvOps<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
        act: ConvActivation<'_, R>,
        residual: Option<ConvResidual<'_, R>>,
    ) -> Result<Tensor<R>> {
        client
            .conv2d_fused(
                x,
                &self.weight,
                Some(&self.bias),
                &self.config,
                act,
                residual,
            )
            .map_err(|e| self.forward_error(x, e))
    }

    /// Run the conv and write the result into `out`.
    ///
    /// `out` can be a strided view, such as a channel slice of a larger
    /// buffer. It must not overlap `x` or `residual`. See
    /// [`FusedConvOps::conv2d_fused_into`] for the aliasing rule.
    pub fn forward_into<C: FusedConvOps<R>>(
        &self,
        client: &C,
        out: &Tensor<R>,
        x: &Tensor<R>,
        act: ConvActivation<'_, R>,
        residual: Option<ConvResidual<'_, R>>,
    ) -> Result<()> {
        client
            .conv2d_fused_into(
                out,
                x,
                &self.weight,
                Some(&self.bias),
                &self.config,
                act,
                residual,
            )
            .map_err(|e| self.forward_error(x, e))
    }

    /// The stored PReLU as a fused activation, else [`ConvActivation::None`].
    pub fn activation(&self) -> ConvActivation<'_, R> {
        match &self.prelu {
            Some(slope) => ConvActivation::Prelu(slope),
            None => ConvActivation::None,
        }
    }

    /// Full checkpoint key prefix of the conv, such as `body.4`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Weight in the layout's order.
    pub fn weight(&self) -> &Tensor<R> {
        &self.weight
    }

    /// Bias `[C_out]`.
    pub fn bias(&self) -> &Tensor<R> {
        &self.bias
    }

    /// PReLU slope `[C_out]`, when the network has one after this conv.
    pub fn prelu(&self) -> Option<&Tensor<R>> {
        self.prelu.as_ref()
    }

    /// Input channel count.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Output channel count.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn forward_error(&self, x: &Tensor<R>, e: numr::error::Error) -> Error {
        Error::model_op(
            format!(
                "conv '{}' ({} -> {} channels, {:?}) on input {:?} {:?}: {e}",
                self.name,
                self.in_channels,
                self.out_channels,
                self.config.layout,
                x.shape(),
                x.dtype()
            ),
            e,
        )
    }
}

/// Take `name` from `vb`, check its shape, and cast it to `dtype`.
fn take_checked<R, C>(
    client: &C,
    vb: &mut VarBuilder<'_, R>,
    name: &str,
    expected: &[usize],
    dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: SrClient<R>,
{
    let full = vb.full_name(name);
    let tensor = vb.take_tensor(name).map_err(|e| Error::ModelError {
        reason: format!(
            "weight '{full}' not loaded: {e}; check the checkpoint matches the network config"
        ),
    })?;
    if tensor.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "weight '{full}' has shape {:?}, expected {expected:?}; check the network config against the checkpoint",
                tensor.shape()
            ),
        });
    }
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        client.cast(&tensor, dtype).map_err(|e| Error::ModelError {
            reason: format!(
                "casting weight '{full}' from {:?} to {dtype:?}: {e}; F16 and BF16 on CPU need the `f16` feature",
                tensor.dtype()
            ),
        })
    }
}
