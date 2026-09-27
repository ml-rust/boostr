//! Building blocks for [`super::hifigan::VitsHifiGan`]: the residual conv
//! block and the transposed-conv upsample stage.

use crate::error::{Error, Result};
use crate::model::audio::vits::weight_norm_conv::load_weight_normed_conv1d;
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ConvOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One HiFi-GAN residual block: `dilations.len()` `(conv1, conv2)` pairs,
/// each gated by leaky-ReLU on entry, with a residual add every iteration.
/// `conv1` carries the stage's dilation; `conv2` is always dilation `1`.
/// Exact port of HF `HifiGanResidualBlock`.
pub(super) struct HifiGanResidualBlock<R: Runtime> {
    convs1: Vec<Conv1d<R>>,
    convs2: Vec<Conv1d<R>>,
    leaky_relu_slope: f64,
}

impl<R: Runtime<DType = DType>> HifiGanResidualBlock<R> {
    /// Reads `convs1.{0..dilations.len()}.*` and `convs2.{0..dilations.len()}.*`
    /// under `vb`'s current prefix. `get_padding(k, d) = (k*d - d) / 2` —
    /// HF's `HifiGanResidualBlock.get_padding`, "same"-length padding for
    /// stride-1 dilated convs.
    pub(super) fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        kernel_size: usize,
        dilations: &[usize],
        leaky_relu_slope: f64,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (i, &dilation) in dilations.iter().enumerate() {
            let padding1 = (kernel_size * dilation - dilation) / 2;
            convs1.push(load_weight_normed_conv1d(
                client,
                vb,
                &format!("convs1.{i}"),
                1,
                PaddingMode::conv1d(padding1, padding1),
                dilation,
                1,
            )?);
            let padding2 = (kernel_size - 1) / 2;
            convs2.push(load_weight_normed_conv1d(
                client,
                vb,
                &format!("convs2.{i}"),
                1,
                PaddingMode::conv1d(padding2, padding2),
                1,
                1,
            )?);
        }
        Ok(Self {
            convs1,
            convs2,
            leaky_relu_slope,
        })
    }

    pub(super) fn forward<C>(&self, client: &C, hidden_states: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + ActivationOps<R> + BinaryOps<R>,
    {
        let mut hidden_states = hidden_states.clone();
        for (conv1, conv2) in self.convs1.iter().zip(self.convs2.iter()) {
            let residual = hidden_states.clone();
            let out = client
                .leaky_relu(&hidden_states, self.leaky_relu_slope)
                .map_err(Error::Numr)?;
            let out = conv1.forward_inference(client, &out)?;
            let out = client
                .leaky_relu(&out, self.leaky_relu_slope)
                .map_err(Error::Numr)?;
            let out = conv2.forward_inference(client, &out)?;
            hidden_states = client.add(&out, &residual).map_err(Error::Numr)?;
        }
        Ok(hidden_states)
    }
}

/// A weight-norm-aware transposed-conv upsample stage. Holds the raw
/// weight/bias rather than a module: this crate has no
/// `nn::ConvTranspose1d` wrapper (see [`super::weight_norm_conv`]'s doc).
pub(super) struct HifiGanUpsampleLayer<R: Runtime> {
    pub(super) weight: Tensor<R>,
    pub(super) bias: Option<Tensor<R>>,
    pub(super) stride: usize,
    pub(super) padding: PaddingMode,
}

impl<R: Runtime<DType = DType>> HifiGanUpsampleLayer<R> {
    pub(super) fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ConvOps<R>,
    {
        client
            .conv_transpose1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                0,
                1,
                1,
            )
            .map_err(Error::Numr)
    }
}
