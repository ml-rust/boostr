//! `VitsElementwiseAffine` and `VitsConvFlow`, reverse (inference) path only.
//!
//! Exact port of HF `transformers.models.vits.modeling_vits.VitsElementwiseAffine`
//! and `VitsConvFlow`. Both are normalizing-flow steps used inside
//! [`super::duration_predictor::VitsStochasticDurationPredictor`]'s flow
//! stack. Neither implements the forward (training, log-likelihood) branch:
//! this crate is inference-only, and the forward branch is never reached
//! from a loaded checkpoint at inference time.

use crate::error::{Error, Result};
use crate::model::audio::vits::dds_conv::VitsDilatedDepthSeparableConv;
use crate::model::audio::vits::spline::{RqSplineClient, unconstrained_rational_quadratic_spline};
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, ConvOps, NormalizationOps, PaddingMode, ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `translate` and `log_scale`: each `[channels, 1]`, broadcast over
/// `[batch, channels, length]`.
pub struct VitsElementwiseAffine<R: Runtime> {
    translate: Tensor<R>,
    log_scale: Tensor<R>,
}

impl<R: Runtime<DType = DType>> VitsElementwiseAffine<R> {
    /// Reads `translate` and `log_scale` under `vb`'s current prefix.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>) -> Result<Self> {
        let translate = vb.take_tensor("translate")?;
        let log_scale = vb.take_tensor("log_scale")?;
        Ok(Self {
            translate,
            log_scale,
        })
    }

    /// `(inputs - translate) * exp(-log_scale) * padding_mask`. `inputs` and
    /// `padding_mask`: `[batch, channels, length]` and `[batch, 1, length]`.
    pub fn reverse<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let diff = client.sub(inputs, &self.translate).map_err(Error::Numr)?;
        let neg_log_scale = client.neg(&self.log_scale).map_err(Error::Numr)?;
        let inv_scale = client.exp(&neg_log_scale).map_err(Error::Numr)?;
        let scaled = client.mul(&diff, &inv_scale).map_err(Error::Numr)?;
        client.mul(&scaled, padding_mask).map_err(Error::Numr)
    }
}

/// A coupling flow over half the channels, gated by a rational-quadratic
/// spline whose bin parameters come from a [`VitsDilatedDepthSeparableConv`]
/// stack conditioned on the other half.
pub struct VitsConvFlow<R: Runtime> {
    conv_pre: Conv1d<R>,
    conv_dds: VitsDilatedDepthSeparableConv<R>,
    conv_proj: Conv1d<R>,
    half_channels: usize,
    filter_channels: usize,
    num_bins: usize,
    tail_bound: f64,
}

impl<R: Runtime<DType = DType>> VitsConvFlow<R> {
    /// Reads `conv_pre.*`, `conv_dds.*`, `conv_proj.*` under `vb`'s current
    /// prefix. `filter_channels` is `config.hidden_size` (HF's
    /// `VitsConvFlow.filter_channels`, distinct from
    /// `duration_predictor_filter_channels`, which only the non-stochastic
    /// [`super::duration_predictor::VitsDurationPredictor`] uses).
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let filter_channels = config.hidden_size;
        let half_channels = config.depth_separable_channels / 2;
        let num_bins = config.duration_predictor_flow_bins;
        let tail_bound = config.duration_predictor_tail_bound;

        let mut pre_vb = vb.pp("conv_pre");
        let pre_weight = pre_vb.take_tensor("weight")?;
        let pre_bias = pre_vb.take_tensor("bias")?;
        drop(pre_vb);
        let conv_pre = Conv1d::new(
            pre_weight,
            Some(pre_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        let mut dds_vb = vb.pp("conv_dds");
        let conv_dds = VitsDilatedDepthSeparableConv::from_varbuilder(
            &mut dds_vb,
            filter_channels,
            config.duration_predictor_kernel_size,
            config.depth_separable_num_layers,
        )?;
        drop(dds_vb);

        let mut proj_vb = vb.pp("conv_proj");
        let proj_weight = proj_vb.take_tensor("weight")?;
        let proj_bias = proj_vb.take_tensor("bias")?;
        drop(proj_vb);
        let conv_proj = Conv1d::new(
            proj_weight,
            Some(proj_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        Ok(Self {
            conv_pre,
            conv_dds,
            conv_proj,
            half_channels,
            filter_channels,
            num_bins,
            tail_bound,
        })
    }

    /// `inputs`: `[batch, 2 * half_channels, length]`, split into an
    /// untouched first half and a spline-transformed second half.
    /// `global_conditioning`: added inside the depth-separable conv stack
    /// (HF's `inputs` variable from the enclosing duration predictor, the
    /// conv-projected hidden state — not the model's speaker embedding).
    pub fn reverse<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RqSplineClient<R> + ConvOps<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let first_half = inputs
            .narrow(1, 0, self.half_channels)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let second_half = inputs
            .narrow(1, self.half_channels, self.half_channels)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        let hidden = self.conv_pre.forward_inference(client, &first_half)?;
        let hidden =
            self.conv_dds
                .forward(client, &hidden, padding_mask, Some(global_conditioning))?;
        let hidden = self.conv_proj.forward_inference(client, &hidden)?;
        let hidden = client.mul(&hidden, padding_mask).map_err(Error::Numr)?;

        let batch = first_half.shape()[0];
        let channels = first_half.shape()[1];
        let length = first_half.shape()[2];
        let bin_params = 3 * self.num_bins - 1;
        let reshaped = hidden
            .reshape(&[batch, channels, bin_params, length])
            .map_err(Error::Numr)?;
        // HF's `.permute(0, 1, 3, 2)`: [batch, channels, bin_params, length]
        // -> [batch, channels, length, bin_params].
        let permuted = reshaped
            .permute(&[0, 1, 3, 2])
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        let scale = 1.0 / (self.filter_channels as f64).sqrt();
        let unnormalized_widths = client
            .mul_scalar(
                &permuted.narrow(3, 0, self.num_bins).map_err(Error::Numr)?,
                scale,
            )
            .map_err(Error::Numr)?;
        let unnormalized_heights = client
            .mul_scalar(
                &permuted
                    .narrow(3, self.num_bins, self.num_bins)
                    .map_err(Error::Numr)?,
                scale,
            )
            .map_err(Error::Numr)?;
        let unnormalized_derivatives = permuted
            .narrow(3, 2 * self.num_bins, self.num_bins - 1)
            .map_err(Error::Numr)?;

        let (spline_out, _log_abs_det) = unconstrained_rational_quadratic_spline(
            client,
            &second_half,
            &unnormalized_widths,
            &unnormalized_heights,
            &unnormalized_derivatives,
            true,
            self.tail_bound,
        )?;

        let outputs = client
            .cat(&[&first_half, &spline_out], 1)
            .map_err(Error::Numr)?;
        client.mul(&outputs, padding_mask).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn elementwise_affine_reverse_undoes_identity_scale() {
        let (client, device) = cpu_setup();
        // translate=0, log_scale=0 -> reverse is the identity (times the mask).
        let translate = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2, 1], &device).unwrap();
        let log_scale = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2, 1], &device).unwrap();
        let affine = VitsElementwiseAffine {
            translate,
            log_scale,
        };

        let inputs =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2], &device)
                .unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 1, 2], &device).unwrap();
        let out = affine.reverse(&client, &inputs, &mask).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn elementwise_affine_reverse_applies_padding_mask() {
        let (client, device) = cpu_setup();
        let translate = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2, 1], &device).unwrap();
        let log_scale =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0f32], &[2, 1], &device).unwrap();
        let affine = VitsElementwiseAffine {
            translate,
            log_scale,
        };

        let inputs = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[1, 2, 1], &device).unwrap();
        // padded position (mask 0) must come out zero regardless of translate/log_scale.
        let mask = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1, 1], &device).unwrap();
        let out = affine.reverse(&client, &inputs, &mask).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![0.0, 0.0]);
    }
}
