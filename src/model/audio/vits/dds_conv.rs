//! `VitsDilatedDepthSeparableConv`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsDilatedDepthSeparableConv`.
//!
//! Each of `num_layers` blocks is: depthwise dilated conv1d (`groups =
//! channels`, dilation `kernel_size^i`) → channel-wise LayerNorm → exact GELU
//! → pointwise conv1d (`kernel=1`) → channel-wise LayerNorm → exact GELU →
//! residual add, each block masked on entry and the whole stack masked again
//! on exit. Dropout is a training-only no-op here (HF's own `nn.Dropout`
//! forward is identity in eval mode, and this module is inference-only).
//!
//! HF applies `nn.functional.gelu` with no `approximate` argument, which is
//! the exact (erf) form — numr's `gelu_erf`, not `gelu` (tanh approximation).

use crate::error::{Error, Result};
use crate::nn::{Conv1d, LayerNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, PaddingMode, ScalarOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsDilatedDepthSeparableConv<R: Runtime> {
    convs_dilated: Vec<Conv1d<R>>,
    convs_pointwise: Vec<Conv1d<R>>,
    norms_1: Vec<LayerNorm<R>>,
    norms_2: Vec<LayerNorm<R>>,
}

impl<R: Runtime<DType = DType>> VitsDilatedDepthSeparableConv<R> {
    /// Reads `convs_dilated.{i}`, `convs_pointwise.{i}`, `norms_1.{i}`,
    /// `norms_2.{i}` for `i` in `0..num_layers`. `channels` is the model's
    /// `hidden_size`; dilation for layer `i` is `kernel_size^i` (note: this is
    /// `kernel_size`, NOT a separate dilation-rate config — matching HF).
    pub fn from_varbuilder(
        vb: &mut VarBuilder<'_, R>,
        channels: usize,
        kernel_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut convs_dilated = Vec::with_capacity(num_layers);
        let mut convs_pointwise = Vec::with_capacity(num_layers);
        let mut norms_1 = Vec::with_capacity(num_layers);
        let mut norms_2 = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dilation = kernel_size.pow(i as u32);
            let padding = (kernel_size * dilation - dilation) / 2;

            let mut dw_vb = vb.pp(&format!("convs_dilated.{i}"));
            let dw_weight = dw_vb.take_tensor("weight")?;
            let dw_bias = dw_vb.take_tensor("bias")?;
            drop(dw_vb);
            convs_dilated.push(Conv1d::new(
                dw_weight,
                Some(dw_bias),
                1,
                PaddingMode::Custom(padding, padding, 0, 0),
                dilation,
                channels,
                false,
            ));

            let mut pw_vb = vb.pp(&format!("convs_pointwise.{i}"));
            let pw_weight = pw_vb.take_tensor("weight")?;
            let pw_bias = pw_vb.take_tensor("bias")?;
            drop(pw_vb);
            convs_pointwise.push(Conv1d::new(
                pw_weight,
                Some(pw_bias),
                1,
                PaddingMode::Valid,
                1,
                1,
                false,
            ));

            let mut n1_vb = vb.pp(&format!("norms_1.{i}"));
            let n1_weight = n1_vb.take_tensor("weight")?;
            let n1_bias = n1_vb.take_tensor("bias")?;
            drop(n1_vb);
            norms_1.push(LayerNorm::new(n1_weight, n1_bias, 1e-5, false));

            let mut n2_vb = vb.pp(&format!("norms_2.{i}"));
            let n2_weight = n2_vb.take_tensor("weight")?;
            let n2_bias = n2_vb.take_tensor("bias")?;
            drop(n2_vb);
            norms_2.push(LayerNorm::new(n2_weight, n2_bias, 1e-5, false));
        }

        Ok(Self {
            convs_dilated,
            convs_pointwise,
            norms_1,
            norms_2,
        })
    }

    /// `inputs`: `[batch, channels, seq_len]`. `padding_mask`: `[batch, 1,
    /// seq_len]` (1.0 valid, 0.0 padded), broadcast over the channel axis.
    /// `global_conditioning`, if given, is added once before the stack (same
    /// shape as `inputs`, or broadcastable to it).
    pub fn forward<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + BinaryOps<R> + ActivationOps<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let mut inputs = match global_conditioning {
            Some(g) => client.add(inputs, g).map_err(Error::Numr)?,
            None => inputs.clone(),
        };

        for i in 0..self.convs_dilated.len() {
            let masked = client.mul(&inputs, padding_mask).map_err(Error::Numr)?;
            let hidden = self.convs_dilated[i].forward_inference(client, &masked)?;
            let hidden = Self::layer_norm_channels(client, &self.norms_1[i], &hidden)?;
            let hidden = client.gelu_erf(&hidden).map_err(Error::Numr)?;
            let hidden = self.convs_pointwise[i].forward_inference(client, &hidden)?;
            let hidden = Self::layer_norm_channels(client, &self.norms_2[i], &hidden)?;
            let hidden = client.gelu_erf(&hidden).map_err(Error::Numr)?;
            inputs = client.add(&inputs, &hidden).map_err(Error::Numr)?;
        }

        client.mul(&inputs, padding_mask).map_err(Error::Numr)
    }

    /// Applies `ln` over the channel axis: `[B, C, T] -> transpose(1,2) ->
    /// LayerNorm over C -> transpose back -> [B, C, T]`. Matches HF's
    /// `norm(hidden_states.transpose(1, -1)).transpose(1, -1)`.
    fn layer_norm_channels<C>(client: &C, ln: &LayerNorm<R>, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let t = x
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let out = ln.forward(client, &Var::new(t, false))?;
        out.tensor()
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn layer_norm_channels_normalizes_over_channel_axis_not_time() {
        let (client, device) = cpu_setup();
        // [B=1, C=4, T=2]: two time steps, each with a distinct channel
        // profile. If the transpose direction were wrong, this would
        // normalize over T=2 instead of C=4 and the shapes would still
        // match but the values would differ from a manual per-time-step
        // channel normalization.
        // Row-major [1, C=4, T=2]: element (c, t) sits at flat index c*2+t.
        // t=0 profile is [1,2,3,4] across channels, t=1 profile is
        // [10,20,30,40] — a positive rescale of the same shape.
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            &[1, 4, 2],
            &device,
        )
        .unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        let ln = LayerNorm::new(weight, bias, 1e-5, false);

        let out =
            VitsDilatedDepthSeparableConv::<CpuRuntime>::layer_norm_channels(&client, &ln, &x)
                .unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        let data: Vec<f32> = out.to_vec();
        // Each time step's 4 channel values are independently normalized to
        // zero mean; column 0 (t=0) and column 1 (t=1) should therefore be
        // numerically identical since [1,2,3,4] and [10,20,30,40] are the
        // same shape up to an affine rescale.
        for c in 0..4 {
            let a = data[c * 2];
            let b = data[c * 2 + 1];
            assert!((a - b).abs() < 1e-3, "channel {c}: {a} vs {b}");
        }
    }
}
