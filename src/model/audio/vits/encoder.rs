//! `VitsEncoder`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsEncoder`.
//!
//! A stack of [`VitsEncoderLayer`], masking `hidden_states` on entry and exit,
//! plus the 2D-to-4D additive attention-mask expansion HF's `VitsEncoder`
//! applies before the stack (`_prepare_4d_attention_mask`). `layerdrop` is a
//! training-only stochastic layer skip; every layer always runs here
//! (inference-only module).

use crate::error::{Error, Result};
use crate::model::attention_mask::mask_fill;
use crate::model::audio::vits::encoder_layer::VitsEncoderLayer;
use crate::model::config::VitsConfig;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ScalarOps, ShapeOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsEncoder<R: Runtime> {
    layers: Vec<VitsEncoderLayer<R>>,
}

impl<R: Runtime<DType = DType>> VitsEncoder<R> {
    /// Reads `layers.{i}.*` for `i` in `0..config.num_hidden_layers`.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let mut layer_vb = vb.pp(&format!("layers.{i}"));
            layers.push(VitsEncoderLayer::from_varbuilder(&mut layer_vb, config)?);
        }
        Ok(Self { layers })
    }

    /// `hidden_states`: `[batch, seq, hidden]`. `padding_mask`: `[batch, seq,
    /// 1]` (1.0 valid, 0.0 padded). `attention_mask`, if given: `[batch,
    /// seq]`, same convention, expanded here into the additive 4D form every
    /// layer's self-attention adds to its scores.
    pub fn forward<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        padding_mask: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let mask_4d = match attention_mask {
            Some(mask) => Some(expand_attention_mask(client, mask)?),
            None => None,
        };

        let mut hidden_states = client
            .mul(hidden_states, padding_mask)
            .map_err(Error::Numr)?;
        for layer in &self.layers {
            hidden_states =
                layer.forward(client, &hidden_states, mask_4d.as_ref(), padding_mask)?;
        }
        client
            .mul(&hidden_states, padding_mask)
            .map_err(Error::Numr)
    }
}

/// `_prepare_4d_attention_mask`: `[batch, src_len]` (1.0 valid, 0.0 padded) ->
/// additive `[batch, 1, 1, src_len]`, `0` where valid and the target dtype's
/// most-negative finite value where padded. The middle two axes broadcast
/// against `[batch, heads, tgt_len, src_len]` attention scores at add time —
/// `1` broadcasts against `heads`, and the query axis is likewise `1` because
/// self-attention masks every query row identically (the same keys are
/// padded regardless of which query is attending).
fn expand_attention_mask<R, C>(client: &C, mask: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>,
{
    let shape = mask.shape();
    let bsz = shape[0];
    let src_len = shape[1];
    let inverted = client.rsub_scalar(mask, 1.0).map_err(Error::Numr)?;
    let fill = mask_fill(mask.dtype());
    let filled = client.mul_scalar(&inverted, fill).map_err(Error::Numr)?;
    filled.reshape(&[bsz, 1, 1, src_len]).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn expand_attention_mask_zeroes_valid_and_fills_padded() {
        let (client, device) = cpu_setup();
        // batch 0 fully valid, batch 1's last position padded.
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0], &[2, 3], &device)
                .unwrap();
        let out = expand_attention_mask::<CpuRuntime, _>(&client, &mask).unwrap();
        assert_eq!(out.shape(), &[2, 1, 1, 3]);
        let data: Vec<f32> = out.to_vec();
        assert_eq!(&data[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&data[3..5], &[0.0, 0.0]);
        assert!(
            data[5] < -1.0e30,
            "padded position must carry the fill: {data:?}"
        );
    }
}
