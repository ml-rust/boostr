//! `VitsWaveNet`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsWaveNet`.
//!
//! A stack of dilated conv1d "in_layers" (dilation `dilation_rate^i`, kernel
//! `wavenet_kernel_size`) whose output is split into tanh/sigmoid halves,
//! optionally biased per-layer by a linear projection of a global
//! conditioning vector, gated (`tanh * sigmoid`), then run through a
//! per-layer 1x1 "res_skip" conv that is itself split into a residual half
//! (added back into the running hidden state, masked) and a skip half
//! (accumulated into the final output). The last layer's res_skip conv has
//! no residual half — its full output goes straight to skip. Every conv here
//! is `torch.nn.utils.parametrizations.weight_norm`-reparameterized in the
//! checkpoint; loading fuses `weight_g`/`weight_v` (or the modern
//! `parametrizations.weight.original0`/`original1` pair) into a plain
//! `weight` once, per [`crate::nn::fuse_weight_norm`].

use crate::error::{Error, Result};
use crate::nn::{Conv1d, VarBuilder, fuse_weight_norm};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConvOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsWaveNet<R: Runtime> {
    in_layers: Vec<Conv1d<R>>,
    res_skip_layers: Vec<Conv1d<R>>,
    /// `Conv1d(speaker_embedding_size, 2*hidden_size*num_layers, 1)`,
    /// present iff the checkpoint's `speaker_embedding_size > 0`.
    cond_layer: Option<Conv1d<R>>,
    hidden_size: usize,
}

/// Reads a weight-norm-reparameterized Conv1d at `{vb}.{prefix}`. Tries the
/// modern `parametrizations.weight.{original0,original1}` pair (`original0`
/// is `g`, `original1` is `v`) first, then the legacy `weight_g`/`weight_v`
/// pair, then falls back to a plain `weight` — detected per tensor, so a
/// source that folds weight-norm ahead of time (as compressr does when
/// embedding a checkpoint) still loads correctly.
#[allow(clippy::too_many_arguments)]
fn load_weight_normed_conv1d<R, C>(
    client: &C,
    vb: &mut VarBuilder<'_, R>,
    prefix: &str,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<Conv1d<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let mut conv_vb = vb.pp(prefix);
    let weight = if conv_vb.contains("parametrizations.weight.original0") {
        let g = conv_vb.take_tensor("parametrizations.weight.original0")?;
        let v = conv_vb.take_tensor("parametrizations.weight.original1")?;
        fuse_weight_norm(client, &v, &g, 0)?
    } else if conv_vb.contains("weight_g") {
        let g = conv_vb.take_tensor("weight_g")?;
        let v = conv_vb.take_tensor("weight_v")?;
        fuse_weight_norm(client, &v, &g, 0)?
    } else {
        conv_vb.take_tensor("weight")?
    };
    let bias = conv_vb.take_tensor_optional("bias")?;
    Ok(Conv1d::new(
        weight, bias, stride, padding, dilation, groups, false,
    ))
}

impl<R: Runtime<DType = DType>> VitsWaveNet<R> {
    /// `speaker_embedding_size == 0` means no `cond_layer` exists in the
    /// checkpoint (matches HF's `if config.speaker_embedding_size != 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        hidden_size: usize,
        kernel_size: usize,
        dilation_rate: usize,
        num_layers: usize,
        speaker_embedding_size: usize,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        let cond_layer = if speaker_embedding_size != 0 {
            Some(load_weight_normed_conv1d(
                client,
                vb,
                "cond_layer",
                1,
                PaddingMode::Valid,
                1,
                1,
            )?)
        } else {
            None
        };

        let mut in_layers = Vec::with_capacity(num_layers);
        let mut res_skip_layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dilation = dilation_rate.pow(i as u32);
            let padding = (kernel_size * dilation - dilation) / 2;
            in_layers.push(load_weight_normed_conv1d(
                client,
                vb,
                &format!("in_layers.{i}"),
                1,
                PaddingMode::Custom(padding, padding, 0, 0),
                dilation,
                1,
            )?);
            res_skip_layers.push(load_weight_normed_conv1d(
                client,
                vb,
                &format!("res_skip_layers.{i}"),
                1,
                PaddingMode::Valid,
                1,
                1,
            )?);
        }

        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            hidden_size,
        })
    }

    /// `inputs`: `[batch, hidden_size, seq_len]`. `padding_mask`: `[batch, 1,
    /// seq_len]`. `global_conditioning`: `[batch, speaker_embedding_size, 1]`,
    /// required iff this instance has a `cond_layer` (an error otherwise).
    pub fn forward<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let global_conditioning = match (global_conditioning, &self.cond_layer) {
            (Some(g), Some(cond_layer)) => Some(cond_layer.forward_inference(client, g)?),
            (Some(_), None) => {
                return Err(Error::ModelError {
                    reason: "VitsWaveNet: global_conditioning given but no cond_layer loaded (speaker_embedding_size == 0)".into(),
                });
            }
            (None, _) => None,
        };

        let mut inputs = inputs.clone();
        let mut outputs: Option<Tensor<R>> = None;
        let num_layers = self.in_layers.len();

        for i in 0..num_layers {
            let hidden = self.in_layers[i].forward_inference(client, &inputs)?;

            let acts_input = match &global_conditioning {
                Some(g) => {
                    let offset = i * 2 * self.hidden_size;
                    let slice = g
                        .narrow(1, offset, 2 * self.hidden_size)
                        .and_then(|t| t.contiguous())
                        .map_err(Error::Numr)?;
                    client.add(&hidden, &slice).map_err(Error::Numr)?
                }
                None => hidden,
            };

            let t_part = acts_input
                .narrow(1, 0, self.hidden_size)
                .and_then(|t| t.contiguous())
                .map_err(Error::Numr)?;
            let s_part = acts_input
                .narrow(1, self.hidden_size, self.hidden_size)
                .and_then(|t| t.contiguous())
                .map_err(Error::Numr)?;
            let t_act = client.tanh(&t_part).map_err(Error::Numr)?;
            let s_act = client.sigmoid(&s_part).map_err(Error::Numr)?;
            let acts = client.mul(&t_act, &s_act).map_err(Error::Numr)?;

            let res_skip_acts = self.res_skip_layers[i].forward_inference(client, &acts)?;

            if i < num_layers - 1 {
                let res_acts = res_skip_acts
                    .narrow(1, 0, self.hidden_size)
                    .and_then(|t| t.contiguous())
                    .map_err(Error::Numr)?;
                let added = client.add(&inputs, &res_acts).map_err(Error::Numr)?;
                inputs = client.mul(&added, padding_mask).map_err(Error::Numr)?;

                let skip = res_skip_acts
                    .narrow(1, self.hidden_size, self.hidden_size)
                    .and_then(|t| t.contiguous())
                    .map_err(Error::Numr)?;
                outputs = Some(match outputs {
                    Some(o) => client.add(&o, &skip).map_err(Error::Numr)?,
                    None => skip,
                });
            } else {
                outputs = Some(match outputs {
                    Some(o) => client.add(&o, &res_skip_acts).map_err(Error::Numr)?,
                    None => res_skip_acts,
                });
            }
        }

        let outputs = outputs.ok_or_else(|| Error::ModelError {
            reason: "VitsWaveNet: num_layers must be > 0".into(),
        })?;
        client.mul(&outputs, padding_mask).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn load_weight_normed_conv1d_falls_back_to_plain_weight() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "conv.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 1, 3], &device).unwrap(),
        );
        map.insert(
            "conv.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.5], &[2], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let conv = load_weight_normed_conv1d(&client, &mut vb, "conv", 1, PaddingMode::Valid, 1, 1)
            .unwrap();
        let w: Vec<f32> = conv.weight().tensor().to_vec();
        assert_eq!(w, vec![1.0f32; 6]);
    }

    #[test]
    fn load_weight_normed_conv1d_fuses_legacy_weight_g_v() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        // v: unit-norm direction per output channel, g: requested magnitude.
        map.insert(
            "conv.weight_v".into(),
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
                &[2, 1, 3],
                &device,
            )
            .unwrap(),
        );
        map.insert(
            "conv.weight_g".into(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1, 1], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let conv = load_weight_normed_conv1d(&client, &mut vb, "conv", 1, PaddingMode::Valid, 1, 1)
            .unwrap();
        let w: Vec<f32> = conv.weight().tensor().to_vec();
        assert!((w[0] - 3.0).abs() < 1e-5);
        assert!((w[4] - 4.0).abs() < 1e-5);
        assert!(conv.bias().is_none());
    }
}
