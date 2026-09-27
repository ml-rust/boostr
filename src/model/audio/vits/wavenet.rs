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
//! `weight` once, per [`crate::model::audio::vits::weight_norm_conv`].

use crate::error::{Error, Result};
use crate::model::audio::vits::weight_norm_conv::load_weight_normed_conv1d;
use crate::nn::{Conv1d, VarBuilder};
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

    /// Two layers, no conditioning: shape is preserved and padded positions
    /// stay zeroed (the mask is applied on every masked step and the final
    /// output).
    #[test]
    fn forward_preserves_shape_and_masks_padding() {
        let (client, device) = cpu_setup();
        let hidden_size = 2;
        let kernel_size = 3;
        let num_layers = 2;
        let mut map = VarMap::<CpuRuntime>::new();
        for i in 0..num_layers {
            map.insert(
                format!("in_layers.{i}.weight"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.1f32; 2 * hidden_size * hidden_size * kernel_size],
                    &[2 * hidden_size, hidden_size, kernel_size],
                    &device,
                )
                .unwrap(),
            );
            map.insert(
                format!("in_layers.{i}.bias"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.0f32; 2 * hidden_size],
                    &[2 * hidden_size],
                    &device,
                )
                .unwrap(),
            );
            let res_skip_out = if i < num_layers - 1 {
                2 * hidden_size
            } else {
                hidden_size
            };
            map.insert(
                format!("res_skip_layers.{i}.weight"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.1f32; res_skip_out * hidden_size],
                    &[res_skip_out, hidden_size, 1],
                    &device,
                )
                .unwrap(),
            );
            map.insert(
                format!("res_skip_layers.{i}.bias"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.0f32; res_skip_out],
                    &[res_skip_out],
                    &device,
                )
                .unwrap(),
            );
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        let wavenet = VitsWaveNet::from_varbuilder(
            &client,
            &mut vb,
            hidden_size,
            kernel_size,
            1,
            num_layers,
            0,
        )
        .unwrap();

        let batch = 1;
        let length = 3;
        let inputs = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3],
            &[batch, hidden_size, length],
            &device,
        )
        .unwrap();
        // last position is padding.
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 0.0], &[batch, 1, length], &device)
                .unwrap();

        let out = wavenet.forward(&client, &inputs, &mask, None).unwrap();
        assert_eq!(out.shape(), &[batch, hidden_size, length]);
        let values: Vec<f32> = out.contiguous().unwrap().to_vec();
        for c in 0..hidden_size {
            assert_eq!(
                values[c * length + length - 1],
                0.0,
                "padded position must be zeroed: {values:?}"
            );
        }
    }

    #[test]
    fn forward_rejects_conditioning_without_cond_layer() {
        let (client, device) = cpu_setup();
        let hidden_size = 1;
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "in_layers.0.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.1], &[2, 1, 1], &device).unwrap(),
        );
        map.insert(
            "in_layers.0.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap(),
        );
        map.insert(
            "res_skip_layers.0.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[1, 1, 1], &device).unwrap(),
        );
        map.insert(
            "res_skip_layers.0.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let wavenet =
            VitsWaveNet::from_varbuilder(&client, &mut vb, hidden_size, 1, 1, 1, 0).unwrap();

        let inputs =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, hidden_size, 1], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();
        let bogus_cond = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();

        let err = wavenet.forward(&client, &inputs, &mask, Some(&bogus_cond));
        assert!(err.is_err());
    }
}
