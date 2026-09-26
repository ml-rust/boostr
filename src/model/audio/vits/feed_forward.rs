//! `VitsFeedForward`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsFeedForward`.
//!
//! Two conv1d layers over the channel axis (`[batch, hidden, seq]`), each
//! preceded by a padding-mask multiply and HF's explicit left/right pad
//! (`(kernel-1)//2` left, `kernel//2` right — folded here into the conv's own
//! `PaddingMode::Custom`, which zero-pads identically to HF's
//! `nn.functional.pad` before a `Valid` conv). `hidden_act` gates the
//! activation between them; every published MMS-TTS checkpoint uses `relu`.

use crate::error::{Error, Result};
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ConvOps, PaddingMode};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsFeedForward<R: Runtime> {
    conv_1: Conv1d<R>,
    conv_2: Conv1d<R>,
    hidden_act: String,
}

/// HF's manual `nn.functional.pad([pad_left, pad_right, 0, 0, 0, 0])` ahead of
/// a `Valid` conv, folded into `PaddingMode`. `kernel_size == 1` needs no pad.
fn ffn_padding(kernel_size: usize) -> PaddingMode {
    if kernel_size > 1 {
        let pad_left = (kernel_size - 1) / 2;
        let pad_right = kernel_size / 2;
        PaddingMode::Custom(pad_left, pad_right, 0, 0)
    } else {
        PaddingMode::Valid
    }
}

impl<R: Runtime<DType = DType>> VitsFeedForward<R> {
    /// Reads `conv_1`/`conv_2` (`weight` + `bias`, always present — HF's
    /// `VitsFeedForward` never disables the conv bias).
    pub fn from_varbuilder(
        vb: &mut VarBuilder<'_, R>,
        hidden_act: &str,
        ffn_kernel_size: usize,
    ) -> Result<Self> {
        if hidden_act != "relu" && hidden_act != "gelu" {
            return Err(Error::ModelError {
                reason: format!(
                    "VitsFeedForward: unsupported hidden_act {hidden_act:?} (supported: relu, gelu)"
                ),
            });
        }
        let padding = ffn_padding(ffn_kernel_size);

        let mut c1 = vb.pp("conv_1");
        let w1 = c1.take_tensor("weight")?;
        let b1 = c1.take_tensor("bias")?;
        drop(c1);
        let conv_1 = Conv1d::new(w1, Some(b1), 1, padding, 1, 1, false);

        let mut c2 = vb.pp("conv_2");
        let w2 = c2.take_tensor("weight")?;
        let b2 = c2.take_tensor("bias")?;
        drop(c2);
        let conv_2 = Conv1d::new(w2, Some(b2), 1, padding, 1, 1, false);

        Ok(Self {
            conv_1,
            conv_2,
            hidden_act: hidden_act.to_string(),
        })
    }

    /// `hidden_states`: `[batch, seq, hidden]`. `padding_mask`: `[batch, seq,
    /// 1]` (1.0 valid, 0.0 padded) — matches HF's `input_padding_mask` layout
    /// before `VitsModel.forward`'s later transpose. Returns `[batch, seq,
    /// hidden]`.
    pub fn forward<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        padding_mask: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + BinaryOps<R> + ActivationOps<R>,
    {
        let hidden = hidden_states
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let mask = padding_mask
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        let hidden = client.mul(&hidden, &mask).map_err(Error::Numr)?;
        let hidden = self.conv_1.forward_inference(client, &hidden)?;
        let hidden = self.activate(client, &hidden)?;
        let hidden = client.mul(&hidden, &mask).map_err(Error::Numr)?;
        let hidden = self.conv_2.forward_inference(client, &hidden)?;
        let hidden = client.mul(&hidden, &mask).map_err(Error::Numr)?;

        hidden
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)
    }

    fn activate<C: ActivationOps<R>>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>> {
        match self.hidden_act.as_str() {
            "relu" => client.relu(x).map_err(Error::Numr),
            "gelu" => client.gelu_erf(x).map_err(Error::Numr),
            other => Err(Error::ModelError {
                reason: format!("VitsFeedForward: unsupported hidden_act {other:?}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn build(kernel: usize, hidden: usize, ffn: usize) -> VitsFeedForward<CpuRuntime> {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "conv_1.weight".into(),
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; ffn * hidden * kernel],
                &[ffn, hidden, kernel],
                &device,
            )
            .unwrap(),
        );
        map.insert(
            "conv_1.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; ffn], &[ffn], &device).unwrap(),
        );
        map.insert(
            "conv_2.weight".into(),
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; hidden * ffn * kernel],
                &[hidden, ffn, kernel],
                &device,
            )
            .unwrap(),
        );
        map.insert(
            "conv_2.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; hidden], &[hidden], &device).unwrap(),
        );
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        VitsFeedForward::from_varbuilder(&mut vb, "relu", kernel).unwrap()
    }

    #[test]
    fn forward_preserves_shape_and_masks_padding() {
        let (client, device) = cpu_setup();
        let ff = build(3, 4, 6);

        let batch = 2;
        let seq_len = 5;
        let hidden = 4;
        let hidden_states = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32; 2 * 5 * 4],
            &[batch, seq_len, hidden],
            &device,
        )
        .unwrap();
        let mut mask = vec![1.0f32; batch * seq_len];
        mask[seq_len + 3] = 0.0;
        mask[seq_len + 4] = 0.0;
        let padding_mask =
            Tensor::<CpuRuntime>::from_slice(&mask, &[batch, seq_len, 1], &device).unwrap();

        let out = ff.forward(&client, &hidden_states, &padding_mask).unwrap();
        assert_eq!(out.shape(), &[batch, seq_len, hidden]);

        let data: Vec<f32> = out.to_vec();
        for c in 0..hidden {
            assert_eq!(
                data[(seq_len + 3) * hidden + c],
                0.0,
                "padded position must be zero"
            );
            assert_eq!(
                data[(seq_len + 4) * hidden + c],
                0.0,
                "padded position must be zero"
            );
        }
    }

    #[test]
    fn rejects_unsupported_activation() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "conv_1.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 1, 3], &device).unwrap(),
        );
        map.insert(
            "conv_1.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[2], &device).unwrap(),
        );
        map.insert(
            "conv_2.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[1, 2, 3], &device).unwrap(),
        );
        map.insert(
            "conv_2.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 1], &[1], &device).unwrap(),
        );
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        assert!(VitsFeedForward::<CpuRuntime>::from_varbuilder(&mut vb, "swiglu", 3).is_err());
    }
}
