//! `AudioVaeEncoder` — VoxCPM2's `AudioVAE` encoder top-level assembly
//! (inference only).
//!
//! ```text
//! waveform [B, 1 or _, T] @ 16 kHz
//!   -> preprocess: right-pad with zeros to a multiple of HOP_LENGTH (640)
//!   -> CausalConv1d(1->128, k=7)
//!   -> 4x EncoderBlock (strides 2,5,8,8; 128->256->512->1024->2048)
//!   -> fc_mu: CausalConv1d(2048->64, k=3)
//! mu [B, 64, ceil(T/640)]                          (2*5*8*8 = 640)
//! ```
//!
//! No sample-rate conditioning: that machinery is decoder-only, the encoder
//! always runs at a fixed 16 kHz. No trailing Snake/conv after the block
//! stack — `fc_mu` is the last layer. `fc_logvar` exists in the checkpoint
//! with the same shape as `fc_mu` but is never used: `AudioVAE.encode()`
//! returns `mu` deterministically with no reparameterisation sampling, so
//! this port does not load it at all (see `vae/loader/encoder.rs`).
//!
//! Inference-only: built from plain [`Tensor<R>`] weights, no autograd.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::encoder_block::EncoderBlock;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-stage downsample strides, in forward order.
pub const STRIDES: [usize; 4] = [2, 5, 8, 8];
/// Product of [`STRIDES`]: input samples consumed per output latent frame.
pub const HOP_LENGTH: usize = 640;

pub(crate) const INPUT_CHANNELS: usize = 1;
pub(crate) const FRONT_HIDDEN: usize = 128;
pub(crate) const FINAL_HIDDEN: usize = 2048;
pub(crate) const OUTPUT_CHANNELS: usize = 64;
/// Kernel size for the front conv and every `ResUnit` dilated conv.
pub(crate) const RES_KERNEL: usize = 7;
/// Kernel size for `fc_mu` / `fc_logvar`.
pub(crate) const HEAD_KERNEL: usize = 3;
/// Per-`ResUnit` dilations within an `EncoderBlock`, in order.
pub(crate) const RES_UNIT_DILATIONS: [usize; 3] = [1, 3, 9];

/// Bundled, already-built weights for the full `AudioVAE` encoder.
pub struct AudioVaeEncoderWeights<R: Runtime> {
    pub front: CausalConv1d<R>,
    pub blocks: [EncoderBlock<R>; 4],
    pub fc_mu: CausalConv1d<R>,
}

/// VoxCPM2 `AudioVAE` encoder: waveform `[B, 1, T]` @ 16 kHz -> `mu [B, 64,
/// ceil(T/640)]`.
pub struct AudioVaeEncoder<R: Runtime> {
    front: CausalConv1d<R>,
    blocks: [EncoderBlock<R>; 4],
    fc_mu: CausalConv1d<R>,
}

impl<R: Runtime<DType = DType>> AudioVaeEncoder<R> {
    pub fn new(weights: AudioVaeEncoderWeights<R>) -> Self {
        Self {
            front: weights.front,
            blocks: weights.blocks,
            fc_mu: weights.fc_mu,
        }
    }

    /// Right-pads `wave [B, 1, T]` with zeros so `T` becomes a multiple of
    /// [`HOP_LENGTH`]. A no-op if `T` is already a multiple.
    fn preprocess<C>(client: &C, wave: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let t = wave.shape()[2];
        let remainder = t % HOP_LENGTH;
        if remainder == 0 {
            return Ok(wave.clone());
        }
        let pad_right = HOP_LENGTH - remainder;
        client.pad(wave, &[0, pad_right], 0.0).map_err(Error::Numr)
    }

    /// `wave`: `[B, T]` or `[B, 1, T]` @ 16 kHz -> `mu [B, 64, ceil(T/640)]`.
    pub fn forward<C>(&self, client: &C, wave: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let shape = wave.shape().to_vec();
        let wave = match shape.len() {
            2 => wave
                .reshape(&[shape[0], 1, shape[1]])
                .map_err(Error::Numr)?,
            3 if shape[1] == INPUT_CHANNELS => wave.clone(),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "wave",
                    reason: format!("expected [B, T] or [B, 1, T], got {shape:?}"),
                });
            }
        };

        let x = Self::preprocess(client, &wave)?;
        let mut x = self.front.forward(client, &x)?;

        for block in &self.blocks {
            x = block.forward(client, &x)?;
        }

        self.fc_mu.forward(client, &x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::vae::encoder_block::EncoderBlockWeights;
    use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
    use crate::model::audio::voxcpm::vae::snake::Snake;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn snake(c: usize, device: &<CpuRuntime as Runtime>::Device) -> Snake<CpuRuntime> {
        let alpha = Tensor::<CpuRuntime>::from_slice(&vec![0.2f32; c], &[1, c, 1], device).unwrap();
        Snake::new(alpha).unwrap()
    }

    fn depthwise(
        c: usize,
        k: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let weight =
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * k], &[c, 1, k], device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap();
        CausalConv1d::new(weight, Some(bias), k, dilation, c).unwrap()
    }

    fn pointwise(
        c_in: usize,
        c_out: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.005f32; c_out * c_in],
            &[c_out, c_in, 1],
            device,
        )
        .unwrap();
        let bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c_out], &[c_out], device).unwrap();
        CausalConv1d::new(weight, Some(bias), 1, 1, 1).unwrap()
    }

    fn res_unit(
        c: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> ResUnit<CpuRuntime> {
        ResUnit::new(
            snake(c, device),
            depthwise(c, RES_KERNEL, dilation, device),
            snake(c, device),
            pointwise(c, c, device),
        )
    }

    fn encoder_block(
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> EncoderBlock<CpuRuntime> {
        let k = 2 * stride;
        let down_weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; output_dim * input_dim * k],
            &[output_dim, input_dim, k],
            device,
        )
        .unwrap();
        let down_bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
                .unwrap();
        EncoderBlock::new(EncoderBlockWeights {
            res1: res_unit(input_dim, RES_UNIT_DILATIONS[0], device),
            res3: res_unit(input_dim, RES_UNIT_DILATIONS[1], device),
            res9: res_unit(input_dim, RES_UNIT_DILATIONS[2], device),
            snake: snake(input_dim, device),
            downsample: CausalConv1d::new_strided(down_weight, Some(down_bias), stride, 1).unwrap(),
        })
    }

    fn front_conv(device: &<CpuRuntime as Runtime>::Device) -> CausalConv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; FRONT_HIDDEN * INPUT_CHANNELS * RES_KERNEL],
            &[FRONT_HIDDEN, INPUT_CHANNELS, RES_KERNEL],
            device,
        )
        .unwrap();
        let bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; FRONT_HIDDEN], &[FRONT_HIDDEN], device)
                .unwrap();
        CausalConv1d::new(weight, Some(bias), RES_KERNEL, 1, 1).unwrap()
    }

    fn head_conv(device: &<CpuRuntime as Runtime>::Device) -> CausalConv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.005f32; OUTPUT_CHANNELS * FINAL_HIDDEN * HEAD_KERNEL],
            &[OUTPUT_CHANNELS, FINAL_HIDDEN, HEAD_KERNEL],
            device,
        )
        .unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; OUTPUT_CHANNELS],
            &[OUTPUT_CHANNELS],
            device,
        )
        .unwrap();
        CausalConv1d::new(weight, Some(bias), HEAD_KERNEL, 1, 1).unwrap()
    }

    fn build_encoder(device: &<CpuRuntime as Runtime>::Device) -> AudioVaeEncoder<CpuRuntime> {
        let dims = [
            (FRONT_HIDDEN, FRONT_HIDDEN * 2),
            (FRONT_HIDDEN * 2, FRONT_HIDDEN * 4),
            (FRONT_HIDDEN * 4, FRONT_HIDDEN * 8),
            (FRONT_HIDDEN * 8, FRONT_HIDDEN * 16),
        ];
        let blocks: [EncoderBlock<CpuRuntime>; 4] =
            std::array::from_fn(|i| encoder_block(dims[i].0, dims[i].1, STRIDES[i], device));

        AudioVaeEncoder::new(AudioVaeEncoderWeights {
            front: front_conv(device),
            blocks,
            fc_mu: head_conv(device),
        })
    }

    #[test]
    fn forward_output_shape_matches_hop_length_multiple_input() {
        let (client, device) = cpu_setup();
        let encoder = build_encoder(&device);
        let frames = 3;
        let t = frames * HOP_LENGTH;
        let wave_data: Vec<f32> = (0..t).map(|i| (i as f32 * 0.001).sin()).collect();
        let wave = Tensor::<CpuRuntime>::from_slice(&wave_data, &[1, 1, t], &device).unwrap();
        let out = encoder.forward(&client, &wave).unwrap();
        assert_eq!(out.shape(), &[1, OUTPUT_CHANNELS, frames]);
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_pads_non_multiple_input_and_rounds_up() {
        let (client, device) = cpu_setup();
        let encoder = build_encoder(&device);
        let t = HOP_LENGTH + 1; // not a multiple -> should round up to 2 frames
        let wave_data: Vec<f32> = (0..t).map(|i| (i as f32 * 0.001).cos()).collect();
        let wave = Tensor::<CpuRuntime>::from_slice(&wave_data, &[1, 1, t], &device).unwrap();
        let out = encoder.forward(&client, &wave).unwrap();
        assert_eq!(out.shape(), &[1, OUTPUT_CHANNELS, 2]);
    }

    #[test]
    fn forward_accepts_2d_input() {
        let (client, device) = cpu_setup();
        let encoder = build_encoder(&device);
        let t = HOP_LENGTH;
        let wave_data: Vec<f32> = vec![0.1f32; t];
        let wave = Tensor::<CpuRuntime>::from_slice(&wave_data, &[1, t], &device).unwrap();
        let out = encoder.forward(&client, &wave).unwrap();
        assert_eq!(out.shape(), &[1, OUTPUT_CHANNELS, 1]);
    }

    #[test]
    fn rejects_wrong_channel_count() {
        let (client, device) = cpu_setup();
        let encoder = build_encoder(&device);
        let wave = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32; 2 * HOP_LENGTH],
            &[1, 2, HOP_LENGTH],
            &device,
        )
        .unwrap();
        assert!(encoder.forward(&client, &wave).is_err());
    }
}
