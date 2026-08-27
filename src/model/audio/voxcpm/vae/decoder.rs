//! `AudioVaeDecoder` — VoxCPM2's `AudioVAE` decoder top-level assembly.
//!
//! ```text
//! latent [B, 64, T]
//!   -> front: depthwise CausalConv1d(64->64, k=7, groups=64)
//!   -> front: pointwise CausalConv1d(64->2048, k=1)
//!   -> 6x DecoderBlock (strides 8,6,5,2,2,2; 2048->1024->512->256->128->64->32)
//!   -> Snake(32)
//!   -> CausalConv1d(32->1, k=7)
//!   -> tanh
//! waveform [B, 1, T * 1920]                      (8*6*5*2*2*2 = 1920)
//! ```
//!
//! No `NoiseBlock` (`use_noise_block=false` in this checkpoint) and no
//! `out_layer` for the sample-rate conditioning (`Identity` upstream) — both
//! absent from the real `model.safetensors`, so neither is modeled here.
//!
//! Inference-only: built from plain [`Tensor<R>`] weights, no autograd.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::decoder_block::DecoderBlock;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-stage upsample strides, outermost (closest to the latent) first.
pub const STRIDES: [usize; 6] = [8, 6, 5, 2, 2, 2];
/// Product of [`STRIDES`]: samples produced per input latent frame.
pub const HOP_LENGTH: usize = 1920;
/// `AudioVAE` output sample rate for this decoder's fixed sr-conditioning
/// bucket (see [`DEFAULT_SR_BUCKET`]).
pub const SAMPLE_RATE: usize = 48_000;

/// VoxCPM2 calls `decode()` with no explicit `sr_cond`, which defaults to
/// `out_sample_rate = 48000`. Bucketizing 48000 against the checkpoint's
/// `sr_bin_boundaries = [20000, 30000, 40000]` (upstream `torch.bucketize`,
/// right-open bins) lands in bucket 3 — the last bin, for anything at or
/// above 40 kHz. This is NOT derived from the 16 kHz *input* rate to the
/// encoder (that would bucketize to 0, a different, wrong affine transform
/// for this decoder). Every production call decodes to 48 kHz, so the bucket
/// is fixed here rather than threaded through as a runtime `sr_cond` input.
pub const DEFAULT_SR_BUCKET: usize = 3;

pub(crate) const INPUT_CHANNELS: usize = 64;
pub(crate) const FRONT_HIDDEN: usize = 2048;
pub(crate) const FINAL_CHANNELS: usize = 32;
pub(crate) const OUTPUT_CHANNELS: usize = 1;
/// Depthwise/final conv kernel size used throughout the decoder (everywhere
/// except the two pointwise convs and the transposed upsamples).
pub(crate) const CAUSAL_KERNEL: usize = 7;
/// Per-`ResUnit` dilations within a `DecoderBlock`, in order.
pub(crate) const RES_UNIT_DILATIONS: [usize; 3] = [1, 3, 9];
/// Number of sample-rate conditioning buckets (`scale_embed`/`bias_embed`
/// row count).
pub(crate) const NUM_SR_BUCKETS: usize = 4;

/// Bundled, already-built weights for the full `AudioVAE` decoder.
pub struct AudioVaeDecoderWeights<R: Runtime> {
    pub front_dw: CausalConv1d<R>,
    pub front_pw: CausalConv1d<R>,
    pub blocks: [DecoderBlock<R>; 6],
    pub final_snake: Snake<R>,
    pub final_conv: CausalConv1d<R>,
}

/// VoxCPM2 `AudioVAE` decoder: latent `[B, 64, T]` -> waveform `[B, 1, T *
/// 1920]` at 48 kHz.
pub struct AudioVaeDecoder<R: Runtime> {
    front_dw: CausalConv1d<R>,
    front_pw: CausalConv1d<R>,
    blocks: [DecoderBlock<R>; 6],
    final_snake: Snake<R>,
    final_conv: CausalConv1d<R>,
}

impl<R: Runtime<DType = DType>> AudioVaeDecoder<R> {
    pub fn new(weights: AudioVaeDecoderWeights<R>) -> Self {
        Self {
            front_dw: weights.front_dw,
            front_pw: weights.front_pw,
            blocks: weights.blocks,
            final_snake: weights.final_snake,
            final_conv: weights.final_conv,
        }
    }

    /// Dtype every decoder weight was loaded at.
    ///
    /// The `AudioVAE` is never cast at load time (it is verified against F32
    /// PyTorch fixtures), so a transformer stack running at another dtype must
    /// convert its latent to THIS dtype before decoding.
    pub fn dtype(&self) -> DType {
        self.front_dw.dtype()
    }

    /// Full forward with the default sample-rate bucket ([`DEFAULT_SR_BUCKET`]
    /// = 48 kHz, the only rate VoxCPM2's `decode()` ever produces).
    pub fn forward<C>(&self, client: &C, latent: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        self.forward_with_bucket(client, latent, DEFAULT_SR_BUCKET)
    }

    /// Full forward with an explicit sample-rate bucket, for callers that
    /// need a non-default `sr_cond` target.
    pub fn forward_with_bucket<C>(
        &self,
        client: &C,
        latent: &Tensor<R>,
        sr_bucket: usize,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let shape = latent.shape();
        if shape.len() != 3 || shape[1] != INPUT_CHANNELS {
            return Err(Error::InvalidArgument {
                arg: "latent",
                reason: format!("expected [B, {INPUT_CHANNELS}, T], got {shape:?}"),
            });
        }

        let x = self.front_dw.forward(client, latent)?;
        let mut x = self.front_pw.forward(client, &x)?;

        for block in &self.blocks {
            x = block.forward(client, &x, sr_bucket)?;
        }

        let x = self.final_snake.forward(client, &x)?;
        let x = self.final_conv.forward(client, &x)?;
        client.tanh(&x).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::vae::causal_transpose_conv1d::CausalTransposeConv1d;
    use crate::model::audio::voxcpm::vae::decoder_block::DecoderBlockWeights;
    use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
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
            depthwise(c, 7, dilation, device),
            snake(c, device),
            pointwise(c, c, device),
        )
    }

    fn decoder_block(
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> DecoderBlock<CpuRuntime> {
        let k = 2 * stride;
        let up_weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; input_dim * output_dim * k],
            &[input_dim, output_dim, k],
            device,
        )
        .unwrap();
        let up_bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
                .unwrap();
        let num_buckets = 4;
        let scale_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![1.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .unwrap();
        let bias_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .unwrap();
        DecoderBlock::new(DecoderBlockWeights {
            snake: snake(input_dim, device),
            upsample: CausalTransposeConv1d::new(up_weight, Some(up_bias), stride).unwrap(),
            res1: res_unit(output_dim, 1, device),
            res3: res_unit(output_dim, 3, device),
            res9: res_unit(output_dim, 9, device),
            scale_embed,
            bias_embed,
        })
        .unwrap()
    }

    fn build_decoder(device: &<CpuRuntime as Runtime>::Device) -> AudioVaeDecoder<CpuRuntime> {
        let dims = [
            (FRONT_HIDDEN, FRONT_HIDDEN / 2),
            (FRONT_HIDDEN / 2, FRONT_HIDDEN / 4),
            (FRONT_HIDDEN / 4, FRONT_HIDDEN / 8),
            (FRONT_HIDDEN / 8, FRONT_HIDDEN / 16),
            (FRONT_HIDDEN / 16, FRONT_HIDDEN / 32),
            (FRONT_HIDDEN / 32, FINAL_CHANNELS),
        ];
        let blocks =
            std::array::from_fn(|i| decoder_block(dims[i].0, dims[i].1, STRIDES[i], device));

        AudioVaeDecoder::new(AudioVaeDecoderWeights {
            front_dw: depthwise(INPUT_CHANNELS, 7, 1, device),
            front_pw: pointwise(INPUT_CHANNELS, FRONT_HIDDEN, device),
            blocks,
            final_snake: snake(FINAL_CHANNELS, device),
            final_conv: {
                let weight = Tensor::<CpuRuntime>::from_slice(
                    &vec![0.001f32; OUTPUT_CHANNELS * FINAL_CHANNELS * 7],
                    &[OUTPUT_CHANNELS, FINAL_CHANNELS, 7],
                    device,
                )
                .unwrap();
                let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[OUTPUT_CHANNELS], device)
                    .unwrap();
                CausalConv1d::new(weight, Some(bias), 7, 1, 1).unwrap()
            },
        })
    }

    #[test]
    fn forward_output_shape_matches_hop_length() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let t = 3;
        let latent_data: Vec<f32> = (0..(INPUT_CHANNELS * t))
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let latent =
            Tensor::<CpuRuntime>::from_slice(&latent_data, &[1, INPUT_CHANNELS, t], &device)
                .unwrap();
        let out = decoder.forward(&client, &latent).unwrap();
        assert_eq!(out.shape(), &[1, OUTPUT_CHANNELS, t * HOP_LENGTH]);
    }

    #[test]
    fn forward_output_is_bounded_by_tanh() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let t = 2;
        let latent = Tensor::<CpuRuntime>::from_slice(
            &vec![0.3f32; INPUT_CHANNELS * t],
            &[1, INPUT_CHANNELS, t],
            &device,
        )
        .unwrap();
        let out = decoder.forward(&client, &latent).unwrap();
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
            assert!((-1.0..=1.0).contains(&v), "tanh output out of range: {v}");
        }
    }

    #[test]
    fn rejects_wrong_input_channels() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let latent =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8 * 3], &[1, 8, 3], &device).unwrap();
        assert!(decoder.forward(&client, &latent).is_err());
    }
}
