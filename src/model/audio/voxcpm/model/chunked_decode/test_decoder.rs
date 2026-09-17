//! A small-weight `AudioVaeDecoder` at full architecture scale (`STRIDES`,
//! `CAUSAL_KERNEL`, dilations all match production) for the windowed-decode
//! tests, so `CONTEXT_FRAMES` is exercised exactly as derived. Non-degenerate
//! weights: a zero decoder would make every equality test vacuous.

use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::causal_transpose_conv1d::CausalTransposeConv1d;
use crate::model::audio::voxcpm::vae::decoder::{
    AudioVaeDecoder, AudioVaeDecoderWeights, FINAL_CHANNELS, FRONT_HIDDEN, INPUT_CHANNELS, STRIDES,
};
use crate::model::audio::voxcpm::vae::decoder_block::{DecoderBlock, DecoderBlockWeights};
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

pub(super) fn snake(c: usize, device: &<CpuRuntime as Runtime>::Device) -> Snake<CpuRuntime> {
    let alpha =
        Tensor::<CpuRuntime>::from_slice(&vec![0.2f32; c], &[1, c, 1], device).expect("alpha");
    Snake::new(alpha).expect("snake")
}

pub(super) fn depthwise(
    c: usize,
    k: usize,
    dilation: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> CausalConv1d<CpuRuntime> {
    let weight = Tensor::<CpuRuntime>::from_slice(
        &(0..c * k)
            .map(|i| 0.01 * ((i % 5) as f32 - 2.0))
            .collect::<Vec<f32>>(),
        &[c, 1, k],
        device,
    )
    .expect("weight");
    let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).expect("bias");
    CausalConv1d::new(weight, Some(bias), k, dilation, c).expect("depthwise")
}

pub(super) fn pointwise(
    c_in: usize,
    c_out: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> CausalConv1d<CpuRuntime> {
    let weight = Tensor::<CpuRuntime>::from_slice(
        &(0..c_out * c_in)
            .map(|i| 0.005 * ((i % 7) as f32 - 3.0))
            .collect::<Vec<f32>>(),
        &[c_out, c_in, 1],
        device,
    )
    .expect("weight");
    let bias =
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c_out], &[c_out], device).expect("bias");
    CausalConv1d::new(weight, Some(bias), 1, 1, 1).expect("pointwise")
}

pub(super) fn res_unit(
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

pub(super) fn decoder_block(
    input_dim: usize,
    output_dim: usize,
    stride: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> DecoderBlock<CpuRuntime> {
    let k = 2 * stride;
    let up_weight = Tensor::<CpuRuntime>::from_slice(
        &(0..input_dim * output_dim * k)
            .map(|i| 0.01 * ((i % 5) as f32 - 2.0))
            .collect::<Vec<f32>>(),
        &[input_dim, output_dim, k],
        device,
    )
    .expect("up_weight");
    let up_bias =
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
            .expect("up_bias");
    let num_buckets = 4;
    let scale_embed = Tensor::<CpuRuntime>::from_slice(
        &vec![1.0f32; num_buckets * input_dim],
        &[num_buckets, input_dim],
        device,
    )
    .expect("scale_embed");
    let bias_embed = Tensor::<CpuRuntime>::from_slice(
        &vec![0.0f32; num_buckets * input_dim],
        &[num_buckets, input_dim],
        device,
    )
    .expect("bias_embed");
    DecoderBlock::new(DecoderBlockWeights {
        snake: snake(input_dim, device),
        upsample: CausalTransposeConv1d::new(up_weight, Some(up_bias), stride).expect("upsample"),
        res1: res_unit(output_dim, 1, device),
        res3: res_unit(output_dim, 3, device),
        res9: res_unit(output_dim, 9, device),
        scale_embed,
        bias_embed,
    })
    .expect("decoder_block")
}

/// A real (small-weight, non-degenerate) `AudioVaeDecoder` at full
/// architecture scale (`STRIDES`, `CAUSAL_KERNEL`, dilations all match
/// production), so `CONTEXT_FRAMES` is exercised exactly as derived.
pub(super) fn build_decoder(
    device: &<CpuRuntime as Runtime>::Device,
) -> AudioVaeDecoder<CpuRuntime> {
    let dims = [
        (FRONT_HIDDEN, FRONT_HIDDEN / 2),
        (FRONT_HIDDEN / 2, FRONT_HIDDEN / 4),
        (FRONT_HIDDEN / 4, FRONT_HIDDEN / 8),
        (FRONT_HIDDEN / 8, FRONT_HIDDEN / 16),
        (FRONT_HIDDEN / 16, FRONT_HIDDEN / 32),
        (FRONT_HIDDEN / 32, FINAL_CHANNELS),
    ];
    let blocks = std::array::from_fn(|i| decoder_block(dims[i].0, dims[i].1, STRIDES[i], device));

    AudioVaeDecoder::new(AudioVaeDecoderWeights {
        front_dw: depthwise(INPUT_CHANNELS, 7, 1, device),
        front_pw: pointwise(INPUT_CHANNELS, FRONT_HIDDEN, device),
        blocks,
        final_snake: snake(FINAL_CHANNELS, device),
        final_conv: {
            let weight = Tensor::<CpuRuntime>::from_slice(
                &(0..FINAL_CHANNELS * 7)
                    .map(|i| 0.001 * ((i % 5) as f32 - 2.0))
                    .collect::<Vec<f32>>(),
                &[1, FINAL_CHANNELS, 7],
                device,
            )
            .expect("final_conv weight");
            let bias =
                Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], device).expect("final_conv bias");
            CausalConv1d::new(weight, Some(bias), 7, 1, 1).expect("final_conv")
        },
    })
}

pub(super) fn latent(
    frames: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    let data: Vec<f32> = (0..INPUT_CHANNELS * frames)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[1, INPUT_CHANNELS, frames], device).expect("latent")
}
