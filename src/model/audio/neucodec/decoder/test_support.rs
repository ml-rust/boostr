//! Shared test-construction helpers for the `decoder` module's test suites.

use super::{NeuCodecDecoder, NeuCodecDecoderWeights};
use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::model::audio::neucodec::istft_head::{IstftHead, IstftHeadWeights};
use crate::model::audio::neucodec::resnet_block::{ResnetBlock, ResnetBlockWeights};
use crate::model::audio::neucodec::transformer_block::{TransformerBlock, TransformerBlockWeights};
use crate::nn::{Conv1d, GroupNorm, LayerNorm, Linear, RmsNorm};
use crate::test_utils::cpu_setup;
use numr::ops::PaddingMode;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(crate) fn linear(
    out_f: usize,
    in_f: usize,
    val: f32,
    device: &CpuDevice,
) -> Linear<CpuRuntime> {
    Linear::new(
        Tensor::<CpuRuntime>::from_slice(&vec![val; out_f * in_f], &[out_f, in_f], device).unwrap(),
        Some(Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; out_f], &[out_f], device).unwrap()),
        false,
    )
}

pub(crate) fn layer_norm(c: usize, device: &CpuDevice) -> LayerNorm<CpuRuntime> {
    LayerNorm::new(
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap(),
        1e-5,
        false,
    )
}

pub(crate) fn rms_norm(c: usize, device: &CpuDevice) -> RmsNorm<CpuRuntime> {
    RmsNorm::new(
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
        1e-6,
        false,
    )
}

pub(crate) fn conv(c: usize, k: usize, val: f32, device: &CpuDevice) -> Conv1d<CpuRuntime> {
    let n = c * c * k;
    Conv1d::new(
        Tensor::<CpuRuntime>::from_slice(&vec![val; n], &[c, c, k], device).unwrap(),
        Some(Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap()),
        1,
        PaddingMode::Same,
        1,
        1,
        false,
    )
}

/// GroupNorm with a test-scale group count (production uses 32; the
/// synthetic decoder here is only 8 channels wide).
pub(crate) fn group_norm(c: usize, groups: usize, device: &CpuDevice) -> GroupNorm<CpuRuntime> {
    GroupNorm::new(
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap(),
        groups,
        1e-6,
        false,
    )
}

pub(crate) fn resnet_block(
    c: usize,
    k: usize,
    val: f32,
    device: &CpuDevice,
) -> ResnetBlock<CpuRuntime> {
    ResnetBlock::new(ResnetBlockWeights {
        norm1: group_norm(c, 2, device),
        conv1: conv(c, k, val, device),
        norm2: group_norm(c, 2, device),
        conv2: conv(c, k, val, device),
    })
}

pub(crate) fn transformer_block(
    hidden: usize,
    heads: usize,
    head_dim: usize,
    mlp: usize,
    val: f32,
    device: &CpuDevice,
) -> TransformerBlock<CpuRuntime> {
    TransformerBlock::new(
        TransformerBlockWeights {
            input_layernorm: rms_norm(hidden, device),
            q_proj: linear(hidden, hidden, val, device),
            k_proj: linear(hidden, hidden, val, device),
            v_proj: linear(hidden, hidden, val, device),
            o_proj: linear(hidden, hidden, val, device),
            post_attention_layernorm: rms_norm(hidden, device),
            mlp_fc1: linear(mlp, hidden, val, device),
            mlp_fc2: linear(hidden, mlp, val, device),
        },
        heads,
        head_dim,
    )
    .unwrap()
}

/// A small synthetic decoder: hidden=8, heads=2, head_dim=4, mlp=16,
/// fc_in_dim=6, n_fft=8 (F=5), hop=3. Weights are tiny nonzero values so
/// outputs are non-degenerate but numerically small (finite-output check).
pub(crate) fn make_decoder(
    val: f32,
) -> (
    NeuCodecDecoder<CpuRuntime>,
    numr::runtime::cpu::CpuClient,
    CpuDevice,
    NeuCodecDecoderConfig,
) {
    let (client, device) = cpu_setup();
    let hidden = 8;
    let heads = 2;
    let head_dim = 4;
    let mlp = 16;
    let fc_in = 6;
    let n_fft = 8;
    // `n_fft - hop` must be EVEN for the `samples == frames * hop` identity
    // to hold exactly (the Vocos trim is `(n_fft - hop) / 2`, floored).
    // The real config satisfies this: 1920 - 480 = 1440.
    let hop = 4;
    let config = NeuCodecDecoderConfig {
        hidden_size: hidden,
        fc_in_dim: fc_in,
        embed_kernel_size: 3,
        resnet_kernel_size: 3,
        num_prior_resnet_blocks: 2,
        num_post_resnet_blocks: 2,
        num_transformer_layers: 2,
        num_heads: heads,
        head_dim,
        mlp_intermediate_size: mlp,
        rms_norm_eps: 1e-6,
        resnet_norm_groups: 2,
        resnet_norm_eps: 1e-6,
        layer_norm_eps: 1e-6,
        n_fft,
        hop_length: hop,
        mag_clamp_max: 1e2,
    };

    let weights = NeuCodecDecoderWeights {
        fc: linear(hidden, fc_in, val, &device),
        embed: conv(hidden, config.embed_kernel_size, val, &device),
        prior_net: (0..config.num_prior_resnet_blocks)
            .map(|_| resnet_block(hidden, config.resnet_kernel_size, val, &device))
            .collect(),
        layers: (0..config.num_transformer_layers)
            .map(|_| transformer_block(hidden, heads, head_dim, mlp, val, &device))
            .collect(),
        norm: layer_norm(hidden, &device),
        post_net: (0..config.num_post_resnet_blocks)
            .map(|_| resnet_block(hidden, config.resnet_kernel_size, val, &device))
            .collect(),
        head: IstftHead::new(
            IstftHeadWeights {
                linear: linear(config.head_out_dim(), hidden, val, &device),
            },
            n_fft,
            config.mag_clamp_max,
        )
        .unwrap(),
    };

    let decoder = NeuCodecDecoder::new(config, weights).unwrap();
    (decoder, client, device, config)
}
