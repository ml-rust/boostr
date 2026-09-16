//! Shared test-construction helpers for the `prosody_predictor` module's test
//! suites.

use crate::model::audio::kokoro::{AdaLayerNorm, KokoroAdaIn1d};
use crate::nn::{BiLstm, Conv1d, Lstm};
use numr::ops::PaddingMode;
use numr::runtime::{Runtime, cpu::CpuRuntime};
use numr::tensor::Tensor;

pub(crate) fn zeros(
    shape: &[usize],
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device).unwrap()
}
pub(crate) fn ones(
    shape: &[usize],
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device).unwrap()
}

pub(crate) fn bilstm(
    in_dim: usize,
    hidden: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> BiLstm<CpuRuntime> {
    let mk = || {
        Lstm::new(
            zeros(&[4 * hidden, in_dim], device),
            zeros(&[4 * hidden, hidden], device),
            zeros(&[4 * hidden], device),
            zeros(&[4 * hidden], device),
        )
        .unwrap()
    };
    BiLstm::new(mk(), mk()).unwrap()
}

pub(crate) fn adaln(
    channels: usize,
    style_dim: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> AdaLayerNorm<CpuRuntime> {
    AdaLayerNorm::new(
        zeros(&[2 * channels, style_dim], device),
        zeros(&[2 * channels], device),
        1e-5,
    )
    .unwrap()
}

pub(crate) fn kadain(
    c: usize,
    s: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> KokoroAdaIn1d<CpuRuntime> {
    KokoroAdaIn1d::new(
        zeros(&[2 * c, s], device),
        zeros(&[2 * c], device),
        ones(&[c], device),
        zeros(&[c], device),
        1e-5,
    )
    .unwrap()
}

pub(crate) fn conv(
    c_out: usize,
    c_in: usize,
    k: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Conv1d<CpuRuntime> {
    Conv1d::new(
        zeros(&[c_out, c_in, k], device),
        Some(zeros(&[c_out], device)),
        1,
        PaddingMode::Same,
        1,
        1,
        false,
    )
}
