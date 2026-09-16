//! Shared test-construction helpers for the `generator` module's test suites.

use crate::model::audio::kokoro::{AdaINResBlock1, KokoroAdaIn1d};
use crate::nn::Conv1d;
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
pub(crate) fn adain(
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

pub(crate) fn resblock(
    c: usize,
    s: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> AdaINResBlock1<CpuRuntime> {
    AdaINResBlock1::new(
        [
            conv(c, c, 3, device),
            conv(c, c, 3, device),
            conv(c, c, 3, device),
        ],
        [
            conv(c, c, 3, device),
            conv(c, c, 3, device),
            conv(c, c, 3, device),
        ],
        [
            adain(c, s, device),
            adain(c, s, device),
            adain(c, s, device),
        ],
        [
            adain(c, s, device),
            adain(c, s, device),
            adain(c, s, device),
        ],
        [
            ones(&[1, c, 1], device),
            ones(&[1, c, 1], device),
            ones(&[1, c, 1], device),
        ],
        [
            ones(&[1, c, 1], device),
            ones(&[1, c, 1], device),
            ones(&[1, c, 1], device),
        ],
        1e-9,
    )
    .unwrap()
}
