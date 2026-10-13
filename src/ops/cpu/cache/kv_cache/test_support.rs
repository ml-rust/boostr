//! Shared F32 fill fixtures for the CPU KV-cache unit tests.

use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

pub(super) fn zeros(
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    shape: &[usize],
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device).unwrap()
}

pub(super) fn ones(
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    shape: &[usize],
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device).unwrap()
}
