//! CPU Adagrad fused kernel implementations (f32, f64, and widened BF16/F16).

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Adagrad step for a parameter narrower than F32 (BF16/F16).
///
/// Widens every input to f32, runs the update in f32, and narrows only the
/// stores — matching the CUDA `fused_adagrad_bf16` / `fused_adagrad_f16`
/// kernels.
#[cfg(feature = "f16")]
pub(super) fn fused_adagrad_narrow(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    accum: &Tensor<CpuRuntime>,
    lr: f64,
    eps: f64,
    wd: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    use super::narrow::{read_f32, write_narrow};

    const OP: &str = "fused_adagrad_step";

    let n: usize = param.shape().iter().product();
    let p = read_f32(param, OP)?;
    let g = read_f32(grad, OP)?;
    let a = read_f32(accum, OP)?;

    let mut new_p = vec![0.0f32; n];
    let mut new_a = vec![0.0f32; n];

    let lr_f = lr as f32;
    let e = eps as f32;
    let w = wd as f32;

    for i in 0..n {
        let grad_wd = if w > 0.0 { g[i] + w * p[i] } else { g[i] };
        let acc = a[i] + grad_wd * grad_wd;
        new_a[i] = acc;
        new_p[i] = p[i] - lr_f * grad_wd / (acc.sqrt() + e);
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        write_narrow(&new_p, param.dtype(), shape, device, OP)?,
        write_narrow(&new_a, accum.dtype(), shape, device, OP)?,
    ))
}

pub(super) fn fused_adagrad_f32(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    accum: &Tensor<CpuRuntime>,
    lr: f64,
    eps: f64,
    wd: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f32>();
    let g = grad.to_vec::<f32>();
    let a = accum.to_vec::<f32>();

    let mut new_p = vec![0.0f32; n];
    let mut new_a = vec![0.0f32; n];

    let lr_f = lr as f32;
    let e = eps as f32;
    let w = wd as f32;

    for i in 0..n {
        let grad_wd = if w > 0.0 { g[i] + w * p[i] } else { g[i] };
        let acc = a[i] + grad_wd * grad_wd;
        new_a[i] = acc;
        new_p[i] = p[i] - lr_f * grad_wd / (acc.sqrt() + e);
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device)?,
        Tensor::<CpuRuntime>::from_slice(&new_a, shape, device)?,
    ))
}

pub(super) fn fused_adagrad_f64(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    accum: &Tensor<CpuRuntime>,
    lr: f64,
    eps: f64,
    wd: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f64>();
    let g = grad.to_vec::<f64>();
    let a = accum.to_vec::<f64>();

    let mut new_p = vec![0.0f64; n];
    let mut new_a = vec![0.0f64; n];

    for i in 0..n {
        let grad_wd = if wd > 0.0 { g[i] + wd * p[i] } else { g[i] };
        let acc = a[i] + grad_wd * grad_wd;
        new_a[i] = acc;
        new_p[i] = p[i] - lr * grad_wd / (acc.sqrt() + eps);
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device)?,
        Tensor::<CpuRuntime>::from_slice(&new_a, shape, device)?,
    ))
}
