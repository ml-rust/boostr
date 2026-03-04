//! CPU SGD fused kernel implementations (f32 and f64).

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_sgd_f32(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    momentum_buf: Option<&Tensor<CpuRuntime>>,
    lr: f64,
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f32>();
    let g = grad.to_vec::<f32>();
    let has_buf = momentum_buf.is_some();
    let buf_data = momentum_buf.map(|b| b.to_vec::<f32>());

    let mut new_p = vec![0.0f32; n];
    let mut new_buf = vec![0.0f32; n];

    let lr_f = lr as f32;
    let mom = momentum as f32;
    let damp = dampening as f32;
    let w = wd as f32;

    for i in 0..n {
        let grad_wd = if w > 0.0 { g[i] + w * p[i] } else { g[i] };

        let buf = if mom > 0.0 {
            if has_buf {
                let prev = buf_data.as_ref().expect("has_buf is true")[i];
                mom * prev + (1.0 - damp) * grad_wd
            } else {
                grad_wd
            }
        } else {
            grad_wd
        };
        new_buf[i] = buf;

        let update = if nesterov && mom > 0.0 {
            grad_wd + mom * buf
        } else if mom > 0.0 {
            buf
        } else {
            grad_wd
        };

        new_p[i] = p[i] - lr_f * update;
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_buf, shape, device),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_sgd_f64(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    momentum_buf: Option<&Tensor<CpuRuntime>>,
    lr: f64,
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f64>();
    let g = grad.to_vec::<f64>();
    let has_buf = momentum_buf.is_some();
    let buf_data = momentum_buf.map(|b| b.to_vec::<f64>());

    let mut new_p = vec![0.0f64; n];
    let mut new_buf = vec![0.0f64; n];

    for i in 0..n {
        let grad_wd = if wd > 0.0 { g[i] + wd * p[i] } else { g[i] };

        let buf = if momentum > 0.0 {
            if has_buf {
                let prev = buf_data.as_ref().expect("has_buf is true")[i];
                momentum * prev + (1.0 - dampening) * grad_wd
            } else {
                grad_wd
            }
        } else {
            grad_wd
        };
        new_buf[i] = buf;

        let update = if nesterov && momentum > 0.0 {
            grad_wd + momentum * buf
        } else if momentum > 0.0 {
            buf
        } else {
            grad_wd
        };

        new_p[i] = p[i] - lr * update;
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_buf, shape, device),
    ))
}
