//! CPU LAMB fused kernel implementations (f32 and f64).

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_lamb_f32(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    m: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    bias_corr1: f64,
    bias_corr2: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f32>();
    let g = grad.to_vec::<f32>();
    let m_data = m.to_vec::<f32>();
    let v_data = v.to_vec::<f32>();

    let mut new_m = vec![0.0f32; n];
    let mut new_v = vec![0.0f32; n];
    let mut update = vec![0.0f32; n];

    let b1 = beta1 as f32;
    let b2 = beta2 as f32;
    let e = eps as f32;
    let w = wd as f32;
    let bc1 = bias_corr1 as f32;
    let bc2 = bias_corr2 as f32;

    for i in 0..n {
        let gi = g[i];
        let mi = b1 * m_data[i] + (1.0 - b1) * gi;
        let vi = b2 * v_data[i] + (1.0 - b2) * gi * gi;
        new_m[i] = mi;
        new_v[i] = vi;

        let m_hat = mi / bc1;
        let v_hat = vi / bc2;
        let adam_update = m_hat / (v_hat.sqrt() + e);
        update[i] = if w > 0.0 {
            adam_update + w * p[i]
        } else {
            adam_update
        };
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&update, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_m, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_v, shape, device),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_lamb_f64(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    m: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    bias_corr1: f64,
    bias_corr2: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f64>();
    let g = grad.to_vec::<f64>();
    let m_data = m.to_vec::<f64>();
    let v_data = v.to_vec::<f64>();

    let mut new_m = vec![0.0f64; n];
    let mut new_v = vec![0.0f64; n];
    let mut update = vec![0.0f64; n];

    for i in 0..n {
        let gi = g[i];
        let mi = beta1 * m_data[i] + (1.0 - beta1) * gi;
        let vi = beta2 * v_data[i] + (1.0 - beta2) * gi * gi;
        new_m[i] = mi;
        new_v[i] = vi;

        let m_hat = mi / bias_corr1;
        let v_hat = vi / bias_corr2;
        let adam_update = m_hat / (v_hat.sqrt() + eps);
        update[i] = if wd > 0.0 {
            adam_update + wd * p[i]
        } else {
            adam_update
        };
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&update, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_m, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_v, shape, device),
    ))
}
