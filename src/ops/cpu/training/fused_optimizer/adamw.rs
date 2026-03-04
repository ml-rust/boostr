//! CPU AdamW fused kernel implementations (f32 and f64).

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_adamw_f32(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    m: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    step_size: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f32>();
    let g = grad.to_vec::<f32>();
    let m_data = m.to_vec::<f32>();
    let v_data = v.to_vec::<f32>();

    let mut new_p = vec![0.0f32; n];
    let mut new_m = vec![0.0f32; n];
    let mut new_v = vec![0.0f32; n];

    let b1 = beta1 as f32;
    let b2 = beta2 as f32;
    let e = eps as f32;
    let ss = step_size as f32;
    let decay = (lr * wd) as f32;

    for i in 0..n {
        let gi = g[i];
        let mi = b1 * m_data[i] + (1.0 - b1) * gi;
        let vi = b2 * v_data[i] + (1.0 - b2) * gi * gi;
        let update = ss * mi / (vi.sqrt() + e);
        let decayed = p[i] * (1.0 - decay);
        new_p[i] = decayed - update;
        new_m[i] = mi;
        new_v[i] = vi;
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_m, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_v, shape, device),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_adamw_f64(
    param: &Tensor<CpuRuntime>,
    grad: &Tensor<CpuRuntime>,
    m: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    step_size: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let n: usize = param.shape().iter().product();
    let p = param.to_vec::<f64>();
    let g = grad.to_vec::<f64>();
    let m_data = m.to_vec::<f64>();
    let v_data = v.to_vec::<f64>();

    let mut new_p = vec![0.0f64; n];
    let mut new_m = vec![0.0f64; n];
    let mut new_v = vec![0.0f64; n];

    let decay = lr * wd;

    for i in 0..n {
        let gi = g[i];
        let mi = beta1 * m_data[i] + (1.0 - beta1) * gi;
        let vi = beta2 * v_data[i] + (1.0 - beta2) * gi * gi;
        let update = step_size * mi / (vi.sqrt() + eps);
        let decayed = p[i] * (1.0 - decay);
        new_p[i] = decayed - update;
        new_m[i] = mi;
        new_v[i] = vi;
    }

    let shape = param.shape();
    let device = param.device();
    Ok((
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_m, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_v, shape, device),
    ))
}
