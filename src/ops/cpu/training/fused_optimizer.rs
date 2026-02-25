//! CPU implementation of FusedOptimizerOps
//!
//! Single-pass parameter updates using raw pointer arithmetic.
//! Each fused kernel reads all inputs and writes all outputs in one loop,
//! reducing memory traffic by 4-8x vs composing individual ops.

use crate::error::{Error, Result};
use crate::ops::traits::FusedOptimizerOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
impl FusedOptimizerOps<CpuRuntime> for CpuClient {
    fn fused_adamw_step(
        &self,
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
        validate_shapes(param, grad, "fused_adamw_step")?;
        validate_shapes(param, m, "fused_adamw_step")?;
        validate_shapes(param, v, "fused_adamw_step")?;

        match param.dtype() {
            DType::F32 => fused_adamw_f32(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size),
            DType::F64 => fused_adamw_f64(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("fused_adamw_step: unsupported dtype {:?}", dt),
            }),
        }
    }

    fn fused_sgd_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        momentum_buf: Option<&Tensor<CpuRuntime>>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_sgd_step")?;
        if let Some(buf) = momentum_buf {
            validate_shapes(param, buf, "fused_sgd_step")?;
        }

        match param.dtype() {
            DType::F32 => fused_sgd_f32(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            DType::F64 => fused_sgd_f64(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("fused_sgd_step: unsupported dtype {:?}", dt),
            }),
        }
    }

    fn fused_adagrad_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        accum: &Tensor<CpuRuntime>,
        lr: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_adagrad_step")?;
        validate_shapes(param, accum, "fused_adagrad_step")?;

        match param.dtype() {
            DType::F32 => fused_adagrad_f32(param, grad, accum, lr, eps, wd),
            DType::F64 => fused_adagrad_f64(param, grad, accum, lr, eps, wd),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("fused_adagrad_step: unsupported dtype {:?}", dt),
            }),
        }
    }

    fn fused_lamb_step(
        &self,
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
        validate_shapes(param, grad, "fused_lamb_step")?;
        validate_shapes(param, m, "fused_lamb_step")?;
        validate_shapes(param, v, "fused_lamb_step")?;

        match param.dtype() {
            DType::F32 => fused_lamb_f32(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            DType::F64 => fused_lamb_f64(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("fused_lamb_step: unsupported dtype {:?}", dt),
            }),
        }
    }

    fn fused_multi_tensor_adamw(
        &self,
        groups: &[(
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
        )],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<Vec<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
        groups
            .iter()
            .map(|(param, grad, m, v)| {
                self.fused_adamw_step(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            })
            .collect()
    }
}

fn validate_shapes(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>, op: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!("{}: shape mismatch {:?} vs {:?}", op, a.shape(), b.shape()),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fused_adamw_f32(
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
fn fused_adamw_f64(
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

#[allow(clippy::too_many_arguments)]
fn fused_sgd_f32(
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
                let prev = buf_data.as_ref().unwrap()[i];
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
fn fused_sgd_f64(
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
                let prev = buf_data.as_ref().unwrap()[i];
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

fn fused_adagrad_f32(
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
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_a, shape, device),
    ))
}

fn fused_adagrad_f64(
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
        Tensor::<CpuRuntime>::from_slice(&new_p, shape, device),
        Tensor::<CpuRuntime>::from_slice(&new_a, shape, device),
    ))
}

#[allow(clippy::too_many_arguments)]
fn fused_lamb_f32(
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
fn fused_lamb_f64(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_fused_adamw_basic() {
        let (client, device) = cpu_setup();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device);
        let m = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device);
        let v = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device);

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let bc1 = 1.0 - beta1;
        let bc2 = (1.0_f64 - beta2).sqrt();
        let step_size = lr * bc2 / bc1;

        let (new_p, new_m, new_v) = client
            .fused_adamw_step(&param, &grad, &m, &v, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        let p_data = new_p.to_vec::<f32>();
        assert!(p_data[0] < 1.0, "param should decrease: {}", p_data[0]);
        assert!(new_m.to_vec::<f32>()[0] > 0.0, "m should be positive");
        assert!(new_v.to_vec::<f32>()[0] > 0.0, "v should be positive");
    }

    #[test]
    fn test_fused_sgd_basic() {
        let (client, device) = cpu_setup();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);

        let (new_p, _buf) = client
            .fused_sgd_step(&param, &grad, None, 0.1, 0.0, 0.0, 0.0, false)
            .unwrap();

        let p = new_p.to_vec::<f32>();
        assert!((p[0] - 0.99).abs() < 1e-6);
        assert!((p[1] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn test_fused_multi_tensor_adamw() {
        let (client, device) = cpu_setup();

        let p1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let g1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let m1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);
        let v1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);

        let p2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &device);
        let g2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4, 0.5], &[3], &device);
        let m2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device);
        let v2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device);

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let bc1 = 1.0 - beta1;
        let bc2 = (1.0_f64 - beta2).sqrt();
        let step_size = lr * bc2 / bc1;

        let groups = vec![(&p1, &g1, &m1, &v1), (&p2, &g2, &m2, &v2)];

        let results = client
            .fused_multi_tensor_adamw(&groups, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        assert_eq!(results.len(), 2);

        // Verify results match individual calls
        let (ref_p1, ref_m1, ref_v1) = client
            .fused_adamw_step(&p1, &g1, &m1, &v1, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();
        let (ref_p2, ref_m2, ref_v2) = client
            .fused_adamw_step(&p2, &g2, &m2, &v2, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        assert_eq!(results[0].0.to_vec::<f32>(), ref_p1.to_vec::<f32>());
        assert_eq!(results[0].1.to_vec::<f32>(), ref_m1.to_vec::<f32>());
        assert_eq!(results[0].2.to_vec::<f32>(), ref_v1.to_vec::<f32>());
        assert_eq!(results[1].0.to_vec::<f32>(), ref_p2.to_vec::<f32>());
        assert_eq!(results[1].1.to_vec::<f32>(), ref_m2.to_vec::<f32>());
        assert_eq!(results[1].2.to_vec::<f32>(), ref_v2.to_vec::<f32>());
    }

    #[test]
    fn test_fused_adagrad_basic() {
        let (client, device) = cpu_setup();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let accum = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);

        let (new_p, new_acc) = client
            .fused_adagrad_step(&param, &grad, &accum, 0.1, 1e-10, 0.0)
            .unwrap();

        let p = new_p.to_vec::<f32>();
        assert!((p[0] - 0.9).abs() < 1e-4);
        assert!((p[1] - 1.9).abs() < 1e-4);
        assert!(new_acc.to_vec::<f32>()[0] > 0.0);
    }
}
