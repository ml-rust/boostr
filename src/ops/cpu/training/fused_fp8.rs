//! CPU implementation of FusedFp8TrainingOps
//!
//! Single-pass gradient unscale + clip with inf/nan detection.

use crate::error::{Error, Result};
use crate::ops::impl_generic::training::dynamic_loss_scale_update_impl;
use crate::ops::traits::FusedFp8TrainingOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl FusedFp8TrainingOps<CpuRuntime> for CpuClient {
    fn fused_grad_unscale_clip(
        &self,
        grad: &Tensor<CpuRuntime>,
        max_norm: f64,
        loss_scale: f64,
    ) -> Result<(Tensor<CpuRuntime>, f64, bool)> {
        match grad.dtype() {
            DType::F32 => fused_grad_unscale_clip_f32(grad, max_norm, loss_scale),
            DType::F64 => fused_grad_unscale_clip_f64(grad, max_norm, loss_scale),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("fused_grad_unscale_clip: unsupported dtype {:?}", dt),
            }),
        }
    }

    fn dynamic_loss_scale_update(
        &self,
        found_inf: bool,
        loss_scale: f64,
        growth_tracker: i32,
        growth_interval: i32,
        backoff_factor: f64,
    ) -> Result<(f64, i32)> {
        dynamic_loss_scale_update_impl(
            found_inf,
            loss_scale,
            growth_tracker,
            growth_interval,
            backoff_factor,
        )
    }
}

fn fused_grad_unscale_clip_f32(
    grad: &Tensor<CpuRuntime>,
    max_norm: f64,
    loss_scale: f64,
) -> Result<(Tensor<CpuRuntime>, f64, bool)> {
    let n: usize = grad.shape().iter().product();
    let g = grad.to_vec::<f32>();
    let inv_scale = (1.0 / loss_scale) as f32;
    let max_norm_f = max_norm as f32;

    // Pass 1: check inf/nan + unscale + accumulate norm²
    let mut unscaled = vec![0.0f32; n];
    let mut norm_sq: f64 = 0.0;
    let mut found_inf = false;

    for i in 0..n {
        let gi = g[i];
        if gi.is_infinite() || gi.is_nan() {
            found_inf = true;
            break;
        }
        let u = gi * inv_scale;
        unscaled[i] = u;
        norm_sq += (u as f64) * (u as f64);
    }

    if found_inf {
        // Return zeros — caller should skip this step
        let zeros = Tensor::<CpuRuntime>::zeros(grad.shape(), DType::F32, grad.device());
        return Ok((zeros, 0.0, true));
    }

    let norm = (norm_sq as f32).sqrt();

    // Pass 2: clip if norm > max_norm
    if norm > max_norm_f && max_norm_f > 0.0 {
        let clip_coef = max_norm_f / (norm + 1e-6);
        for u in &mut unscaled {
            *u *= clip_coef;
        }
    }

    let result = Tensor::<CpuRuntime>::from_slice(&unscaled, grad.shape(), grad.device());
    Ok((result, norm as f64, false))
}

fn fused_grad_unscale_clip_f64(
    grad: &Tensor<CpuRuntime>,
    max_norm: f64,
    loss_scale: f64,
) -> Result<(Tensor<CpuRuntime>, f64, bool)> {
    let n: usize = grad.shape().iter().product();
    let g = grad.to_vec::<f64>();
    let inv_scale = 1.0 / loss_scale;

    // Pass 1: check inf/nan + unscale + accumulate norm²
    let mut unscaled = vec![0.0f64; n];
    let mut norm_sq: f64 = 0.0;
    let mut found_inf = false;

    for i in 0..n {
        let gi = g[i];
        if gi.is_infinite() || gi.is_nan() {
            found_inf = true;
            break;
        }
        let u = gi * inv_scale;
        unscaled[i] = u;
        norm_sq += u * u;
    }

    if found_inf {
        let zeros = Tensor::<CpuRuntime>::zeros(grad.shape(), DType::F64, grad.device());
        return Ok((zeros, 0.0, true));
    }

    let norm = norm_sq.sqrt();

    // Pass 2: clip if norm > max_norm
    if norm > max_norm && max_norm > 0.0 {
        let clip_coef = max_norm / (norm + 1e-6);
        for u in &mut unscaled {
            *u *= clip_coef;
        }
    }

    let result = Tensor::<CpuRuntime>::from_slice(&unscaled, grad.shape(), grad.device());
    Ok((result, norm, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_fused_grad_unscale_clip_basic() {
        let (client, device) = cpu_setup();
        let grad = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[4], &device);
        let loss_scale = 2.0;
        let max_norm = 10.0;

        let (clipped, norm, found_inf) = client
            .fused_grad_unscale_clip(&grad, max_norm, loss_scale)
            .unwrap();

        assert!(!found_inf);
        // unscaled = [1, 2, 3, 4], norm = sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
        let data = clipped.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((norm - 30.0f64.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_fused_grad_unscale_clip_clips() {
        let (client, device) = cpu_setup();
        let grad = Tensor::<CpuRuntime>::from_slice(&[20.0f32, 40.0, 60.0, 80.0], &[4], &device);
        let loss_scale = 2.0;
        let max_norm = 1.0;

        let (clipped, _norm, found_inf) = client
            .fused_grad_unscale_clip(&grad, max_norm, loss_scale)
            .unwrap();

        assert!(!found_inf);
        // unscaled = [10, 20, 30, 40], norm ≈ 54.77, should be clipped to max_norm=1.0
        let data = clipped.to_vec::<f32>();
        let clipped_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fused_grad_unscale_clip_inf() {
        let (client, device) = cpu_setup();
        let grad =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, f32::INFINITY, 3.0, 4.0], &[4], &device);

        let (_clipped, _norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 1.0).unwrap();

        assert!(found_inf);
    }

    #[test]
    fn test_fused_grad_unscale_clip_nan() {
        let (client, device) = cpu_setup();
        let grad = Tensor::<CpuRuntime>::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0], &[4], &device);

        let (_clipped, _norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 1.0).unwrap();

        assert!(found_inf);
    }

    #[test]
    fn test_dynamic_loss_scale_growth() {
        let (client, _device) = cpu_setup();
        let (scale, tracker) = client
            .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
            .unwrap();
        assert!((scale - 2048.0).abs() < 1e-10);
        assert_eq!(tracker, 0);
    }

    #[test]
    fn test_dynamic_loss_scale_backoff() {
        let (client, _device) = cpu_setup();
        let (scale, tracker) = client
            .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
            .unwrap();
        assert!((scale - 512.0).abs() < 1e-10);
        assert_eq!(tracker, 0);
    }
}
