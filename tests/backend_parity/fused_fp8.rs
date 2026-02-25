//! Backend parity tests for FusedFp8TrainingOps
//!
//! Verifies that CPU, CUDA, and WebGPU produce numerically identical results
//! for fused gradient unscale+clip and dynamic loss scale update.

use super::helpers::{assert_parity_f32, setup_cpu};
use boostr::FusedFp8TrainingOps;
use numr::tensor::Tensor;

// ---- Gradient unscale + clip parity ----

#[test]
fn test_fused_grad_unscale_clip_cpu_reference() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[4], &device);

    let (clipped, norm, found_inf) = client.fused_grad_unscale_clip(&grad, 10.0, 2.0).unwrap();

    assert!(!found_inf);
    // unscaled = [1, 2, 3, 4], norm = sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
    let data = clipped.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-5, "data[0]={}", data[0]);
    assert!((data[1] - 2.0).abs() < 1e-5, "data[1]={}", data[1]);
    assert!((data[2] - 3.0).abs() < 1e-5, "data[2]={}", data[2]);
    assert!((data[3] - 4.0).abs() < 1e-5, "data[3]={}", data[3]);
    assert!((norm - 30.0f64.sqrt()).abs() < 1e-4, "norm={}", norm);
}

#[test]
fn test_fused_grad_unscale_clip_no_clip_cpu() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device);

    let (clipped, norm, found_inf) = client.fused_grad_unscale_clip(&grad, 100.0, 1.0).unwrap();

    assert!(!found_inf);
    // norm = sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14) ≈ 0.374 — well under max_norm=100
    let data = clipped.to_vec::<f32>();
    assert!((data[0] - 0.1).abs() < 1e-6, "data[0]={}", data[0]);
    assert!((data[1] - 0.2).abs() < 1e-6, "data[1]={}", data[1]);
    assert!((data[2] - 0.3).abs() < 1e-6, "data[2]={}", data[2]);
    assert!((norm - 0.14f64.sqrt()).abs() < 1e-4, "norm={}", norm);
}

#[test]
fn test_fused_grad_unscale_clip_with_clipping_cpu() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[20.0f32, 40.0, 60.0, 80.0], &[4], &device);

    let (clipped, norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 2.0).unwrap();

    assert!(!found_inf);
    // unscaled = [10, 20, 30, 40], norm = sqrt(100+400+900+1600) ≈ 54.77
    // clipped norm should be ≈ max_norm = 1.0
    let data = clipped.to_vec::<f32>();
    let clipped_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (clipped_norm - 1.0).abs() < 1e-3,
        "clipped_norm={}",
        clipped_norm
    );
    assert!(norm > 50.0, "norm={} should be large", norm);
}

#[test]
fn test_fused_grad_unscale_clip_inf_detection_cpu() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[1.0f32, f32::INFINITY, 3.0, 4.0], &[4], &device);

    let (_clipped, _norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 1.0).unwrap();

    assert!(found_inf, "should detect infinity");
}

#[test]
fn test_fused_grad_unscale_clip_nan_detection_cpu() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0], &[4], &device);

    let (_clipped, _norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 1.0).unwrap();

    assert!(found_inf, "should detect NaN");
}

// ---- Dynamic loss scale parity ----

#[test]
fn test_dynamic_loss_scale_update_growth_cpu() {
    let (client, _device) = setup_cpu();

    // Not at growth interval yet — scale unchanged, tracker increments
    let (scale, tracker) = client
        .dynamic_loss_scale_update(false, 1024.0, 10, 500, 0.5)
        .unwrap();
    assert!((scale - 1024.0).abs() < 1e-10, "scale={}", scale);
    assert_eq!(tracker, 11);

    // At growth interval → double scale, reset tracker
    let (scale, tracker) = client
        .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
        .unwrap();
    assert!((scale - 2048.0).abs() < 1e-10, "scale={}", scale);
    assert_eq!(tracker, 0);
}

#[test]
fn test_dynamic_loss_scale_update_backoff_cpu() {
    let (client, _device) = setup_cpu();

    let (scale, tracker) = client
        .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
        .unwrap();
    assert!((scale - 512.0).abs() < 1e-10, "scale={}", scale);
    assert_eq!(tracker, 0);
}

// ---- CUDA parity ----

#[cfg(feature = "cuda")]
#[test]
fn test_fused_grad_unscale_clip_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let shape = [8];
    let max_norm = 10.0;
    let loss_scale = 2.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device);
        let (cuda_out, cuda_norm, cuda_inf) = cuda_client
            .fused_grad_unscale_clip(&cuda_grad, max_norm, loss_scale)
            .unwrap();

        assert_eq!(cpu_inf, cuda_inf, "found_inf mismatch");
        assert!(
            (cpu_norm - cuda_norm).abs() < 1e-3,
            "norm mismatch: {} vs {}",
            cpu_norm,
            cuda_norm
        );
        assert_parity_f32(
            &cpu_out.to_vec::<f32>(),
            &cuda_out.to_vec::<f32>(),
            "fused_grad_unscale_clip",
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_grad_unscale_clip_inf_cuda() {
    use super::helpers::with_cuda_backend;

    with_cuda_backend(|cuda_client, cuda_device| {
        let grad_data = [1.0f32, f32::INFINITY, 3.0, 4.0];
        let cuda_grad = Tensor::from_slice(&grad_data, &[4], &cuda_device);
        let (_out, _norm, found_inf) = cuda_client
            .fused_grad_unscale_clip(&cuda_grad, 1.0, 1.0)
            .unwrap();
        assert!(found_inf, "CUDA should detect infinity");
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_dynamic_loss_scale_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, _) = setup_cpu();

    // Growth case
    let (cpu_scale, cpu_tracker) = cpu_client
        .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
        .unwrap();

    with_cuda_backend(|cuda_client, _cuda_device| {
        let (cuda_scale, cuda_tracker) = cuda_client
            .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
            .unwrap();
        assert!((cpu_scale - cuda_scale).abs() < 1e-10);
        assert_eq!(cpu_tracker, cuda_tracker);
    });

    // Backoff case
    let (cpu_scale, cpu_tracker) = cpu_client
        .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
        .unwrap();

    with_cuda_backend(|cuda_client, _cuda_device| {
        let (cuda_scale, cuda_tracker) = cuda_client
            .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
            .unwrap();
        assert!((cpu_scale - cuda_scale).abs() < 1e-10);
        assert_eq!(cpu_tracker, cuda_tracker);
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_grad_unscale_clip_large_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let n = 1024;
    let grad_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 10.0).collect();
    let shape = [n];
    let max_norm = 5.0;
    let loss_scale = 4.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device);
        let (cuda_out, cuda_norm, cuda_inf) = cuda_client
            .fused_grad_unscale_clip(&cuda_grad, max_norm, loss_scale)
            .unwrap();

        assert_eq!(cpu_inf, cuda_inf, "found_inf mismatch");
        assert!(
            (cpu_norm - cuda_norm).abs() < 0.1,
            "norm mismatch: {} vs {}",
            cpu_norm,
            cuda_norm
        );
        assert_parity_f32(
            &cpu_out.to_vec::<f32>(),
            &cuda_out.to_vec::<f32>(),
            "fused_grad_unscale_clip large",
        );
    });
}

// ---- WebGPU parity ----

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_grad_unscale_clip_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let shape = [8];
    let max_norm = 10.0;
    let loss_scale = 2.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_grad = Tensor::from_slice(&grad_data, &shape, &wgpu_device);
        let (wgpu_out, wgpu_norm, wgpu_inf) = wgpu_client
            .fused_grad_unscale_clip(&wgpu_grad, max_norm, loss_scale)
            .unwrap();

        assert_eq!(cpu_inf, wgpu_inf, "found_inf mismatch");
        assert!(
            (cpu_norm - wgpu_norm).abs() < 1e-2,
            "norm mismatch: {} vs {}",
            cpu_norm,
            wgpu_norm
        );
        assert_parity_f32(
            &cpu_out.to_vec::<f32>(),
            &wgpu_out.to_vec::<f32>(),
            "fused_grad_unscale_clip wgpu",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_dynamic_loss_scale_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, _) = setup_cpu();

    // Growth case
    let (cpu_scale, cpu_tracker) = cpu_client
        .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
        .unwrap();

    with_wgpu_backend(|wgpu_client, _wgpu_device| {
        let (wgpu_scale, wgpu_tracker) = wgpu_client
            .dynamic_loss_scale_update(false, 1024.0, 499, 500, 0.5)
            .unwrap();
        assert!((cpu_scale - wgpu_scale).abs() < 1e-10);
        assert_eq!(cpu_tracker, wgpu_tracker);
    });

    // Backoff case
    let (cpu_scale, cpu_tracker) = cpu_client
        .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
        .unwrap();

    with_wgpu_backend(|wgpu_client, _wgpu_device| {
        let (wgpu_scale, wgpu_tracker) = wgpu_client
            .dynamic_loss_scale_update(true, 1024.0, 100, 500, 0.5)
            .unwrap();
        assert!((cpu_scale - wgpu_scale).abs() < 1e-10);
        assert_eq!(cpu_tracker, wgpu_tracker);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_grad_unscale_clip_large_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let n = 1024;
    let grad_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 10.0).collect();
    let shape = [n];
    let max_norm = 5.0;
    let loss_scale = 4.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_grad = Tensor::from_slice(&grad_data, &shape, &wgpu_device);
        let (wgpu_out, wgpu_norm, wgpu_inf) = wgpu_client
            .fused_grad_unscale_clip(&wgpu_grad, max_norm, loss_scale)
            .unwrap();

        assert_eq!(cpu_inf, wgpu_inf, "found_inf mismatch");
        assert!(
            (cpu_norm - wgpu_norm).abs() < 0.1,
            "norm mismatch: {} vs {}",
            cpu_norm,
            wgpu_norm
        );
        assert_parity_f32(
            &cpu_out.to_vec::<f32>(),
            &wgpu_out.to_vec::<f32>(),
            "fused_grad_unscale_clip large wgpu",
        );
    });
}
