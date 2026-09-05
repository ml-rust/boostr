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
    let grad = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[4], &device).unwrap();

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
    let grad = Tensor::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device).unwrap();

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
    let grad = Tensor::from_slice(&[20.0f32, 40.0, 60.0, 80.0], &[4], &device).unwrap();

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
    let grad = Tensor::from_slice(&[1.0f32, f32::INFINITY, 3.0, 4.0], &[4], &device).unwrap();

    let (_clipped, _norm, found_inf) = client.fused_grad_unscale_clip(&grad, 1.0, 1.0).unwrap();

    assert!(found_inf, "should detect infinity");
}

#[test]
fn test_fused_grad_unscale_clip_nan_detection_cpu() {
    let (client, device) = setup_cpu();
    let grad = Tensor::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0], &[4], &device).unwrap();

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

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device).unwrap();
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
        let cuda_grad = Tensor::from_slice(&grad_data, &[4], &cuda_device).unwrap();
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

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device).unwrap();
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

#[cfg(feature = "cuda")]
#[test]
fn test_fused_grad_unscale_clip_f64_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    // 600 elements at 256 threads/block spans 3 blocks (256 + 256 + 88),
    // exercising the stage-1 atomicAdd across a multi-block reduction.
    let n = 600;
    let grad_data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.137).sin() * 10.0).collect();
    let shape = [n];
    let max_norm = 5.0;
    let loss_scale = 4.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();
    // unscaled = grad / 4 = sin(..)*2.5; sin^2 averages ~0.5 over 600 samples,
    // so norm² ≈ 600 * 2.5² * 0.5 ≈ 1875, norm ≈ 43 — well above max_norm=5,
    // so clip_scale_f64 (stage 2) actually runs instead of early-returning.
    assert!(
        cpu_norm > max_norm,
        "fixture must exceed max_norm to exercise clip_scale; got {cpu_norm}"
    );

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device).unwrap();
        let (cuda_out, cuda_norm, cuda_inf) = cuda_client
            .fused_grad_unscale_clip(&cuda_grad, max_norm, loss_scale)
            .unwrap();

        assert_eq!(cpu_inf, cuda_inf, "found_inf mismatch");
        // The CUDA f64 kernel accumulates norm_sq in FLOAT by design ("norm_sq
        // is f32 (sufficient precision for loss scale decisions)" in
        // fused_grad_unscale_clip.cu), while the CPU reference sums in f64. So
        // the norm agrees to f32 precision, not f64, and the clip factor
        // derived from it carries that error into every element.
        assert!(
            (cpu_norm - cuda_norm).abs() <= 1e-6 * cpu_norm.abs(),
            "norm mismatch: {} vs {}",
            cpu_norm,
            cuda_norm
        );

        let cpu_data = cpu_out.to_vec::<f64>();
        let cuda_data = cuda_out.to_vec::<f64>();
        assert_eq!(cpu_data.len(), cuda_data.len());
        for (i, (a, b)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            let diff = (a - b).abs();
            // f32-precision norm (see the comment on the norm assert above),
            // not f64, so the clip factor and thus every element agree only to
            // about f32 epsilon.
            let tol = 1e-9 + 1e-6 * b.abs();
            assert!(
                diff <= tol,
                "fused_grad_unscale_clip f64 at {i}: {a} vs {b} (diff={diff}, tol={tol})"
            );
        }
    });
}

// The CPU reference (`src/ops/cpu/training/fused_fp8.rs`) only implements F32
// and F64 — F16/BF16 hit `Error::InvalidArgument` there — so there is no
// same-dtype CPU baseline for the narrow dtypes below. Following the
// cast-and-compare shape used by `src/optimizer/grad_clip.rs`'s narrow-dtype
// tests: build an F32 fixture, take the CPU F32 result as ground truth, run
// CUDA on the same fixture cast to F16/BF16, cast the CUDA result back to F32,
// and compare with a tolerance sized to the narrow dtype's mantissa width.

#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn test_fused_grad_unscale_clip_f16_cuda_parity() {
    use super::helpers::{assert_parity_f32_tol, with_cuda_backend};
    use numr::dtype::DType;

    let (cpu_client, cpu_device) = setup_cpu();
    let n = 600; // 3 blocks of 256 threads (256 + 256 + 88)
    let grad_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 10.0).collect();
    let shape = [n];
    let max_norm = 5.0;
    let loss_scale = 4.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();
    // Same fixture as the F64 case: norm ≈ 43, well above max_norm=5, so
    // clip_scale_f16 is exercised, not early-returned.
    assert!(
        cpu_norm > max_norm,
        "fixture must exceed max_norm to exercise clip_scale; got {cpu_norm}"
    );

    with_cuda_backend(|cuda_client, cuda_device| {
        let f32_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device).unwrap();
        let f16_grad = f32_grad.to_dtype(DType::F16).unwrap();
        let (f16_out, f16_norm, f16_inf) = cuda_client
            .fused_grad_unscale_clip(&f16_grad, max_norm, loss_scale)
            .unwrap();
        let cuda_out = f16_out.to_dtype(DType::F32).unwrap();

        assert_eq!(cpu_inf, f16_inf, "found_inf mismatch");
        // F16 keeps 10 explicit mantissa bits (~2^-11 ≈ 5e-4 relative
        // precision per stored value); 5% covers per-element rounding
        // compounded through the norm reduction and the clip multiply.
        assert!(
            (cpu_norm - f16_norm).abs() < 5e-2 * cpu_norm,
            "norm mismatch: {} vs {}",
            cpu_norm,
            f16_norm
        );
        assert_parity_f32_tol(
            &cpu_out.to_vec::<f32>(),
            &cuda_out.to_vec::<f32>(),
            "fused_grad_unscale_clip f16",
            5e-2,
            5e-3,
        );
    });
}

#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn test_fused_grad_unscale_clip_bf16_cuda_parity() {
    use super::helpers::{assert_parity_f32_tol, with_cuda_backend};
    use numr::dtype::DType;

    let (cpu_client, cpu_device) = setup_cpu();
    let n = 600; // 3 blocks of 256 threads (256 + 256 + 88)
    let grad_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 10.0).collect();
    let shape = [n];
    let max_norm = 5.0;
    let loss_scale = 4.0;

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();
    assert!(
        cpu_norm > max_norm,
        "fixture must exceed max_norm to exercise clip_scale; got {cpu_norm}"
    );

    with_cuda_backend(|cuda_client, cuda_device| {
        let f32_grad = Tensor::from_slice(&grad_data, &shape, &cuda_device).unwrap();
        let bf16_grad = f32_grad.to_dtype(DType::BF16).unwrap();
        let (bf16_out, bf16_norm, bf16_inf) = cuda_client
            .fused_grad_unscale_clip(&bf16_grad, max_norm, loss_scale)
            .unwrap();
        let cuda_out = bf16_out.to_dtype(DType::F32).unwrap();

        assert_eq!(cpu_inf, bf16_inf, "found_inf mismatch");
        // BF16 keeps only 8 explicit mantissa bits (~2^-9 ≈ 2e-3 relative
        // precision per stored value) — half the mantissa of F16, so the
        // tolerance band is widened accordingly.
        assert!(
            (cpu_norm - bf16_norm).abs() < 1e-1 * cpu_norm,
            "norm mismatch: {} vs {}",
            cpu_norm,
            bf16_norm
        );
        assert_parity_f32_tol(
            &cpu_out.to_vec::<f32>(),
            &cuda_out.to_vec::<f32>(),
            "fused_grad_unscale_clip bf16",
            1e-1,
            1e-2,
        );
    });
}

#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn test_fused_grad_unscale_clip_f16_inf_cuda() {
    use super::helpers::with_cuda_backend;
    use numr::dtype::DType;

    with_cuda_backend(|cuda_client, cuda_device| {
        let grad_data = [1.0f32, f32::INFINITY, 3.0, 4.0];
        let f32_grad = Tensor::from_slice(&grad_data, &[4], &cuda_device).unwrap();
        let f16_grad = f32_grad.to_dtype(DType::F16).unwrap();
        let (f16_out, _norm, found_inf) = cuda_client
            .fused_grad_unscale_clip(&f16_grad, 1.0, 1.0)
            .unwrap();

        assert!(found_inf, "CUDA F16 should detect infinity");
        // found_inf takes the early-return path in clip_scale_f16 — data must
        // come back unscaled (still holding the original, un-clipped values).
        let out_f32 = f16_out.to_dtype(DType::F32).unwrap().to_vec::<f32>();
        assert!((out_f32[0] - 1.0).abs() < 1e-2, "out[0]={}", out_f32[0]);
        assert!(out_f32[1].is_infinite(), "out[1] should stay infinite");
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

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_grad = Tensor::from_slice(&grad_data, &shape, &wgpu_device).unwrap();
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

    let cpu_grad = Tensor::from_slice(&grad_data, &shape, &cpu_device).unwrap();
    let (cpu_out, cpu_norm, cpu_inf) = cpu_client
        .fused_grad_unscale_clip(&cpu_grad, max_norm, loss_scale)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_grad = Tensor::from_slice(&grad_data, &shape, &wgpu_device).unwrap();
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
