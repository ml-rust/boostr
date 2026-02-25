//! Backend parity tests for FusedOptimizerOps
//!
//! Verifies that CPU, CUDA, and WebGPU produce numerically identical results
//! for all fused optimizer operations.

use super::helpers::{assert_parity_f32, setup_cpu};
use boostr::FusedOptimizerOps;
use numr::dtype::DType;
use numr::tensor::Tensor;

// ---- AdamW parity ----

#[test]
fn test_fused_adamw_cpu_reference() {
    let (client, device) = setup_cpu();
    let param = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let grad = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device);
    let m = Tensor::zeros(&[4], DType::F32, &device);
    let v = Tensor::zeros(&[4], DType::F32, &device);

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

    // Verify m = beta1*0 + (1-beta1)*grad = 0.1*grad
    let m_data = new_m.to_vec::<f32>();
    assert!((m_data[0] - 0.01).abs() < 1e-6, "m[0]={}", m_data[0]);
    assert!((m_data[1] - 0.02).abs() < 1e-6, "m[1]={}", m_data[1]);

    // Verify v = beta2*0 + (1-beta2)*grad^2 = 0.001*grad^2
    let v_data = new_v.to_vec::<f32>();
    assert!(
        (v_data[0] - 0.001 * 0.01).abs() < 1e-8,
        "v[0]={}",
        v_data[0]
    );

    // Param should decrease
    let p_data = new_p.to_vec::<f32>();
    assert!(p_data[0] < 1.0, "p[0] should decrease: {}", p_data[0]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_adamw_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let param_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let shape = [8];

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let m_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);
    let v_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let lr = 1e-3;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let wd = 0.01;
    let bc1 = 1.0 - beta1;
    let bc2 = (1.0_f64 - beta2).sqrt();
    let step_size = lr * bc2 / bc1;

    let (cpu_p, cpu_m, cpu_v) = cpu_client
        .fused_adamw_step(
            &param_cpu, &grad_cpu, &m_cpu, &v_cpu, lr, beta1, beta2, eps, wd, step_size,
        )
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let param_cuda = Tensor::from_slice(&param_data, &shape, &cuda_device);
        let grad_cuda = Tensor::from_slice(&grad_data, &shape, &cuda_device);
        let m_cuda = Tensor::zeros(&shape, DType::F32, &cuda_device);
        let v_cuda = Tensor::zeros(&shape, DType::F32, &cuda_device);

        let (cuda_p, cuda_m, cuda_v) = cuda_client
            .fused_adamw_step(
                &param_cuda,
                &grad_cuda,
                &m_cuda,
                &v_cuda,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                step_size,
            )
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &cuda_p.to_vec::<f32>(),
            "fused_adamw param",
        );
        assert_parity_f32(
            &cpu_m.to_vec::<f32>(),
            &cuda_m.to_vec::<f32>(),
            "fused_adamw m",
        );
        assert_parity_f32(
            &cpu_v.to_vec::<f32>(),
            &cuda_v.to_vec::<f32>(),
            "fused_adamw v",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_adamw_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let param_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let shape = [8];

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let m_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);
    let v_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let lr = 1e-3;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let wd = 0.01;
    let bc1 = 1.0 - beta1;
    let bc2 = (1.0_f64 - beta2).sqrt();
    let step_size = lr * bc2 / bc1;

    let (cpu_p, cpu_m, cpu_v) = cpu_client
        .fused_adamw_step(
            &param_cpu, &grad_cpu, &m_cpu, &v_cpu, lr, beta1, beta2, eps, wd, step_size,
        )
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let param_wgpu = Tensor::from_slice(&param_data, &shape, &wgpu_device);
        let grad_wgpu = Tensor::from_slice(&grad_data, &shape, &wgpu_device);
        let m_wgpu = Tensor::zeros(&shape, DType::F32, &wgpu_device);
        let v_wgpu = Tensor::zeros(&shape, DType::F32, &wgpu_device);

        let (wgpu_p, wgpu_m, wgpu_v) = wgpu_client
            .fused_adamw_step(
                &param_wgpu,
                &grad_wgpu,
                &m_wgpu,
                &v_wgpu,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                step_size,
            )
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &wgpu_p.to_vec::<f32>(),
            "fused_adamw param wgpu",
        );
        assert_parity_f32(
            &cpu_m.to_vec::<f32>(),
            &wgpu_m.to_vec::<f32>(),
            "fused_adamw m wgpu",
        );
        assert_parity_f32(
            &cpu_v.to_vec::<f32>(),
            &wgpu_v.to_vec::<f32>(),
            "fused_adamw v wgpu",
        );
    });
}

// ---- SGD parity ----

#[test]
fn test_fused_sgd_cpu_reference() {
    let (client, device) = setup_cpu();
    let param = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let grad = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device);

    // Vanilla SGD (no momentum)
    let (new_p, _buf) = client
        .fused_sgd_step(&param, &grad, None, 0.1, 0.0, 0.0, 0.0, false)
        .unwrap();

    let p = new_p.to_vec::<f32>();
    assert!((p[0] - 0.99).abs() < 1e-6);
    assert!((p[1] - 1.98).abs() < 1e-6);
}

#[test]
fn test_fused_sgd_momentum_cpu() {
    let (client, device) = setup_cpu();
    let param = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device);
    let grad = Tensor::from_slice(&[0.1f32, 0.2], &[2], &device);

    // First step (no buf)
    let (new_p, buf) = client
        .fused_sgd_step(&param, &grad, None, 0.1, 0.9, 0.0, 0.0, false)
        .unwrap();

    // buf = grad for first step
    let buf_data = buf.to_vec::<f32>();
    assert!((buf_data[0] - 0.1).abs() < 1e-6);

    // Second step (has buf)
    let grad2 = Tensor::from_slice(&[0.2f32, 0.3], &[2], &device);
    let (new_p2, buf2) = client
        .fused_sgd_step(&new_p, &grad2, Some(&buf), 0.1, 0.9, 0.0, 0.0, false)
        .unwrap();

    // buf = 0.9 * 0.1 + 1.0 * 0.2 = 0.29
    let buf2_data = buf2.to_vec::<f32>();
    assert!((buf2_data[0] - 0.29).abs() < 1e-5, "buf={}", buf2_data[0]);

    let p2 = new_p2.to_vec::<f32>();
    assert!(p2[0] < new_p.to_vec::<f32>()[0], "param should decrease");
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_sgd_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);

    let (cpu_p, cpu_buf) = cpu_client
        .fused_sgd_step(&param_cpu, &grad_cpu, None, 0.1, 0.9, 0.0, 0.01, false)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let param_cuda = Tensor::from_slice(&param_data, &shape, &cuda_device);
        let grad_cuda = Tensor::from_slice(&grad_data, &shape, &cuda_device);

        let (cuda_p, cuda_buf) = cuda_client
            .fused_sgd_step(&param_cuda, &grad_cuda, None, 0.1, 0.9, 0.0, 0.01, false)
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &cuda_p.to_vec::<f32>(),
            "fused_sgd param",
        );
        assert_parity_f32(
            &cpu_buf.to_vec::<f32>(),
            &cuda_buf.to_vec::<f32>(),
            "fused_sgd buf",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_sgd_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);

    let (cpu_p, cpu_buf) = cpu_client
        .fused_sgd_step(&param_cpu, &grad_cpu, None, 0.1, 0.9, 0.0, 0.01, false)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let param_wgpu = Tensor::from_slice(&param_data, &shape, &wgpu_device);
        let grad_wgpu = Tensor::from_slice(&grad_data, &shape, &wgpu_device);

        let (wgpu_p, wgpu_buf) = wgpu_client
            .fused_sgd_step(&param_wgpu, &grad_wgpu, None, 0.1, 0.9, 0.0, 0.01, false)
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &wgpu_p.to_vec::<f32>(),
            "fused_sgd param wgpu",
        );
        assert_parity_f32(
            &cpu_buf.to_vec::<f32>(),
            &wgpu_buf.to_vec::<f32>(),
            "fused_sgd buf wgpu",
        );
    });
}

// ---- AdaGrad parity ----

#[test]
fn test_fused_adagrad_cpu_reference() {
    let (client, device) = setup_cpu();
    let param = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device);
    let grad = Tensor::from_slice(&[0.1f32, 0.2], &[2], &device);
    let accum = Tensor::zeros(&[2], DType::F32, &device);

    let (new_p, new_acc) = client
        .fused_adagrad_step(&param, &grad, &accum, 0.1, 1e-10, 0.0)
        .unwrap();

    let p = new_p.to_vec::<f32>();
    // accum = grad^2, update = lr * grad / sqrt(grad^2) = lr * sign(grad)
    assert!((p[0] - 0.9).abs() < 1e-4, "p[0]={}", p[0]);
    assert!((p[1] - 1.9).abs() < 1e-4, "p[1]={}", p[1]);

    let a = new_acc.to_vec::<f32>();
    assert!((a[0] - 0.01).abs() < 1e-6, "acc[0]={}", a[0]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_adagrad_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let grad_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let accum_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let (cpu_p, cpu_a) = cpu_client
        .fused_adagrad_step(&param_cpu, &grad_cpu, &accum_cpu, 0.1, 1e-10, 0.01)
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let param_cuda = Tensor::from_slice(&param_data, &shape, &cuda_device);
        let grad_cuda = Tensor::from_slice(&grad_data, &shape, &cuda_device);
        let accum_cuda = Tensor::zeros(&shape, DType::F32, &cuda_device);

        let (cuda_p, cuda_a) = cuda_client
            .fused_adagrad_step(&param_cuda, &grad_cuda, &accum_cuda, 0.1, 1e-10, 0.01)
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &cuda_p.to_vec::<f32>(),
            "fused_adagrad param",
        );
        assert_parity_f32(
            &cpu_a.to_vec::<f32>(),
            &cuda_a.to_vec::<f32>(),
            "fused_adagrad accum",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_adagrad_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let grad_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let accum_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let (cpu_p, cpu_a) = cpu_client
        .fused_adagrad_step(&param_cpu, &grad_cpu, &accum_cpu, 0.1, 1e-10, 0.01)
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let param_wgpu = Tensor::from_slice(&param_data, &shape, &wgpu_device);
        let grad_wgpu = Tensor::from_slice(&grad_data, &shape, &wgpu_device);
        let accum_wgpu = Tensor::zeros(&shape, DType::F32, &wgpu_device);

        let (wgpu_p, wgpu_a) = wgpu_client
            .fused_adagrad_step(&param_wgpu, &grad_wgpu, &accum_wgpu, 0.1, 1e-10, 0.01)
            .unwrap();

        assert_parity_f32(
            &cpu_p.to_vec::<f32>(),
            &wgpu_p.to_vec::<f32>(),
            "fused_adagrad param wgpu",
        );
        assert_parity_f32(
            &cpu_a.to_vec::<f32>(),
            &wgpu_a.to_vec::<f32>(),
            "fused_adagrad accum wgpu",
        );
    });
}

// ---- LAMB parity ----

#[test]
fn test_fused_lamb_cpu_reference() {
    let (client, device) = setup_cpu();
    let param = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let grad = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device);
    let m = Tensor::zeros(&[4], DType::F32, &device);
    let v = Tensor::zeros(&[4], DType::F32, &device);

    let beta1 = 0.9;
    let beta2 = 0.999;
    let bc1 = 1.0 - beta1;
    let bc2 = 1.0 - beta2;

    let (update, new_m, _new_v) = client
        .fused_lamb_step(&param, &grad, &m, &v, beta1, beta2, 1e-6, 0.01, bc1, bc2)
        .unwrap();

    let u = update.to_vec::<f32>();
    let m_data = new_m.to_vec::<f32>();
    // m = 0.1 * grad
    assert!((m_data[0] - 0.01).abs() < 1e-6, "m[0]={}", m_data[0]);
    // update should be finite and non-zero
    assert!(u[0].is_finite() && u[0].abs() > 0.0, "u[0]={}", u[0]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_lamb_cuda_parity() {
    use super::helpers::with_cuda_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let grad_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let m_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);
    let v_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let (cpu_u, cpu_m, cpu_v) = cpu_client
        .fused_lamb_step(
            &param_cpu, &grad_cpu, &m_cpu, &v_cpu, 0.9, 0.999, 1e-6, 0.01, 0.1, 0.001,
        )
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let param_cuda = Tensor::from_slice(&param_data, &shape, &cuda_device);
        let grad_cuda = Tensor::from_slice(&grad_data, &shape, &cuda_device);
        let m_cuda = Tensor::zeros(&shape, DType::F32, &cuda_device);
        let v_cuda = Tensor::zeros(&shape, DType::F32, &cuda_device);

        let (cuda_u, cuda_m, cuda_v) = cuda_client
            .fused_lamb_step(
                &param_cuda,
                &grad_cuda,
                &m_cuda,
                &v_cuda,
                0.9,
                0.999,
                1e-6,
                0.01,
                0.1,
                0.001,
            )
            .unwrap();

        assert_parity_f32(
            &cpu_u.to_vec::<f32>(),
            &cuda_u.to_vec::<f32>(),
            "fused_lamb update",
        );
        assert_parity_f32(
            &cpu_m.to_vec::<f32>(),
            &cuda_m.to_vec::<f32>(),
            "fused_lamb m",
        );
        assert_parity_f32(
            &cpu_v.to_vec::<f32>(),
            &cuda_v.to_vec::<f32>(),
            "fused_lamb v",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fused_lamb_wgpu_parity() {
    use super::helpers::with_wgpu_backend;

    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [8];
    let param_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let grad_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();

    let param_cpu = Tensor::from_slice(&param_data, &shape, &cpu_device);
    let grad_cpu = Tensor::from_slice(&grad_data, &shape, &cpu_device);
    let m_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);
    let v_cpu = Tensor::zeros(&shape, DType::F32, &cpu_device);

    let (cpu_u, cpu_m, cpu_v) = cpu_client
        .fused_lamb_step(
            &param_cpu, &grad_cpu, &m_cpu, &v_cpu, 0.9, 0.999, 1e-6, 0.01, 0.1, 0.001,
        )
        .unwrap();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let param_wgpu = Tensor::from_slice(&param_data, &shape, &wgpu_device);
        let grad_wgpu = Tensor::from_slice(&grad_data, &shape, &wgpu_device);
        let m_wgpu = Tensor::zeros(&shape, DType::F32, &wgpu_device);
        let v_wgpu = Tensor::zeros(&shape, DType::F32, &wgpu_device);

        let (wgpu_u, wgpu_m, wgpu_v) = wgpu_client
            .fused_lamb_step(
                &param_wgpu,
                &grad_wgpu,
                &m_wgpu,
                &v_wgpu,
                0.9,
                0.999,
                1e-6,
                0.01,
                0.1,
                0.001,
            )
            .unwrap();

        assert_parity_f32(
            &cpu_u.to_vec::<f32>(),
            &wgpu_u.to_vec::<f32>(),
            "fused_lamb update wgpu",
        );
        assert_parity_f32(
            &cpu_m.to_vec::<f32>(),
            &wgpu_m.to_vec::<f32>(),
            "fused_lamb m wgpu",
        );
        assert_parity_f32(
            &cpu_v.to_vec::<f32>(),
            &wgpu_v.to_vec::<f32>(),
            "fused_lamb v wgpu",
        );
    });
}
