//! F32 happy-path tests for the CPU `FusedOptimizerOps` kernels — no feature gate.

mod common;

use boostr::FusedOptimizerOps;
use common::cpu_setup;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_fused_adamw_basic() {
    let (client, device) = cpu_setup();
    let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device).unwrap();
    let m = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();
    let v = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();

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
    let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();

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

    let p1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    let g1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
    let m1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
    let v1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();

    let p2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &device).unwrap();
    let g2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4, 0.5], &[3], &device).unwrap();
    let m2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device).unwrap();
    let v2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device).unwrap();

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
    let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
    let accum = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();

    let (new_p, new_acc) = client
        .fused_adagrad_step(&param, &grad, &accum, 0.1, 1e-10, 0.0)
        .unwrap();

    let p = new_p.to_vec::<f32>();
    assert!((p[0] - 0.9).abs() < 1e-4);
    assert!((p[1] - 1.9).abs() < 1e-4);
    assert!(new_acc.to_vec::<f32>()[0] > 0.0);
}
