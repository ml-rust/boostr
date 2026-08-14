//! Backend parity for the ST-MoE router z-loss.
//!
//! Regression: a real training run produced a finite z-loss on CPU and NaN on
//! CUDA, while every individual op's parity test passed. The composite
//! (log_softmax -> sub -> mean -> pow -> mean) is what diverges, so it needs its
//! own parity coverage.

use super::helpers::*;
use boostr::nn::loss::router_z_loss;
use numr::autograd::Var;
use numr::tensor::Tensor;

/// Logits shaped like a real router: [num_tokens, num_experts].
fn logits_data(num_tokens: usize, num_experts: usize) -> Vec<f32> {
    (0..num_tokens * num_experts)
        .map(|i| (i as f32 * 0.37).sin() * 0.8)
        .collect()
}

#[test]
fn test_router_z_loss_cpu_is_finite() {
    let (client, device) = setup_cpu();
    let (num_tokens, num_experts) = (384, 4);
    let data = logits_data(num_tokens, num_experts);
    let logits = Var::new(
        Tensor::from_slice(&data, &[num_tokens, num_experts], &device),
        true,
    );

    let loss = router_z_loss(&client, &logits).expect("cpu router_z_loss");
    let v: f32 = loss.tensor().item().expect("scalar");
    assert!(v.is_finite(), "cpu z-loss must be finite, got {v}");
}

/// Large-magnitude logits: log_softmax must subtract the row max, or `exp`
/// overflows to inf and the z-loss becomes NaN. numr requires parity tests to
/// cover large values explicitly.
#[cfg(feature = "cuda")]
#[test]
fn test_router_z_loss_parity_large_logits() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (num_tokens, num_experts) = (64, 4);
    let data: Vec<f32> = (0..num_tokens * num_experts)
        .map(|i| (i as f32 * 0.37).sin() * 120.0)
        .collect();

    let cpu_logits = Var::new(
        Tensor::from_slice(&data, &[num_tokens, num_experts], &cpu_device),
        true,
    );
    let cpu_loss = router_z_loss(&cpu_client, &cpu_logits).expect("cpu router_z_loss");
    let cpu_v: f32 = cpu_loss.tensor().item().expect("cpu scalar");
    assert!(cpu_v.is_finite(), "cpu z-loss must be finite, got {cpu_v}");

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_logits = Var::new(
            Tensor::from_slice(&data, &[num_tokens, num_experts], &cuda_device),
            true,
        );
        let cuda_loss = router_z_loss(&cuda_client, &cuda_logits).expect("cuda router_z_loss");
        let cuda_v: f32 = cuda_loss.tensor().item().expect("cuda scalar");
        assert!(
            cuda_v.is_finite(),
            "cuda z-loss is not finite ({cuda_v}) on large logits while cpu is {cpu_v}"
        );
        assert_parity_f32_tol(&[cuda_v], &[cpu_v], "router_z_loss_large", 1e-3, 1e-4);
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_router_z_loss_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (num_tokens, num_experts) = (384, 4);
    let data = logits_data(num_tokens, num_experts);

    let cpu_logits = Var::new(
        Tensor::from_slice(&data, &[num_tokens, num_experts], &cpu_device),
        true,
    );
    let cpu_loss = router_z_loss(&cpu_client, &cpu_logits).expect("cpu router_z_loss");
    let cpu_v: f32 = cpu_loss.tensor().item().expect("cpu scalar");
    assert!(cpu_v.is_finite(), "cpu z-loss must be finite, got {cpu_v}");

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_logits = Var::new(
            Tensor::from_slice(&data, &[num_tokens, num_experts], &cuda_device),
            true,
        );
        let cuda_loss = router_z_loss(&cuda_client, &cuda_logits).expect("cuda router_z_loss");
        let cuda_v: f32 = cuda_loss.tensor().item().expect("cuda scalar");

        assert!(
            cuda_v.is_finite(),
            "cuda z-loss is not finite ({cuda_v}) while cpu is {cpu_v}"
        );
        assert_parity_f32_tol(&[cuda_v], &[cpu_v], "router_z_loss", 1e-4, 1e-6);
    });
}
