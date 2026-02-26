//! Integration tests verifying RoPE backward pass (gradient flow) for all variants.
//!
//! All three RoPE variants (standard, interleaved, YaRN) compose autograd ops
//! (var_mul, var_sub, var_add, var_cat, var_narrow), so backward is automatic.
//! These tests verify gradients exist, are non-zero, and are numerically correct
//! via finite-difference checks.

use boostr::ops::traits::position::rope::RoPEOps;
use numr::autograd::{Var, backward, var_sum};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn det_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect()
}

/// Helper: compute scalar loss = sum(f(x)) for finite-difference gradient check.
///
/// Applies the specified RoPE variant to `x_data` and returns `sum(output)`,
/// which is used as the loss in a two-point finite-difference approximation.
#[allow(clippy::too_many_arguments)]
fn compute_loss(
    client: &CpuClient,
    device: &CpuDevice,
    x_data: &[f32],
    cos_data: &[f32],
    sin_data: &[f32],
    shape: &[usize],
    variant: &str,
    attn_scale: f32,
) -> TestResult<f32> {
    let x = Var::<CpuRuntime>::new(Tensor::from_slice(x_data, shape, device), false);
    let (s, d) = (shape[2], shape[3]);
    let cos = Var::<CpuRuntime>::new(Tensor::from_slice(cos_data, &[s, d / 2], device), false);
    let sin = Var::<CpuRuntime>::new(Tensor::from_slice(sin_data, &[s, d / 2], device), false);

    let out = match variant {
        "standard" => client.apply_rope(&x, &cos, &sin)?,
        "interleaved" => client.apply_rope_interleaved(&x, &cos, &sin)?,
        "yarn" => client.apply_rope_yarn(&x, &cos, &sin, attn_scale)?,
        _ => panic!("unknown variant: {variant}"),
    };

    let out_vec = out.tensor().to_vec::<f32>();
    Ok(out_vec.iter().sum::<f32>())
}

/// Verify gradients via backward and finite-difference check.
fn verify_rope_backward(variant: &str, attn_scale: f32) -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 4, 8);
    let shape = [b, h, s, d];
    let n = b * h * s * d;

    let x_data = det_data(n);
    let cos_data: Vec<f32> = (0..s * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin_data: Vec<f32> = (0..s * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();

    // Forward + backward
    let x = Var::<CpuRuntime>::new(
        Tensor::from_slice(&x_data, &shape, &device),
        true, // requires_grad
    );
    let cos = Var::<CpuRuntime>::new(Tensor::from_slice(&cos_data, &[s, d / 2], &device), false);
    let sin = Var::<CpuRuntime>::new(Tensor::from_slice(&sin_data, &[s, d / 2], &device), false);

    let out = match variant {
        "standard" => client.apply_rope(&x, &cos, &sin)?,
        "interleaved" => client.apply_rope_interleaved(&x, &cos, &sin)?,
        "yarn" => client.apply_rope_yarn(&x, &cos, &sin, attn_scale)?,
        _ => panic!("unknown variant: {variant}"),
    };

    // loss = sum(out)
    let loss = var_sum(&out, &[0, 1, 2, 3], false, &client)?;
    let grads = backward(&loss, &client)?;

    // Verify gradient exists for x
    let x_grad = grads
        .get(x.tensor().id())
        .ok_or_else(|| format!("{variant}: no gradient for input x"))?;
    let grad_vec = x_grad.to_vec::<f32>();

    // Verify gradients are non-zero
    let grad_norm: f32 = grad_vec.iter().map(|g| g * g).sum::<f32>();
    assert!(
        grad_norm > 1e-10,
        "{variant}: gradient is all zeros (norm={grad_norm})"
    );

    // Finite-difference check
    let eps = 1e-4f32;
    for idx in [0, n / 4, n / 2, 3 * n / 4, n - 1] {
        let mut x_plus = x_data.clone();
        let mut x_minus = x_data.clone();
        x_plus[idx] += eps;
        x_minus[idx] -= eps;

        let loss_plus = compute_loss(
            &client, &device, &x_plus, &cos_data, &sin_data, &shape, variant, attn_scale,
        )?;
        let loss_minus = compute_loss(
            &client, &device, &x_minus, &cos_data, &sin_data, &shape, variant, attn_scale,
        )?;

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical_grad = grad_vec[idx];

        let diff = (numerical_grad - analytical_grad).abs();
        let tol = 5e-3 * numerical_grad.abs().max(analytical_grad.abs()).max(1.0);
        assert!(
            diff < tol,
            "{variant}: grad mismatch at idx {idx}: analytical={analytical_grad}, numerical={numerical_grad}, diff={diff}"
        );
    }

    Ok(())
}

#[test]
fn test_rope_standard_backward() -> TestResult {
    verify_rope_backward("standard", 1.0)
}

#[test]
fn test_rope_interleaved_backward() -> TestResult {
    verify_rope_backward("interleaved", 1.0)
}

#[test]
fn test_rope_yarn_backward() -> TestResult {
    verify_rope_backward("yarn", 1.0826)
}

#[test]
fn test_rope_yarn_unit_scale_backward() -> TestResult {
    // With scale=1.0, yarn should behave like standard for backward
    verify_rope_backward("yarn", 1.0)
}
