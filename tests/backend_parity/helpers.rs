//! Shared helpers for boostr backend parity tests.

use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
static CUDA_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
#[cfg(feature = "wgpu")]
static WGPU_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn setup_cpu() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Deterministic pseudo-random tensor using sin-based pattern.
pub fn det_tensor(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device)
}

/// Deterministic I32 tensor (for block tables, cu_seqlens, etc.).
pub fn det_i32_tensor(data: &[i32], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(data, shape, device)
}

/// Relaxed parity check for backward passes (atomicAdd causes FP non-determinism).
pub fn assert_parity_f32_relaxed(a: &[f32], b: &[f32], op: &str) {
    assert_parity_f32_tol(a, b, op, 1e-4, 1e-5);
}

pub fn assert_parity_f32_tol(a: &[f32], b: &[f32], op: &str, rtol: f32, atol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        if diff > tol {
            panic!(
                "parity_f32[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

pub fn assert_parity_f32(a: &[f32], b: &[f32], op: &str) {
    let rtol = 1e-5f32;
    let atol = 1e-7f32;
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        if diff > tol {
            panic!(
                "parity_f32[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

#[cfg(feature = "cuda")]
pub fn with_cuda_backend<F>(mut f: F)
where
    F: FnMut(numr::runtime::cuda::CudaClient, numr::runtime::cuda::CudaDevice),
{
    let _guard = CUDA_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!("CUDA feature enabled but runtime unavailable, skipping");
        return;
    }
    let device = numr::runtime::cuda::CudaDevice::new(0);
    let client = match numr::runtime::cuda::CudaClient::new(device.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create CudaClient: {:?}, skipping", e);
            return;
        }
    };
    f(client, device);
}

#[cfg(feature = "wgpu")]
pub fn with_wgpu_backend<F>(mut f: F)
where
    F: FnMut(numr::runtime::wgpu::WgpuClient, numr::runtime::wgpu::WgpuDevice),
{
    let _guard = WGPU_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let device = numr::runtime::wgpu::WgpuDevice::new(0);
    let client = match numr::runtime::wgpu::WgpuClient::new(device.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create WgpuClient: {:?}, skipping", e);
            return;
        }
    };
    f(client, device);
}

/// Reference standard attention: softmax(Q @ K^T / sqrt(d)) @ V
/// Used to verify flash attention output against a naive O(N²) baseline.
pub fn reference_attention(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    causal: bool,
) -> Tensor<CpuRuntime> {
    let head_dim = q.shape()[3];
    let seq_len_q = q.shape()[2];
    let seq_len_k = k.shape()[2];
    let scale = (head_dim as f64).sqrt().recip();

    let k_t = k.transpose(-2, -1).unwrap().contiguous();
    let scores = client.matmul(q, &k_t).unwrap();
    let scores = client.mul_scalar(&scores, scale).unwrap();

    let scores = if causal {
        let mask_data: Vec<f32> = (0..seq_len_q * seq_len_k)
            .map(|idx| {
                let i = idx / seq_len_k;
                let j = idx % seq_len_k;
                if j <= i { 0.0 } else { -1e9 }
            })
            .collect();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&mask_data, &[1, 1, seq_len_q, seq_len_k], q.device());
        client.add(&scores, &mask).unwrap()
    } else {
        scores
    };

    let weights = client.softmax(&scores, -1).unwrap();
    client.matmul(&weights, v).unwrap()
}

/// Compute the maximum absolute difference between two CPU tensors.
pub fn max_abs_diff(client: &CpuClient, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
    let diff = client.sub(a, b).unwrap();
    let abs_diff = client.abs(&diff).unwrap();
    let max = client.max(&abs_diff, &[], false).unwrap();
    max.to_vec::<f32>()[0]
}
