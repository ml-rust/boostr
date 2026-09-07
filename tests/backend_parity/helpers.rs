//! Shared helpers for boostr backend parity tests.

use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::sync::{Mutex, OnceLock};
use tcf_core::NativeEncoding;

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
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
}

/// Deterministic I32 tensor (for block tables, cu_seqlens, etc.).
pub fn det_i32_tensor(data: &[i32], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
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

/// Cosine floor for a CUDA path that quantizes its activation to Q8_1 while
/// the CPU reference keeps the activation in f32. `tests/gguf_dequant_cpu_cuda_parity.rs`
/// establishes this same floor for the GGUF formats that share the pattern.
pub const COSINE_FLOOR: f64 = 0.999;

/// Gate for a CUDA `quant_matmul` path that quantizes the activation to Q8_1
/// (e.g. the TCF `Q8S32T64` MMQ kernel) against a CPU reference that keeps the
/// activation in f32.
///
/// An element-wise tolerance cannot bound this comparison: the activation's
/// per-element quantization error is scaled by the row's weight magnitudes
/// during the reduction, but the output itself can be much smaller than that
/// through cancellation between terms, so no fixed ratio between the two
/// holds for arbitrary weight bytes. Cosine similarity sidesteps this because
/// it compares direction rather than a per-element bound: a correct-but-lossy
/// result stays within a hair of 1.0, while a decode or accumulation defect
/// (e.g. a wrong plane offset or scale index) scrambles the output direction
/// and drives the score toward 0. `COSINE_FLOOR` sits in the gap between the
/// two.
pub fn assert_cosine_parity(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch: {} vs {}",
        a.len(),
        b.len()
    );

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite(),
            "{label}: index {i} is not finite in the CUDA output: {x}"
        );
        assert!(
            y.is_finite(),
            "{label}: index {i} is not finite in the CPU output: {y}"
        );
        dot += f64::from(x) * f64::from(y);
        norm_a += f64::from(x) * f64::from(x);
        norm_b += f64::from(y) * f64::from(y);
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());

    println!("{label}: cosine={cosine:.6}");

    assert!(
        cosine >= COSINE_FLOOR,
        "{label}: cosine {cosine:.6} is below the {COSINE_FLOOR} floor. Correct-but-lossy \
         results sit near 1.0; a decode or accumulation defect collapses the score toward 0. \
         Raising the floor is never the fix."
    );
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
    // Use default_client to get the SAME client that Tensor::from_slice
    // uses internally (via get_or_create_client). Creating a separate
    // CudaClient::new() would use a different CUDA stream, causing race
    // conditions with async copies.
    use numr::runtime::Runtime;
    let device = numr::runtime::cuda::CudaDevice::new(0);
    let client = numr::runtime::cuda::CudaRuntime::default_client(&device);
    f(client.clone(), device);
    use numr::runtime::RuntimeClient;
    client.synchronize();
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

    let k_t = k.transpose(-2, -1).unwrap().contiguous().unwrap();
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
            Tensor::<CpuRuntime>::from_slice(&mask_data, &[1, 1, seq_len_q, seq_len_k], q.device())
                .unwrap();
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

/// Whether `native` at `m` tokens takes the feature-major MMQ path rather
/// than the f32 GEMV/GEMM tile.
///
/// True when the encoding has a `FeatMajorFormat` AND `m` is at or above
/// `TCF_FEAT_MAJOR_MIN_M` (`quant/cuda/quant_matmul/impl_ops.rs`). The
/// encodings listed here must mirror `feat_major_format` in
/// `quant/cuda/quant_matmul/mmq_feat_major/formats/tcf.rs`, the single
/// mapping site in the library — that function is crate-visible only, so
/// this test cannot call it directly. A missing entry here shows up
/// immediately as a parity failure rather than as a silent wrong gate, so
/// the duplication is self-correcting.
///
/// This does not model the K-multiple or device-capability gates: on a
/// device without `int8_mma_m16n8k32`, or at a K the format's `k_multiple`
/// does not divide, the case actually stays on the f32 path and the cosine
/// gate is merely looser, never wrong.
pub fn takes_mmq_path(native: NativeEncoding, m: usize) -> bool {
    const TCF_FEAT_MAJOR_MIN_M: usize = 2;
    matches!(
        native,
        NativeEncoding::Q8S32T64 | NativeEncoding::Q4AS32DT64
    ) && m >= TCF_FEAT_MAJOR_MIN_M
}
