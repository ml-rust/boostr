//! GPU parity tests for head_dim=256 varlen attention.
//!
//! Validates that the CUDA head_dim=256 forward kernel produces the same output
//! as the CPU reference implementation (which supports arbitrary head_dim).
//!
//! Two tests:
//!   - GQA (num_heads=4, num_kv_heads=2, head_dim=256) vs CPU
//!   - MHA (num_heads=4, num_kv_heads=4, head_dim=256) vs CPU
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test varlen_hd256_cuda
//!
//! The entire file is gated on the `cuda` feature so it compiles away entirely
//! on CPU-only builds.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

// ---------------------------------------------------------------------------
// CUDA serialization lock — same pattern as varlen_gqa_cuda.rs
// ---------------------------------------------------------------------------
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let threshold = tol + tol * y.abs();
        assert!(
            diff <= threshold,
            "{label} at index {i}: {x} vs {y} (diff={diff}, tol={threshold})"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared input builder
// ---------------------------------------------------------------------------

/// Build a packed 2-sequence batch with head_dim=256.
/// Returns (q_data, k_data, v_data, cu_seqlens, total_tokens, max_seqlen).
fn build_hd256_inputs(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>, usize, usize, usize) {
    // Sequence lengths: [4, 6] → total=10, max=6
    let seq_lens: Vec<usize> = vec![4, 6];
    let total_tokens: usize = seq_lens.iter().sum();
    let max_seqlen: usize = *seq_lens.iter().max().unwrap();
    let batch_size: usize = seq_lens.len();

    let cu_seqlens: Vec<i32> = {
        let mut v = vec![0i32];
        let mut acc = 0i32;
        for &l in &seq_lens {
            acc += l as i32;
            v.push(acc);
        }
        v
    };

    let n_q = total_tokens * num_heads * head_dim;
    let n_kv = total_tokens * num_kv_heads * head_dim;

    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.11).sin() * 0.3).collect();
    let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let v_data: Vec<f32> = (0..n_kv)
        .map(|i| ((i as f32) * 0.19).sin() * 0.25)
        .collect();

    (
        q_data,
        k_data,
        v_data,
        cu_seqlens,
        total_tokens,
        max_seqlen,
        batch_size,
    )
}

// ---------------------------------------------------------------------------
// Test 1: CUDA head_dim=256 GQA matches CPU reference
// ---------------------------------------------------------------------------

/// Verify that CUDA varlen_attention_fwd with head_dim=256, num_heads=4,
/// num_kv_heads=2 (GQA) matches the CPU reference within 1e-2.
///
/// CPU is the parity reference; it supports arbitrary head_dim natively.
#[test]
fn test_cuda_varlen_hd256_matches_cpu() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test_cuda_varlen_hd256_matches_cpu");
        return;
    }

    let _lock = cuda_lock();

    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 256usize;

    let (q_data, k_data, v_data, cu_data, total_tokens, max_seqlen, batch_size) =
        build_hd256_inputs(num_heads, num_kv_heads, head_dim);

    // --- CPU reference ---
    let (cpu_client, cpu_device) = cpu_setup();

    let q_cpu = Tensor::<CpuRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cpu_device,
    );
    let k_cpu = Tensor::<CpuRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    );
    let v_cpu = Tensor::<CpuRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    );
    let cu_cpu = Tensor::<CpuRuntime>::from_slice(&cu_data, &[batch_size + 1], &cpu_device);

    let (out_cpu, _) = cpu_client
        .varlen_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &cu_cpu,
            &cu_cpu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .expect("CPU GQA hd256 fwd failed");

    let cpu_vec = out_cpu.to_vec::<f32>();

    // --- CUDA hd256 GQA ---
    let (cuda_client, cuda_device) = cuda_setup();

    let q_c = Tensor::<CudaRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    );
    let k_c = Tensor::<CudaRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    );
    let v_c = Tensor::<CudaRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    );
    let cu_q = Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
    let cu_k = Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device);

    let (out_cuda, _) = cuda_client
        .varlen_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            &cu_q,
            &cu_k,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .expect("CUDA GQA hd256 fwd failed");

    let cuda_vec = out_cuda.to_vec::<f32>();

    assert_close(
        &cuda_vec,
        &cpu_vec,
        "CUDA GQA hd256 vs CPU GQA hd256 (f32)",
        1e-2,
    );
}

// ---------------------------------------------------------------------------
// Test 2: CUDA head_dim=256 MHA (num_kv_heads==num_heads) matches CPU
// ---------------------------------------------------------------------------

/// Verify that CUDA varlen_attention_fwd with head_dim=256, num_heads=4,
/// num_kv_heads=4 (plain MHA) matches the CPU reference within 1e-2.
#[test]
fn test_cuda_varlen_hd256_mha() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test_cuda_varlen_hd256_mha");
        return;
    }

    let _lock = cuda_lock();

    let num_heads = 4usize;
    let num_kv_heads = 4usize; // MHA: no GQA
    let head_dim = 256usize;

    let (q_data, k_data, v_data, cu_data, total_tokens, max_seqlen, batch_size) =
        build_hd256_inputs(num_heads, num_kv_heads, head_dim);

    // --- CPU reference ---
    let (cpu_client, cpu_device) = cpu_setup();

    let q_cpu = Tensor::<CpuRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cpu_device,
    );
    let k_cpu = Tensor::<CpuRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    );
    let v_cpu = Tensor::<CpuRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    );
    let cu_cpu = Tensor::<CpuRuntime>::from_slice(&cu_data, &[batch_size + 1], &cpu_device);

    let (out_cpu, _) = cpu_client
        .varlen_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &cu_cpu,
            &cu_cpu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .expect("CPU MHA hd256 fwd failed");

    let cpu_vec = out_cpu.to_vec::<f32>();

    // --- CUDA hd256 MHA ---
    let (cuda_client, cuda_device) = cuda_setup();

    let q_c = Tensor::<CudaRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    );
    let k_c = Tensor::<CudaRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    );
    let v_c = Tensor::<CudaRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    );
    let cu_q = Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
    let cu_k = Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device);

    let (out_cuda, _) = cuda_client
        .varlen_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            &cu_q,
            &cu_k,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .expect("CUDA MHA hd256 fwd failed");

    let cuda_vec = out_cuda.to_vec::<f32>();

    assert_close(
        &cuda_vec,
        &cpu_vec,
        "CUDA MHA hd256 vs CPU MHA hd256 (f32)",
        1e-2,
    );
}
