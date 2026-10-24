//! GPU parity tests for GQA varlen attention.
//!
//! Validates that the CUDA GQA kernel produces the same output as:
//!   1. CUDA MHA with K/V expanded (GQA correctness)
//!   2. CPU GQA (CPU vs CUDA parity)
//!   3. Both of the above at BF16, against a CPU reference on the BF16-rounded
//!      inputs (needs the `f16` feature)
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test varlen_gqa_cuda
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
// CUDA serialization lock — same pattern as encoder_cuda_graph.rs
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

/// Repeats each KV head `gqa_ratio` times: `[tokens, nkv, hd]` to
/// `[tokens, nkv * gqa_ratio, hd]`, Q head `kv_h * gqa_ratio + rep` reading KV
/// head `kv_h`.
fn expand_kv(
    data: &[f32],
    total_tokens: usize,
    num_kv_heads: usize,
    gqa_ratio: usize,
    head_dim: usize,
) -> Vec<f32> {
    let num_heads = num_kv_heads * gqa_ratio;
    let mut expanded = vec![0.0f32; total_tokens * num_heads * head_dim];
    for tok in 0..total_tokens {
        for kv_h in 0..num_kv_heads {
            for rep in 0..gqa_ratio {
                let q_h = kv_h * gqa_ratio + rep;
                let src_base = (tok * num_kv_heads + kv_h) * head_dim;
                let dst_base = (tok * num_heads + q_h) * head_dim;
                expanded[dst_base..dst_base + head_dim]
                    .copy_from_slice(&data[src_base..src_base + head_dim]);
            }
        }
    }
    expanded
}

// ---------------------------------------------------------------------------
// Test 1: CUDA GQA == CUDA MHA(expanded K/V)
// ---------------------------------------------------------------------------

/// Verify that the CUDA GQA forward kernel produces the same output as CUDA MHA
/// with K/V heads repeated (num_heads / num_kv_heads) times.
///
/// Configuration: num_heads=8, num_kv_heads=2, head_dim=64.
#[test]
fn test_cuda_gqa_equals_expanded_mha() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test_cuda_gqa_equals_expanded_mha");
        return;
    }

    let _lock = cuda_lock();

    let num_heads = 8usize;
    let num_kv_heads = 2usize;
    let gqa_ratio = num_heads / num_kv_heads; // 4
    let head_dim = 64usize;
    let total_tokens = 10usize;
    let batch_size = 2usize;
    let max_seqlen = 6usize;

    let n_q = total_tokens * num_heads * head_dim;
    let n_kv = total_tokens * num_kv_heads * head_dim;

    // Deterministic inputs (same pattern as CPU test)
    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
    let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let v_data: Vec<f32> = (0..n_kv)
        .map(|i| ((i as f32) * 0.17).sin() * 0.25)
        .collect();

    // Expand K and V to [total_tokens, num_heads, head_dim]
    let k_expanded = expand_kv(&k_data, total_tokens, num_kv_heads, gqa_ratio, head_dim);
    let v_expanded = expand_kv(&v_data, total_tokens, num_kv_heads, gqa_ratio, head_dim);

    let cu_data: Vec<i32> = vec![0, 4, 10];

    let (cuda_client, cuda_device) = cuda_setup();

    // Upload to CUDA
    let q_c = Tensor::<CudaRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let k_gqa_c = Tensor::<CudaRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let v_gqa_c = Tensor::<CudaRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let k_exp_c = Tensor::<CudaRuntime>::from_slice(
        &k_expanded,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let v_exp_c = Tensor::<CudaRuntime>::from_slice(
        &v_expanded,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let cu_q =
        Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
    let cu_k =
        Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();

    // Reference: MHA with expanded K/V
    let (out_mha, _) = cuda_client
        .varlen_attention_fwd(
            &q_c, &k_exp_c, &v_exp_c, &cu_q, &cu_k, batch_size, num_heads, num_heads, max_seqlen,
            max_seqlen, head_dim, false,
        )
        .expect("CUDA MHA fwd failed");

    // Under test: GQA with packed K/V
    let (out_gqa, _) = cuda_client
        .varlen_attention_fwd(
            &q_c,
            &k_gqa_c,
            &v_gqa_c,
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
        .expect("CUDA GQA fwd failed");

    let mha_vec = out_mha.to_vec::<f32>();
    let gqa_vec = out_gqa.to_vec::<f32>();

    assert_close(&gqa_vec, &mha_vec, "CUDA GQA vs CUDA expanded-MHA", 1e-2);
}

// ---------------------------------------------------------------------------
// Test 2: CUDA GQA == CPU GQA (cross-backend parity)
// ---------------------------------------------------------------------------

/// Verify that the CUDA GQA forward kernel matches the CPU GQA reference
/// implementation within numerical tolerance.
///
/// Configuration: num_heads=8, num_kv_heads=2, head_dim=64.
#[test]
fn test_cuda_gqa_vs_cpu_gqa() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test_cuda_gqa_vs_cpu_gqa");
        return;
    }

    let _lock = cuda_lock();

    let num_heads = 8usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let total_tokens = 10usize;
    let batch_size = 2usize;
    let max_seqlen = 6usize;

    let n_q = total_tokens * num_heads * head_dim;
    let n_kv = total_tokens * num_kv_heads * head_dim;

    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
    let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let v_data: Vec<f32> = (0..n_kv)
        .map(|i| ((i as f32) * 0.17).sin() * 0.25)
        .collect();

    let cu_data: Vec<i32> = vec![0, 4, 10];

    // --- CPU reference ---
    let (cpu_client, cpu_device) = cpu_setup();

    let q_cpu = Tensor::<CpuRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cpu_device,
    )
    .unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    )
    .unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cpu_device,
    )
    .unwrap();
    let cu_cpu =
        Tensor::<CpuRuntime>::from_slice(&cu_data, &[batch_size + 1], &cpu_device).unwrap();

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
        .expect("CPU GQA fwd failed");

    let cpu_vec = out_cpu.to_vec::<f32>();

    // --- CUDA GQA ---
    let (cuda_client, cuda_device) = cuda_setup();

    let q_c = Tensor::<CudaRuntime>::from_slice(
        &q_data,
        &[total_tokens, num_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let k_c = Tensor::<CudaRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let v_c = Tensor::<CudaRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_device,
    )
    .unwrap();
    let cu_q =
        Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
    let cu_k =
        Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();

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
        .expect("CUDA GQA fwd failed");

    let cuda_vec = out_cuda.to_vec::<f32>();

    assert_close(&cuda_vec, &cpu_vec, "CUDA GQA vs CPU GQA (f32)", 1e-2);
}

// ---------------------------------------------------------------------------
// Test 3: BF16 GQA == BF16 expanded MHA == CPU on the BF16-rounded inputs
// ---------------------------------------------------------------------------

/// BF16 forward, num_heads=8 over num_kv_heads=2, head_dim=64, causal.
///
/// GQA and expanded MHA read the same BF16 K/V values per Q head and run the
/// same per-thread arithmetic, so their outputs agree bit for bit.
///
/// The CPU reference runs in F32 on the rounded Q, K and V. The kernel
/// computes in F32 and rounds O once, so with `u = 2^-8` (BF16 unit roundoff)
/// the gap is at most `2u * |ref| + (u / 2) * ref_rms`. The factor 2 and the
/// RMS term cover F32 order and `__expf` noise across one rounding boundary.
#[cfg(feature = "f16")]
#[test]
fn test_cuda_gqa_bf16_vs_expanded_mha_and_cpu() {
    use numr::dtype::DType;

    if !cuda_available() {
        eprintln!("CUDA not available, skipping test_cuda_gqa_bf16_vs_expanded_mha_and_cpu");
        return;
    }
    let _lock = cuda_lock();

    let (num_heads, num_kv_heads, head_dim) = (8usize, 2usize, 64usize);
    let (total_tokens, batch_size, max_seqlen) = (10usize, 2usize, 6usize);
    let gqa_ratio = num_heads / num_kv_heads;
    let n_q = total_tokens * num_heads * head_dim;
    let n_kv = total_tokens * num_kv_heads * head_dim;
    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
    let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let v_data: Vec<f32> = (0..n_kv)
        .map(|i| ((i as f32) * 0.17).sin() * 0.25)
        .collect();
    let cu_data: Vec<i32> = vec![0, 4, 10];

    let (cuda_client, cuda_device) = cuda_setup();
    let bf16 = |data: &[f32], heads: usize| {
        Tensor::<CudaRuntime>::from_slice(data, &[total_tokens, heads, head_dim], &cuda_device)
            .expect("upload fixture")
            .to_dtype(DType::BF16)
            .expect("cast fixture to BF16")
    };
    let read =
        |t: &Tensor<CudaRuntime>| t.to_dtype(DType::F32).expect("cast to F32").to_vec::<f32>();
    let q_c = bf16(&q_data, num_heads);
    let k_c = bf16(&k_data, num_kv_heads);
    let v_c = bf16(&v_data, num_kv_heads);
    let (k_round, v_round) = (read(&k_c), read(&v_c));
    let k_exp_c = bf16(
        &expand_kv(&k_round, total_tokens, num_kv_heads, gqa_ratio, head_dim),
        num_heads,
    );
    let v_exp_c = bf16(
        &expand_kv(&v_round, total_tokens, num_kv_heads, gqa_ratio, head_dim),
        num_heads,
    );
    let cu = Tensor::<CudaRuntime>::from_slice(&cu_data, &[batch_size + 1], &cuda_device)
        .expect("upload cu_seqlens");

    let fwd = |k: &Tensor<CudaRuntime>, v: &Tensor<CudaRuntime>, nkv: usize| {
        cuda_client
            .varlen_attention_fwd(
                &q_c, k, v, &cu, &cu, batch_size, num_heads, nkv, max_seqlen, max_seqlen, head_dim,
                true,
            )
            .expect("CUDA BF16 varlen forward")
            .0
    };
    let out_gqa = fwd(&k_c, &v_c, num_kv_heads);
    let out_mha = fwd(&k_exp_c, &v_exp_c, num_heads);
    assert_eq!(out_gqa.dtype(), DType::BF16, "BF16 GQA output dtype");
    let gqa_vec = read(&out_gqa);
    assert_eq!(gqa_vec, read(&out_mha), "BF16 GQA vs BF16 expanded MHA");

    let (cpu_client, cpu_device) = cpu_setup();
    let cpu = |data: &[f32], heads: usize| {
        Tensor::<CpuRuntime>::from_slice(data, &[total_tokens, heads, head_dim], &cpu_device)
            .expect("build CPU tensor")
    };
    let cu_cpu = Tensor::<CpuRuntime>::from_slice(&cu_data, &[batch_size + 1], &cpu_device)
        .expect("build CPU cu_seqlens");
    let (out_cpu, _) = cpu_client
        .varlen_attention_fwd(
            &cpu(&read(&q_c), num_heads),
            &cpu(&k_round, num_kv_heads),
            &cpu(&v_round, num_kv_heads),
            &cu_cpu,
            &cu_cpu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            true,
        )
        .expect("CPU F32 varlen forward on BF16-rounded inputs");
    let cpu_vec = out_cpu.to_vec::<f32>();

    let u = 1.0f32 / 256.0;
    let rms = (cpu_vec.iter().map(|x| x * x).sum::<f32>() / cpu_vec.len() as f32).sqrt();
    assert!(
        rms > 1e-4,
        "reference RMS {rms:.3e} is degenerate; fix the fixture"
    );
    for (i, (&a, &e)) in gqa_vec.iter().zip(&cpu_vec).enumerate() {
        let bound = 2.0 * u * e.abs() + 0.5 * u * rms;
        assert!(
            (a - e).abs() <= bound,
            "BF16 GQA vs CPU at index {i}: {a} vs {e} (bound {bound:.3e})"
        );
    }
}
