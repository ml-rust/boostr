//! CUDA backward-pass parity tests for GQA varlen attention.
//!
//! Validates:
//!   1. CUDA bwd GQA (hd=64) matches CPU bwd GQA (dq/dk/dv).
//!   2. CUDA bwd hd=256 GQA matches CPU bwd hd=256 GQA.
//!   3. FP16 CUDA bwd GQA (hd=64) does not corrupt memory (atomicAddHalf fix).
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test varlen_bwd_cuda
//!
//! The entire file is gated on the `cuda` feature.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

// ---------------------------------------------------------------------------
// CUDA serialization lock
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
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max diff {max_diff:.2e} at index {max_idx} exceeds tol {tol:.2e}"
    );
}

// ---------------------------------------------------------------------------
// Shared test data builder
// ---------------------------------------------------------------------------

struct BwdTestCase {
    total_tokens: usize,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seqlen: usize,
    cu_seqlens: Vec<i32>,
    q_data: Vec<f32>,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    do_data: Vec<f32>,
}

impl BwdTestCase {
    fn new(
        total_tokens: usize,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seqlen: usize,
        cu_seqlens: Vec<i32>,
    ) -> Self {
        let n_q = total_tokens * num_heads * head_dim;
        let n_kv = total_tokens * num_kv_heads * head_dim;
        let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
        let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
        let v_data: Vec<f32> = (0..n_kv)
            .map(|i| ((i as f32) * 0.17).sin() * 0.25)
            .collect();
        let do_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.11).cos() * 0.1).collect();
        Self {
            total_tokens,
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seqlen,
            cu_seqlens,
            q_data,
            k_data,
            v_data,
            do_data,
        }
    }

    /// Run fwd + bwd on CPU (f32 reference).
    fn run_cpu(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (client, dev) = cpu_setup();
        let q = Tensor::<CpuRuntime>::from_slice(
            &self.q_data,
            &[self.total_tokens, self.num_heads, self.head_dim],
            &dev,
        );
        let k = Tensor::<CpuRuntime>::from_slice(
            &self.k_data,
            &[self.total_tokens, self.num_kv_heads, self.head_dim],
            &dev,
        );
        let v = Tensor::<CpuRuntime>::from_slice(
            &self.v_data,
            &[self.total_tokens, self.num_kv_heads, self.head_dim],
            &dev,
        );
        let dout = Tensor::<CpuRuntime>::from_slice(
            &self.do_data,
            &[self.total_tokens, self.num_heads, self.head_dim],
            &dev,
        );
        let cu = Tensor::<CpuRuntime>::from_slice(&self.cu_seqlens, &[self.batch_size + 1], &dev);

        let (out, lse) = client
            .varlen_attention_fwd(
                &q,
                &k,
                &v,
                &cu,
                &cu,
                self.batch_size,
                self.num_heads,
                self.num_kv_heads,
                self.max_seqlen,
                self.max_seqlen,
                self.head_dim,
                false,
            )
            .unwrap();

        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &dout,
                &q,
                &k,
                &v,
                &out,
                &lse,
                &cu,
                &cu,
                self.batch_size,
                self.num_heads,
                self.num_kv_heads,
                self.max_seqlen,
                self.max_seqlen,
                self.head_dim,
                false,
            )
            .unwrap();

        (dq.to_vec::<f32>(), dk.to_vec::<f32>(), dv.to_vec::<f32>())
    }

    /// Run fwd + bwd on CUDA (f32).
    fn run_cuda_f32(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (client, cuda_dev) = cuda_setup();
        let (_, cpu_dev) = cpu_setup();

        let q = Tensor::<CudaRuntime>::from_slice(
            &self.q_data,
            &[self.total_tokens, self.num_heads, self.head_dim],
            &cuda_dev,
        );
        let k = Tensor::<CudaRuntime>::from_slice(
            &self.k_data,
            &[self.total_tokens, self.num_kv_heads, self.head_dim],
            &cuda_dev,
        );
        let v = Tensor::<CudaRuntime>::from_slice(
            &self.v_data,
            &[self.total_tokens, self.num_kv_heads, self.head_dim],
            &cuda_dev,
        );
        let dout = Tensor::<CudaRuntime>::from_slice(
            &self.do_data,
            &[self.total_tokens, self.num_heads, self.head_dim],
            &cuda_dev,
        );
        let cu =
            Tensor::<CudaRuntime>::from_slice(&self.cu_seqlens, &[self.batch_size + 1], &cuda_dev);

        let (out, lse) = client
            .varlen_attention_fwd(
                &q,
                &k,
                &v,
                &cu,
                &cu,
                self.batch_size,
                self.num_heads,
                self.num_kv_heads,
                self.max_seqlen,
                self.max_seqlen,
                self.head_dim,
                false,
            )
            .unwrap();

        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &dout,
                &q,
                &k,
                &v,
                &out,
                &lse,
                &cu,
                &cu,
                self.batch_size,
                self.num_heads,
                self.num_kv_heads,
                self.max_seqlen,
                self.max_seqlen,
                self.head_dim,
                false,
            )
            .unwrap();

        let _ = cpu_dev; // suppress unused
        (dq.to_vec::<f32>(), dk.to_vec::<f32>(), dv.to_vec::<f32>())
    }
}

// ---------------------------------------------------------------------------
// Test 1: CUDA bwd GQA hd=64 matches CPU bwd
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_varlen_bwd_matches_cpu_gqa() {
    if !cuda_available() {
        eprintln!("test_cuda_varlen_bwd_matches_cpu_gqa: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();

    // Two sequences: lengths 3 and 5
    let tc = BwdTestCase::new(
        /*total_tokens=*/ 8,
        /*batch_size=*/ 2,
        /*num_heads=*/ 8,
        /*num_kv_heads=*/ 2,
        /*head_dim=*/ 64,
        /*max_seqlen=*/ 5,
        vec![0i32, 3, 8],
    );

    let (dq_cpu, dk_cpu, dv_cpu) = tc.run_cpu();
    let (dq_cuda, dk_cuda, dv_cuda) = tc.run_cuda_f32();

    assert_close(&dq_cpu, &dq_cuda, "dQ (hd64 GQA)", 1e-2);
    assert_close(&dk_cpu, &dk_cuda, "dK (hd64 GQA)", 1e-2);
    assert_close(&dv_cpu, &dv_cuda, "dV (hd64 GQA)", 1e-2);
}

// ---------------------------------------------------------------------------
// Test 2: CUDA bwd hd=256 GQA matches CPU bwd
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_varlen_bwd_hd256() {
    if !cuda_available() {
        eprintln!("test_cuda_varlen_bwd_hd256: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();

    // Two sequences: lengths 4 and 6
    let tc = BwdTestCase::new(
        /*total_tokens=*/ 10,
        /*batch_size=*/ 2,
        /*num_heads=*/ 4,
        /*num_kv_heads=*/ 2,
        /*head_dim=*/ 256,
        /*max_seqlen=*/ 6,
        vec![0i32, 4, 10],
    );

    let (dq_cpu, dk_cpu, dv_cpu) = tc.run_cpu();
    let (dq_cuda, dk_cuda, dv_cuda) = tc.run_cuda_f32();

    assert_close(&dq_cpu, &dq_cuda, "dQ (hd256 GQA)", 1e-2);
    assert_close(&dk_cpu, &dk_cuda, "dK (hd256 GQA)", 1e-2);
    assert_close(&dv_cpu, &dv_cuda, "dV (hd256 GQA)", 1e-2);
}

// ---------------------------------------------------------------------------
// Test 3: FP16 CUDA bwd GQA hd=64 — proves atomicAddHalf works (no corruption)
// ---------------------------------------------------------------------------

#[cfg(feature = "f16")]
#[test]
fn test_cuda_varlen_bwd_fp16_gqa_no_corruption() {
    if !cuda_available() {
        eprintln!("test_cuda_varlen_bwd_fp16_gqa_no_corruption: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();

    let total_tokens = 8usize;
    let batch_size = 2usize;
    let num_heads = 8usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let max_seqlen = 5usize;
    let cu_seqlens = vec![0i32, 3, 8];

    let n_q = total_tokens * num_heads * head_dim;
    let n_kv = total_tokens * num_kv_heads * head_dim;

    let q_f32: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
    let k_f32: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let v_f32: Vec<f32> = (0..n_kv)
        .map(|i| ((i as f32) * 0.17).sin() * 0.25)
        .collect();
    let do_f32: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.11).cos() * 0.1).collect();

    // Convert to f16
    let q_f16: Vec<half::f16> = q_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_f16: Vec<half::f16> = k_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_f16: Vec<half::f16> = v_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let do_f16: Vec<half::f16> = do_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

    let (client, cuda_dev) = cuda_setup();

    let q =
        Tensor::<CudaRuntime>::from_slice(&q_f16, &[total_tokens, num_heads, head_dim], &cuda_dev);
    let k = Tensor::<CudaRuntime>::from_slice(
        &k_f16,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_dev,
    );
    let v = Tensor::<CudaRuntime>::from_slice(
        &v_f16,
        &[total_tokens, num_kv_heads, head_dim],
        &cuda_dev,
    );
    let dout =
        Tensor::<CudaRuntime>::from_slice(&do_f16, &[total_tokens, num_heads, head_dim], &cuda_dev);
    let cu = Tensor::<CudaRuntime>::from_slice(&cu_seqlens, &[batch_size + 1], &cuda_dev);

    let (out, lse) = client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu,
            &cu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();

    let (dq, dk, dv) = client
        .varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu,
            &cu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();

    // Verify shapes — proves no silent panic
    assert_eq!(dq.shape(), &[total_tokens, num_heads, head_dim]);
    assert_eq!(dk.shape(), &[total_tokens, num_kv_heads, head_dim]);
    assert_eq!(dv.shape(), &[total_tokens, num_kv_heads, head_dim]);

    // Convert back to f32 for numerical check vs CPU reference
    let dq_f32: Vec<f32> = dq
        .to_vec::<half::f16>()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let dk_f32: Vec<f32> = dk
        .to_vec::<half::f16>()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let dv_f32: Vec<f32> = dv
        .to_vec::<half::f16>()
        .iter()
        .map(|x| x.to_f32())
        .collect();

    // CPU reference (f32)
    let tc = BwdTestCase::new(
        total_tokens,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seqlen,
        cu_seqlens,
    );
    let (dq_cpu, dk_cpu, dv_cpu) = tc.run_cpu();

    // Looser tolerance for fp16 (quantisation + atomicAddHalf precision)
    assert_close(&dq_f32, &dq_cpu, "dQ fp16 GQA vs CPU ref", 2e-2);
    assert_close(&dk_f32, &dk_cpu, "dK fp16 GQA vs CPU ref", 2e-2);
    assert_close(&dv_f32, &dv_cpu, "dV fp16 GQA vs CPU ref", 2e-2);

    // Sanity: none of the values should be NaN or Inf (corruption would cause these)
    for &v in dq_f32.iter().chain(dk_f32.iter()).chain(dv_f32.iter()) {
        assert!(v.is_finite(), "fp16 bwd produced non-finite value: {v}");
    }
}
