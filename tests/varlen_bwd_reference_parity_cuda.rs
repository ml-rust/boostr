//! CUDA varlen attention backward vs CPU reference parity, across the axes
//! `tests/varlen_bwd_cuda.rs` and `tests/varlen_tile_fallback_cuda.rs` leave
//! uncovered.
//!
//! The dK/dV accumulation in `varlen_attention_bwd.cu` /
//! `varlen_attention_bwd_fp16.cu` was just restructured from an `atomicAdd`
//! per `(q_row, k_idx, d)` straight into output storage, to FP32 register
//! accumulation per K-row within a tile with ONE atomic per `(k_row, d)` per
//! Q-block. The identical defect class in the paged-attention backward
//! kernel (see `tests/backend_parity/paged_attention.rs`) produced up to 83%
//! relative error on dK/dV while every prior test still passed — that was
//! caught only by comparing against a CPU reference at half precision. This
//! file does the same for varlen.
//!
//! Reference: `CpuClient::varlen_attention_bwd` (the `impl_generic`-style CPU
//! fallback in `src/ops/cpu/attention/varlen_attention.rs`), the same
//! reference `tests/varlen_bwd_cuda.rs` already uses via `BwdTestCase::run_cpu`.
//! It implements the identical bottom-right causal rule
//! (`key_offset = seq_len_k.saturating_sub(seq_len_q)`, masking
//! `key_offset + qi < ki`) as the CUDA kernel, independently, in plain Rust —
//! not sharing any gather/scatter/indexing code with the CUDA path.
//!
//! `head_dim=128` and `head_dim=256` backward are exercised here for the
//! first time (only `head_dim in {64,256}` in `varlen_bwd_cuda.rs`, and
//! `varlen_tile_fallback_cuda.rs` only proves the large/small CUDA tiles
//! agree with EACH OTHER, not with an independent reference). Every case here
//! also uses a RAGGED batch (unequal per-sequence lengths) with
//! `seq_len_q != seq_len_k`, so the `key_offset` bottom-right rule and the
//! `cu_seqlens_q`/`cu_seqlens_k` indexing are actually exercised — a uniform
//! batch would not touch either.
//!
//! varlen supports ONLY F32/F16 (`src/ops/cuda/attention/varlen_attention_bwd.rs`
//! rejects any other dtype), so no BF16 case is added here.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test varlen_bwd_reference_parity_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

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

/// Deterministic pseudo-random values, distinct per index and per seed —
/// same generator shape as `varlen_tile_fallback_cuda.rs::values`.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Casts an `F32` fixture (built via `Tensor::from_slice`, since `half::f16`
/// is not a numr `Element`) to the dtype under test.
fn cast_to_dtype(
    data: &[f32],
    shape: &[usize],
    device: &CudaDevice,
    dtype: DType,
) -> Tensor<CudaRuntime> {
    let t = Tensor::<CudaRuntime>::from_slice(data, shape, device).unwrap();
    if dtype == DType::F32 {
        t
    } else {
        t.to_dtype(dtype)
            .unwrap_or_else(|e| panic!("cast fixture to {dtype:?} failed: {e}"))
    }
}

/// Reads a CUDA result tensor back to `Vec<f32>`, casting through `F32` first
/// when it is stored as `F16`.
fn read_back_f32(t: &Tensor<CudaRuntime>) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        t.to_vec::<f32>()
    } else {
        t.to_dtype(DType::F32)
            .expect("cast kernel result back to F32 for comparison")
            .to_vec::<f32>()
    }
}

/// Backward tolerance, derived from first principles — NOT tuned to make any
/// particular case pass. Mirrors `paged_attention.rs::paged_bwd_tol`, which
/// this file's kernel restructure is the varlen counterpart of.
///
/// Base pair is the quantization-only backward error from rounding Q/K/V/dO
/// to `dtype` before the kernel runs (`f16`: atol 6e-3, rtol 3e-2 — reused
/// from `mqa_gqa_attention.rs`'s own backward-pass measurement of the same
/// mechanism; `f32` keeps 1e-5/1e-4).
///
/// `dQ` accumulates in FP32 registers for the whole kernel and is cast to
/// `dtype` exactly once, at the very end (`varlen_attention_bwd.cu`'s
/// `dQ_local[HEAD_DIM]` then a single final store) — pass `n_contrib = None`,
/// the base pair is its whole tolerance.
///
/// `dK`/`dV` accumulate every Q row of a tile into an FP32 register per
/// K-row, then issue ONE `atomicAdd` into `dtype` storage per `(k_row, d)`
/// per Q-block — not once per `(q_row, k_idx, d)` as before the restructure.
/// Each such add still rounds the running sum to `dtype`'s mantissa, so for
/// `n` sequential rounded additions Higham's classical recursive-summation
/// bound applies: extra relative error `(n-1) * u` (`u` = unit roundoff:
/// `2^-24` f32, `2^-11` f16). `n_contrib` is passed as
/// `max_seqlen_q * (num_heads / num_kv_heads)` — an upper bound on
/// `num_q_blocks_per_batch * heads_per_kv`, the actual number of atomics a
/// given `(k_row, d)` receives, since `num_q_blocks_per_batch <=
/// max_seqlen_q` for any `BLOCK_M >= 1`. Deliberately loose (mirrors
/// `paged_bwd_tol`'s own reasoning for using `seq_len * heads_per_kv` rather
/// than the strict per-tile count): tightening it to the exact runtime
/// `BLOCK_M` would make this test flake with the device-dependent large/small
/// tile choice in `varlen_attention_block_config.rs` rather than catch a real
/// regression.
fn varlen_bwd_tol(dtype: DType, n_contrib: Option<usize>) -> (f32, f32) {
    let (atol, rtol_base) = match dtype {
        DType::F32 => (1e-5, 1e-4),
        DType::F16 => (6e-3, 3e-2),
        other => {
            unimplemented!("varlen_bwd_tol: unsupported dtype {other:?} (varlen is F32/F16 only)")
        }
    };
    let u: f32 = match dtype {
        DType::F32 => 2f32.powi(-24),
        DType::F16 => 2f32.powi(-11),
        other => {
            unimplemented!("varlen_bwd_tol: unsupported dtype {other:?} (varlen is F32/F16 only)")
        }
    };
    let rtol = match n_contrib {
        None => rtol_base,
        Some(n) => rtol_base + (n.saturating_sub(1) as f32) * u,
    };
    (atol, rtol)
}

/// Compares against the reference, printing an always-on `VARLEN_BWD_DIAG`
/// line (pass or fail) so the measured deviation is visible even on a green
/// run. Returns the normalized error (`max_abs_diff / ref_rms`) for the
/// dK/dV-to-dQ ratio guard.
#[allow(clippy::too_many_arguments)]
fn assert_varlen_bwd_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    tensor: &str,
    dtype: DType,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> f32 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: element count mismatch: kernel {} vs reference {}",
        actual.len(),
        expected.len()
    );

    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    let mut max_rel = 0.0f32;
    let mut max_rel_idx = 0usize;
    let mut sq_sum = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: kernel produced non-finite value {a} at index {i} (reference {e})"
        );
        let diff = (a - e).abs();
        if diff > max_abs {
            max_abs = diff;
            max_abs_idx = i;
        }
        let rel = diff / (e.abs() + 1e-12);
        if rel > max_rel {
            max_rel = rel;
            max_rel_idx = i;
        }
        sq_sum += (*e as f64) * (*e as f64);
    }
    let rms = (sq_sum / expected.len() as f64).sqrt() as f32;
    let tol = atol + rtol * rms;

    println!(
        "VARLEN_BWD_DIAG tensor={tensor} dtype={:?} num_heads={num_heads} \
         num_kv_heads={num_kv_heads} head_dim={head_dim} causal={causal} \
         max_abs={max_abs:.6e} max_abs_idx={max_abs_idx} max_rel={max_rel:.6e} \
         max_rel_idx={max_rel_idx} ref_rms={rms:.6e} atol={atol:.6e} rtol={rtol:.6e} \
         tol={tol:.6e} label=\"{label}\"",
        dtype
    );

    assert!(
        rms > 1e-6,
        "{label}: reference RMS is {rms:.4e} — the fixture is degenerate, so agreement \
         would prove nothing. Fix the fixture, not the tolerance."
    );
    assert!(
        max_abs <= tol,
        "{label}: max_abs_diff {max_abs:.4e} at index {max_abs_idx} (max_rel_diff \
         {max_rel:.4e} at index {max_rel_idx}) exceeds tol {tol:.4e} (ref_rms {rms:.4e}); \
         kernel={} reference={}",
        actual[max_abs_idx],
        expected[max_abs_idx]
    );

    max_abs / rms
}

/// Ragged batch fixture reused by every case in this file: 3 sequences of
/// UNEQUAL length, each with `seq_len_q != seq_len_k`, so `key_offset =
/// max(0, seq_len_k - seq_len_q)` takes three different values:
///   seq 0: seq_len_q=5, seq_len_k=9  -> key_offset=4 (KV cache longer than Q, the decode-with-prefix case)
///   seq 1: seq_len_q=7, seq_len_k=4  -> key_offset=0 (Q longer than K; row 0 still has exactly key 0 valid)
///   seq 2: seq_len_q=3, seq_len_k=3  -> key_offset=0 (the equal-length case, for contrast)
/// `max_seqlen_q=7` keeps every case's `num_q_blocks_per_batch == 1` for
/// every compiled `BLOCK_M` (16/32/128), which `varlen_bwd_tol`'s `n_contrib`
/// bound relies on staying small.
/// Returns `(batch_size, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
/// max_seqlen_k, total_tokens_q, total_tokens_k)`.
#[allow(clippy::type_complexity)]
fn ragged_fixture() -> (usize, Vec<i32>, Vec<i32>, usize, usize, usize, usize) {
    (3, vec![0, 5, 12, 15], vec![0, 9, 13, 16], 7, 9, 15, 16)
}

/// Core comparison: builds the ragged fixture, runs the CPU reference in
/// `F32`, runs the CUDA kernel at `dtype` (casting inputs down and results
/// back up through `F32`), and checks dQ/dK/dV against the reference with the
/// dK/dV-to-dQ ratio guard.
#[allow(clippy::too_many_arguments)]
fn assert_varlen_bwd_parity(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    dtype: DType,
    label: &str,
) {
    if dtype != DType::F32 && !cfg!(feature = "f16") {
        eprintln!(
            "SKIPPED: {label} [{:?}] — boostr built without the `f16` feature, so {:?} \
             tensors cannot be constructed",
            dtype, dtype
        );
        return;
    }
    if !cuda_available() {
        eprintln!("SKIPPED: {label} — CUDA not available");
        return;
    }
    let _lock = cuda_lock();

    let (
        batch_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        total_tokens_q,
        total_tokens_k,
    ) = ragged_fixture();
    let n_q = total_tokens_q * num_heads * head_dim;
    let n_kv = total_tokens_k * num_kv_heads * head_dim;

    let q_data = values(n_q, 0.1);
    let k_data = values(n_kv, 1.3);
    let v_data = values(n_kv, 2.7);
    let do_data = values(n_q, 3.9);

    // CPU reference, always F32.
    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu =
        Tensor::<CpuRuntime>::from_slice(&q_data, &[total_tokens_q, num_heads, head_dim], &cpu_dev)
            .unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(
        &k_data,
        &[total_tokens_k, num_kv_heads, head_dim],
        &cpu_dev,
    )
    .unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(
        &v_data,
        &[total_tokens_k, num_kv_heads, head_dim],
        &cpu_dev,
    )
    .unwrap();
    let do_cpu = Tensor::<CpuRuntime>::from_slice(
        &do_data,
        &[total_tokens_q, num_heads, head_dim],
        &cpu_dev,
    )
    .unwrap();
    let cu_q_cpu =
        Tensor::<CpuRuntime>::from_slice(&cu_seqlens_q, &[batch_size + 1], &cpu_dev).unwrap();
    let cu_k_cpu =
        Tensor::<CpuRuntime>::from_slice(&cu_seqlens_k, &[batch_size + 1], &cpu_dev).unwrap();

    let (out_cpu, lse_cpu) = cpu_client
        .varlen_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &cu_q_cpu,
            &cu_k_cpu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
        .unwrap();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_client
        .varlen_attention_bwd(
            &do_cpu,
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &out_cpu,
            &lse_cpu,
            &cu_q_cpu,
            &cu_k_cpu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
        .unwrap();
    let dq_cpu_vec = dq_cpu.to_vec::<f32>();
    let dk_cpu_vec = dk_cpu.to_vec::<f32>();
    let dv_cpu_vec = dv_cpu.to_vec::<f32>();

    // CUDA at `dtype`.
    let (cuda_client, cuda_dev) = cuda_setup();
    let q_c = cast_to_dtype(
        &q_data,
        &[total_tokens_q, num_heads, head_dim],
        &cuda_dev,
        dtype,
    );
    let k_c = cast_to_dtype(
        &k_data,
        &[total_tokens_k, num_kv_heads, head_dim],
        &cuda_dev,
        dtype,
    );
    let v_c = cast_to_dtype(
        &v_data,
        &[total_tokens_k, num_kv_heads, head_dim],
        &cuda_dev,
        dtype,
    );
    let do_c = cast_to_dtype(
        &do_data,
        &[total_tokens_q, num_heads, head_dim],
        &cuda_dev,
        dtype,
    );
    let cu_q_c =
        Tensor::<CudaRuntime>::from_slice(&cu_seqlens_q, &[batch_size + 1], &cuda_dev).unwrap();
    let cu_k_c =
        Tensor::<CudaRuntime>::from_slice(&cu_seqlens_k, &[batch_size + 1], &cuda_dev).unwrap();

    let (out_c, lse_c) = cuda_client
        .varlen_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            &cu_q_c,
            &cu_k_c,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
        .unwrap();
    let (dq_c, dk_c, dv_c) = cuda_client
        .varlen_attention_bwd(
            &do_c,
            &q_c,
            &k_c,
            &v_c,
            &out_c,
            &lse_c,
            &cu_q_c,
            &cu_k_c,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
        .unwrap();

    let heads_per_kv = num_heads / num_kv_heads;
    let n_contrib = max_seqlen_q * heads_per_kv;
    let (dq_atol, dq_rtol) = varlen_bwd_tol(dtype, None);
    let (dkv_atol, dkv_rtol) = varlen_bwd_tol(dtype, Some(n_contrib));

    let dq_norm = assert_varlen_bwd_diff(
        &read_back_f32(&dq_c),
        &dq_cpu_vec,
        dq_atol,
        dq_rtol,
        &format!("{} dQ CUDA vs CPU [{:?}]", label, dtype),
        "dQ",
        dtype,
        num_heads,
        num_kv_heads,
        head_dim,
        causal,
    );
    let dk_norm = assert_varlen_bwd_diff(
        &read_back_f32(&dk_c),
        &dk_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{} dK CUDA vs CPU [{:?}]", label, dtype),
        "dK",
        dtype,
        num_heads,
        num_kv_heads,
        head_dim,
        causal,
    );
    let dv_norm = assert_varlen_bwd_diff(
        &read_back_f32(&dv_c),
        &dv_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{} dV CUDA vs CPU [{:?}]", label, dtype),
        "dV",
        dtype,
        num_heads,
        num_kv_heads,
        head_dim,
        causal,
    );

    // dQ is the FP32-accumulated control (untouched by the dK/dV
    // restructure), so its normalized error tracks pure input-quantization
    // error only. dK/dV normalized error should sit within a fixed multiple
    // of dQ's at the same shape and dtype — a regression back to
    // per-(q_row, k_idx, d) half-precision atomics is exactly what would
    // blow this ratio up, as it did for the paged kernel (~780x measured
    // pre-fix there, 2.8x-6.7x post-fix). 25x is reused unchanged from
    // `paged_attention.rs::DKV_TO_DQ_RATIO_LIMIT`: same defect class, same
    // one-atomic-per-Q-block-per-K-row restructure, so the same headroom
    // above real post-fix noise and margin below the pre-fix magnitude
    // applies without needing a varlen-specific recalibration.
    const DKV_TO_DQ_RATIO_LIMIT: f32 = 25.0;
    for (name, norm) in [("dK", dk_norm), ("dV", dv_norm)] {
        let ratio = norm / dq_norm;
        assert!(
            ratio <= DKV_TO_DQ_RATIO_LIMIT,
            "{label} {name} CUDA vs CPU [{dtype:?}]: normalized error (max_abs_diff / \
             ref_rms) is {norm:.4e}, {ratio:.1}x dQ's {dq_norm:.4e} (limit \
             {DKV_TO_DQ_RATIO_LIMIT}x) — this most likely means {name}'s atomicAdd \
             accumulation regressed from one atomic per (k_row, d) per Q-block back to \
             one atomic per (q_row, k_idx, d), rounding the running sum to `{dtype:?}` on \
             every contribution instead of once per tile"
        );
    }
}

// ============================================================================
// GQA (num_kv_heads < num_heads): 8 query heads over 2 KV heads, ratio 4.
// head_dim in {64, 128, 256} x dtype in {F32, F16} x causal in {true, false}.
// ============================================================================

#[test]
fn test_varlen_bwd_parity_gqa_hd64_f32_causal() {
    assert_varlen_bwd_parity(8, 2, 64, true, DType::F32, "varlen_bwd gqa hd64 f32 causal");
}

#[test]
fn test_varlen_bwd_parity_gqa_hd64_f32_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        64,
        false,
        DType::F32,
        "varlen_bwd gqa hd64 f32 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd64_f16_causal() {
    assert_varlen_bwd_parity(8, 2, 64, true, DType::F16, "varlen_bwd gqa hd64 f16 causal");
}

#[test]
fn test_varlen_bwd_parity_gqa_hd64_f16_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        64,
        false,
        DType::F16,
        "varlen_bwd gqa hd64 f16 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd128_f32_causal() {
    assert_varlen_bwd_parity(
        8,
        2,
        128,
        true,
        DType::F32,
        "varlen_bwd gqa hd128 f32 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd128_f32_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        128,
        false,
        DType::F32,
        "varlen_bwd gqa hd128 f32 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd128_f16_causal() {
    assert_varlen_bwd_parity(
        8,
        2,
        128,
        true,
        DType::F16,
        "varlen_bwd gqa hd128 f16 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd128_f16_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        128,
        false,
        DType::F16,
        "varlen_bwd gqa hd128 f16 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd256_f32_causal() {
    assert_varlen_bwd_parity(
        8,
        2,
        256,
        true,
        DType::F32,
        "varlen_bwd gqa hd256 f32 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd256_f32_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        256,
        false,
        DType::F32,
        "varlen_bwd gqa hd256 f32 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd256_f16_causal() {
    assert_varlen_bwd_parity(
        8,
        2,
        256,
        true,
        DType::F16,
        "varlen_bwd gqa hd256 f16 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_gqa_hd256_f16_noncausal() {
    assert_varlen_bwd_parity(
        8,
        2,
        256,
        false,
        DType::F16,
        "varlen_bwd gqa hd256 f16 noncausal",
    );
}

// ============================================================================
// No-GQA (num_kv_heads == num_heads): 4 query heads, 4 KV heads, ratio 1.
// head_dim=64 x dtype in {F32, F16} x causal in {true, false}.
// ============================================================================

#[test]
fn test_varlen_bwd_parity_no_gqa_hd64_f32_causal() {
    assert_varlen_bwd_parity(
        4,
        4,
        64,
        true,
        DType::F32,
        "varlen_bwd no-gqa hd64 f32 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_no_gqa_hd64_f32_noncausal() {
    assert_varlen_bwd_parity(
        4,
        4,
        64,
        false,
        DType::F32,
        "varlen_bwd no-gqa hd64 f32 noncausal",
    );
}

#[test]
fn test_varlen_bwd_parity_no_gqa_hd64_f16_causal() {
    assert_varlen_bwd_parity(
        4,
        4,
        64,
        true,
        DType::F16,
        "varlen_bwd no-gqa hd64 f16 causal",
    );
}

#[test]
fn test_varlen_bwd_parity_no_gqa_hd64_f16_noncausal() {
    assert_varlen_bwd_parity(
        4,
        4,
        64,
        false,
        DType::F16,
        "varlen_bwd no-gqa hd64 f16 noncausal",
    );
}
