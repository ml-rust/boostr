//! Flash Attention v3 backward (`kernels/attention/flash_v3_bwd.cu`) vs the
//! CPU reference, for dQ, dK and dV.
//!
//! # Why this file exists
//!
//! Before the restructure this file guards, every low-precision v3 backward
//! impl accumulated `dK_local`/`dV_local` over EVERY K row while `tid` indexed
//! a Q row, then wrote that sum out as the gradient of K position
//! `k_start + tid`. dK/dV were attributed to entirely the wrong K positions,
//! and the store was a plain `=`, so contributions from different Q blocks
//! overwrote each other instead of summing. The FP32 impl had a different
//! defect: `atomicAdd` per `(q_row, k_row, d)` into dK/dV buffers the launcher
//! allocates WITHOUT zeroing.
//!
//! Nothing caught it because NO test in the repo referenced flash_v3 at all,
//! and the kernel only runs on SM 90+. This file is the missing reference
//! check. It is the flash_v3 counterpart of
//! `tests/varlen_bwd_reference_parity_cuda.rs`, whose kernel carried the same
//! defect class, and it reuses that file's diagnostic line and dK/dV-to-dQ
//! error ratio guard.
//!
//! # LOUD SKIP
//!
//! flash_v3 is gated behind `flash_v3::is_hopper` (compute capability major
//! version 9 or above). On any other GPU these kernels CANNOT run, and every
//! test here returns early after printing a `FLASH_V3_BWD_SKIPPED` banner to
//! both stdout
//! and stderr. libtest captures both for passing tests, so on non-Hopper
//! hardware a green run here proves NOTHING — run with `--nocapture` and read
//! the banner. Every test name carries the `_hopper_only` suffix for the same
//! reason.
//!
//! # Reference
//!
//! `CpuClient::flash_attention_fwd` / `flash_attention_bwd`
//! (`src/ops/cpu/attention/flash.rs` -> `impl_generic`'s
//! `standard_attention_bwd`), always in F32. It shares no indexing,
//! accumulation, or tiling code with the CUDA kernel.
//!
//! # `seq_len_q == seq_len_k` is REQUIRED here
//!
//! flash_v3's causal rule is TOP-LEFT (`if (causal && q_global < k_global)`),
//! while the CPU reference is BOTTOM-RIGHT
//! (`key_offset = seq_len_k - seq_len_q`, masking `key_offset + i < j`; see
//! `impl_generic/attention/flash_standard.rs`). The two agree only when
//! `seq_len_q == seq_len_k`, which every case here uses. A ragged case would
//! measure that convention divergence, not the gradient defect this file
//! guards.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_v3_bwd_parity_cuda -- --nocapture

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::flash_v3;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

// CUDA tests in this crate serialize on a process-wide lock.
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

const BATCH: usize = 2;
const NUM_HEADS: usize = 4;

/// 96 is deliberate, not round. flash_v3 backward uses `BLOCK_N = 64` /
/// `BLOCK_M = 32` at `head_dim = 64` and `BLOCK_N = 32` / `BLOCK_M = 16` at
/// `head_dim = 128`, so 96 gives:
///   - head_dim 64:  2 K blocks (the second is a PARTIAL tile, `k_size = 32`)
///     and 3 Q blocks, so dK/dV must sum across Q blocks and the partial-tile
///     guard must hold.
///   - head_dim 128: 3 K blocks and 6 Q blocks.
///
/// A single-K-block, single-Q-block shape would hide both the cross-Q-block
/// overwrite and the wrong-K-position attribution this file exists to catch.
const SEQ: usize = 96;

/// Prints the skip banner to stdout AND stderr, then returns. Deliberately
/// noisy: a silent skip is how the original defect survived.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! FLASH_V3_BWD_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT running\n\
         !! the flash_v3 backward kernel. Treat it as UNTESTED, not as green.\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    );
    println!("{banner}");
    eprintln!("{banner}");
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

/// Deterministic pseudo-random values, distinct per index and per seed — same
/// generator shape as `varlen_bwd_reference_parity_cuda.rs::values`.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Casts an `F32` fixture to the dtype under test (`half::f16` is not a numr
/// `Element`, so fixtures are always built in F32 first).
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

/// Reads a CUDA result tensor back as `Vec<f32>`, casting through `F32` first
/// when it is stored at reduced precision.
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
/// particular case pass. Same construction as
/// `varlen_bwd_reference_parity_cuda.rs::varlen_bwd_tol`.
///
/// Base pair is the quantization-only backward error from rounding Q/K/V/dO to
/// `dtype` before the kernel runs (`f16`: atol 6e-3, rtol 3e-2; `bf16` has 3
/// fewer mantissa bits than f16, so its rtol is 8x f16's; `f32` keeps
/// 1e-5/1e-4).
///
/// The accumulation split is the MIRROR of the varlen kernel's, because
/// flash_v3 parallelizes over K blocks rather than Q blocks:
///
/// - `dK`/`dV` accumulate in FP32 registers for the WHOLE kernel — one
///   accumulator per K position, summed over every Q block — and are rounded
///   to `dtype` exactly once, at a plain non-atomic store. Pass
///   `n_contrib = None`: the base pair is their whole tolerance.
/// - `dQ` accumulates in FP32 within one K block, then issues ONE atomic per
///   `(q_row, d)` per K block. Each atomic rounds the running sum to `dtype`'s
///   mantissa, so for `n` sequential rounded additions Higham's classical
///   recursive-summation bound gives extra relative error `(n - 1) * u`
///   (`u` = unit roundoff: `2^-24` f32, `2^-11` f16, `2^-8` bf16).
///   `n_contrib` is passed as `seq_len_k`, an upper bound on the number of K
///   blocks (`ceil(seq_len_k / BLOCK_N) <= seq_len_k` for any `BLOCK_N >= 1`).
///   Deliberately loose, so the test does not flake if `BLOCK_N` changes.
fn flash_v3_bwd_tol(dtype: DType, n_contrib: Option<usize>) -> (f32, f32) {
    let (atol, rtol_base) = match dtype {
        DType::F32 => (1e-5, 1e-4),
        DType::F16 => (6e-3, 3e-2),
        DType::BF16 => (6e-3, 2.4e-1),
        other => unimplemented!(
            "flash_v3_bwd_tol: unsupported dtype {other:?} (v3 bwd is F32/F16/BF16 only)"
        ),
    };
    let u: f32 = match dtype {
        DType::F32 => 2f32.powi(-24),
        DType::F16 => 2f32.powi(-11),
        DType::BF16 => 2f32.powi(-8),
        other => unimplemented!(
            "flash_v3_bwd_tol: unsupported dtype {other:?} (v3 bwd is F32/F16/BF16 only)"
        ),
    };
    let rtol = match n_contrib {
        None => rtol_base,
        Some(n) => rtol_base + (n.saturating_sub(1) as f32) * u,
    };
    (atol, rtol)
}

/// Compares against the reference, printing an always-on `FLASH_V3_BWD_DIAG`
/// line (pass or fail) so the measured deviation is visible even on a green
/// run. Returns the normalized error (`max_abs_diff / ref_rms`) for the
/// dK/dV-to-dQ ratio guard.
#[allow(clippy::too_many_arguments)]
fn assert_flash_v3_bwd_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    tensor: &str,
    dtype: DType,
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
        "FLASH_V3_BWD_DIAG tensor={tensor} dtype={:?} batch={BATCH} num_heads={NUM_HEADS} \
         seq={SEQ} head_dim={head_dim} causal={causal} max_abs={max_abs:.6e} \
         max_abs_idx={max_abs_idx} max_rel={max_rel:.6e} max_rel_idx={max_rel_idx} \
         ref_rms={rms:.6e} atol={atol:.6e} rtol={rtol:.6e} tol={tol:.6e} label=\"{label}\"",
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

/// Core comparison: runs the CPU reference in F32, runs flash_v3 forward +
/// backward at `dtype` (casting inputs down and results back up through F32),
/// and checks dQ/dK/dV against the reference with the dK/dV-to-dQ ratio guard.
fn assert_flash_v3_bwd_parity(head_dim: usize, causal: bool, dtype: DType, label: &str) {
    if dtype != DType::F32 && !cfg!(feature = "f16") {
        loud_skip(
            label,
            "boostr built without the `f16` feature, so F16/BF16 tensors cannot be built",
        );
        return;
    }
    if !cuda_available() {
        loud_skip(label, "CUDA is not available on this machine");
        return;
    }
    let _lock = cuda_lock();

    let (cuda_client, cuda_dev) = cuda_setup();
    if !flash_v3::is_hopper(&cuda_client, &cuda_dev) {
        loud_skip(
            label,
            "GPU compute capability is below SM 90 — flash_v3 kernels never execute here",
        );
        return;
    }

    let n = BATCH * NUM_HEADS * SEQ * head_dim;
    let shape = [BATCH, NUM_HEADS, SEQ, head_dim];
    let q_data = values(n, 0.1);
    let k_data = values(n, 1.3);
    let v_data = values(n, 2.7);
    let do_data = values(n, 3.9);

    // CPU reference, always F32.
    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu = Tensor::<CpuRuntime>::from_slice(&q_data, &shape, &cpu_dev).unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(&k_data, &shape, &cpu_dev).unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(&v_data, &shape, &cpu_dev).unwrap();
    let do_cpu = Tensor::<CpuRuntime>::from_slice(&do_data, &shape, &cpu_dev).unwrap();

    let (out_cpu, lse_cpu) = cpu_client
        .flash_attention_fwd(
            &q_cpu, &k_cpu, &v_cpu, NUM_HEADS, NUM_HEADS, head_dim, causal, 0, None,
        )
        .unwrap();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_client
        .flash_attention_bwd(
            &do_cpu, &q_cpu, &k_cpu, &v_cpu, &out_cpu, &lse_cpu, NUM_HEADS, NUM_HEADS, head_dim,
            causal, 0,
        )
        .unwrap();
    let dq_cpu_vec = dq_cpu.to_vec::<f32>();
    let dk_cpu_vec = dk_cpu.to_vec::<f32>();
    let dv_cpu_vec = dv_cpu.to_vec::<f32>();

    // CUDA flash_v3 at `dtype`. Both halves come from flash_v3 itself so the
    // LSE convention the backward consumes is the one its own forward wrote.
    let q_c = cast_to_dtype(&q_data, &shape, &cuda_dev, dtype);
    let k_c = cast_to_dtype(&k_data, &shape, &cuda_dev, dtype);
    let v_c = cast_to_dtype(&v_data, &shape, &cuda_dev, dtype);
    let do_c = cast_to_dtype(&do_data, &shape, &cuda_dev, dtype);

    let (out_c, lse_c) = flash_v3::flash_v3_fwd(
        &cuda_client,
        &q_c,
        &k_c,
        &v_c,
        BATCH,
        NUM_HEADS,
        SEQ,
        SEQ,
        head_dim,
        causal,
    )
    .unwrap_or_else(|e| panic!("{label}: flash_v3 forward errored: {e}"))
    .unwrap_or_else(|| {
        panic!(
            "{label}: flash_v3 forward returned None on Hopper for a supported \
             (dtype={dtype:?}, head_dim={head_dim}) — the v3 module or kernel is missing"
        )
    });

    let (dq_c, dk_c, dv_c) = flash_v3::flash_v3_bwd(
        &cuda_client,
        &do_c,
        &q_c,
        &k_c,
        &v_c,
        &out_c,
        &lse_c,
        BATCH,
        NUM_HEADS,
        SEQ,
        SEQ,
        head_dim,
        causal,
    )
    .unwrap_or_else(|e| panic!("{label}: flash_v3 backward errored: {e}"))
    .unwrap_or_else(|| {
        panic!(
            "{label}: flash_v3 backward returned None on Hopper for a supported \
             (dtype={dtype:?}, head_dim={head_dim}) — the v3 bwd module or kernel is missing"
        )
    });

    let (dq_atol, dq_rtol) = flash_v3_bwd_tol(dtype, Some(SEQ));
    let (dkv_atol, dkv_rtol) = flash_v3_bwd_tol(dtype, None);

    let dq_norm = assert_flash_v3_bwd_diff(
        &read_back_f32(&dq_c),
        &dq_cpu_vec,
        dq_atol,
        dq_rtol,
        &format!("{label} dQ CUDA vs CPU [{dtype:?}]"),
        "dQ",
        dtype,
        head_dim,
        causal,
    );
    let dk_norm = assert_flash_v3_bwd_diff(
        &read_back_f32(&dk_c),
        &dk_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{label} dK CUDA vs CPU [{dtype:?}]"),
        "dK",
        dtype,
        head_dim,
        causal,
    );
    let dv_norm = assert_flash_v3_bwd_diff(
        &read_back_f32(&dv_c),
        &dv_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{label} dV CUDA vs CPU [{dtype:?}]"),
        "dV",
        dtype,
        head_dim,
        causal,
    );

    // dQ is the control: it is the one gradient flash_v3 still accumulates
    // through atomics into storage dtype, so its normalized error is an upper
    // envelope for what this kernel's arithmetic costs at this shape and
    // dtype. dK/dV are FP32-accumulated end to end and rounded once, so their
    // normalized error must sit at or below dQ's, up to noise. The pre-fix
    // defect — dK/dV summed over every K row and stored at the WRONG K
    // position, overwritten across Q blocks — puts them at O(1) relative
    // error, hundreds of times dQ's. 25x is reused unchanged from
    // `varlen_bwd_reference_parity_cuda.rs::DKV_TO_DQ_RATIO_LIMIT` (same
    // defect class): far above real post-fix noise, far below the pre-fix
    // magnitude.
    const DKV_TO_DQ_RATIO_LIMIT: f32 = 25.0;
    for (name, norm) in [("dK", dk_norm), ("dV", dv_norm)] {
        let ratio = norm / dq_norm;
        assert!(
            ratio <= DKV_TO_DQ_RATIO_LIMIT,
            "{label} {name} CUDA vs CPU [{dtype:?}]: normalized error (max_abs_diff / \
             ref_rms) is {norm:.4e}, {ratio:.1}x dQ's {dq_norm:.4e} (limit \
             {DKV_TO_DQ_RATIO_LIMIT}x) — this most likely means {name} regressed to being \
             keyed by the wrong index (summed over all K rows while `tid` indexed a Q row, \
             then stored at K position k_start + tid), or that its per-Q-block \
             contributions are overwriting instead of accumulating"
        );
    }
}

// ============================================================================
// head_dim 64 (BLOCK_M=32, BLOCK_N=64) x dtype in {F32, F16, BF16} x causal
// ============================================================================

#[test]
fn test_flash_v3_bwd_parity_hd64_f32_causal_hopper_only() {
    assert_flash_v3_bwd_parity(64, true, DType::F32, "flash_v3_bwd hd64 f32 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd64_f32_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(64, false, DType::F32, "flash_v3_bwd hd64 f32 noncausal");
}

#[test]
fn test_flash_v3_bwd_parity_hd64_f16_causal_hopper_only() {
    assert_flash_v3_bwd_parity(64, true, DType::F16, "flash_v3_bwd hd64 f16 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd64_f16_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(64, false, DType::F16, "flash_v3_bwd hd64 f16 noncausal");
}

#[test]
fn test_flash_v3_bwd_parity_hd64_bf16_causal_hopper_only() {
    assert_flash_v3_bwd_parity(64, true, DType::BF16, "flash_v3_bwd hd64 bf16 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd64_bf16_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(64, false, DType::BF16, "flash_v3_bwd hd64 bf16 noncausal");
}

// ============================================================================
// head_dim 128 (BLOCK_M=16, BLOCK_N=32) x dtype in {F32, F16, BF16} x causal
// ============================================================================

#[test]
fn test_flash_v3_bwd_parity_hd128_f32_causal_hopper_only() {
    assert_flash_v3_bwd_parity(128, true, DType::F32, "flash_v3_bwd hd128 f32 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd128_f32_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(128, false, DType::F32, "flash_v3_bwd hd128 f32 noncausal");
}

#[test]
fn test_flash_v3_bwd_parity_hd128_f16_causal_hopper_only() {
    assert_flash_v3_bwd_parity(128, true, DType::F16, "flash_v3_bwd hd128 f16 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd128_f16_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(128, false, DType::F16, "flash_v3_bwd hd128 f16 noncausal");
}

#[test]
fn test_flash_v3_bwd_parity_hd128_bf16_causal_hopper_only() {
    assert_flash_v3_bwd_parity(128, true, DType::BF16, "flash_v3_bwd hd128 bf16 causal");
}

#[test]
fn test_flash_v3_bwd_parity_hd128_bf16_noncausal_hopper_only() {
    assert_flash_v3_bwd_parity(128, false, DType::BF16, "flash_v3_bwd hd128 bf16 noncausal");
}
