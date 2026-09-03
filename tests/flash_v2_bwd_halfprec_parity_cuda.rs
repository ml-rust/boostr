//! Flash Attention v2 backward (`kernels/attention/flash_v2_bwd.cu`) at F16
//! and BF16, vs the CPU reference, for dQ, dK and dV.
//!
//! # Why this file exists
//!
//! `flash_v2_bwd.cu` was collapsed from three per-dtype implementations into
//! one `template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>`. The
//! riskiest part of that rewrite is shared-memory sizing: the kernel declares
//! `extern __shared__` with element type `T` (`float` / `__half` /
//! `__nv_bfloat16`), while the launcher (`bwd_block_config` +
//! `compute_bwd_smem` in `src/ops/cuda/attention/flash_utils.rs`) allocates
//! `(2*BLOCK_N + 2*BLOCK_M) * HEAD_DIM * dtype.size_in_bytes()`. F16/BF16
//! therefore receive HALF the bytes F32 does for the same block config — if
//! the templated kernel got the element type wrong, the half-precision
//! launches write out of bounds.
//!
//! An `ncu` profile of the full parity suite showed only F32
//! `flash_attention_bwd_*` launches (head_dims 32, 96, 192, 256), plus FP8
//! from a separate kernel file. No F16 or BF16 `flash_attention_bwd_*` launch
//! happened anywhere. This file is the missing coverage.
//!
//! # Reaching `flash_v2_bwd.cu` deliberately, not by accident
//!
//! `CudaClient::flash_attention_bwd` (`src/ops/cuda/attention/flash.rs`)
//! tries, in order: Flash v3 (Hopper only, MHA only, `window_size == 0`), then
//! the dedicated MQA/GQA kernel (`window_size == 0`, dtype in
//! {F32,F16,BF16}, `head_dim` in {32,64,128}, `num_heads % num_kv_heads ==
//! 0`), and only THEN falls through to `flash_bwd::flash_attention_bwd_impl`
//! — this file's target. `flash_attention_bwd_impl` is `pub(super)`, so it is
//! reachable only through the public trait method, and only by construction
//! of a shape neither earlier stage can take:
//!
//! - GQA at `head_dim` 96 / 192 / 256: `head_dim` is not in {32,64,128}, so
//!   the MQA/GQA gate rejects it regardless of head counts, and
//!   `num_kv_heads != num_heads` independently rejects Flash v3 — so this
//!   path is forced off the fast kernels on ANY GPU, Hopper included.
//! - `window_size != 0` at `head_dim = 64` with `num_heads % num_kv_heads ==
//!   0`: a shape the MQA/GQA gate WOULD otherwise accept, bypassed solely by
//!   the nonzero window, which both Flash v3 and the MQA/GQA kernel refuse.
//!
//! # Reference
//!
//! `CpuClient::flash_attention_fwd` / `flash_attention_bwd`
//! (`src/ops/cpu/attention/flash.rs` -> `impl_generic`'s
//! `standard_attention_bwd`), always in F32. It shares no indexing,
//! accumulation, or tiling code with the CUDA kernel.
//!
//! # Multi-tile shapes, not single-block toys
//!
//! `seq_len_q == seq_len_k == 260` is chosen to exceed every BLOCK_M/BLOCK_N
//! this kernel can pick at any head_dim tested here (largest is BLOCK_N=128 at
//! head_dim 96, largest BLOCK_M=64 at head_dim 64/96/192/256) — see
//! `bwd_block_config_large`/`_small` in `flash_utils.rs`. So both the grid's
//! K-tile loop (`grid_y = ceil(seq_len_k / BLOCK_N)`) and the kernel's
//! internal Q-block loop run more than once, on whichever block config this
//! GPU picks, exercising cross-tile shared-memory reuse instead of a single
//! self-contained block.
//!
//! # Kernel-name assertion
//!
//! No `#[doc(hidden)]` hook exposing the resolved backward kernel name exists
//! for this path (unlike `paged_attention_fwd_block_config.rs` /
//! `mla_block_config.rs`), so none is added here per the no-`src/`-changes
//! constraint. Kernel launch coverage for these shapes must be confirmed with
//! `ncu` outside this test file.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_v2_bwd_halfprec_parity_cuda -- --nocapture

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

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

/// Exceeds every BLOCK_M/BLOCK_N this kernel can pick at head_dim 64/96/192/
/// 256 (largest BLOCK_N is 128, largest BLOCK_M is 64), so the K-tile grid
/// loop and the kernel's internal Q-block loop both run more than once
/// regardless of which block config (`_sm` or not) this GPU's shared-memory
/// budget selects. Not a multiple of 128/64/32/16, so the tail K tile and
/// tail Q block are partial too.
const SEQ: usize = 260;

/// Deliberately loud: a silently skipped half-precision case would report
/// green while verifying nothing, which is exactly how the missing coverage
/// this file adds went unnoticed.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! FLASH_V2_BWD_HALFPREC_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT running\n\
         !! the flash_v2 backward kernel at this dtype/shape. Treat it as\n\
         !! UNTESTED, not as green.\n\
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
/// generator shape as `flash_v3_bwd_parity_cuda.rs::values`.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Casts an `F32` fixture to the dtype under test. `half::f16` is not a numr
/// `Element`, so fixtures are always built in F32 first, per house rules.
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

/// Reads a CUDA result tensor back as `Vec<f32>`, casting through F32 first
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

/// Worst-case K-tile count for `flash_v2_bwd.cu`'s dQ atomic-accumulation
/// bound: `ceil(seq_len_k / BLOCK_N)` using the SMALLEST `BLOCK_N` this
/// kernel can pick at `head_dim` (`bwd_block_config_small` in
/// `flash_utils.rs`, duplicated here as the values this test depends on — if
/// that table changes, this bound must be re-derived, not silently widened).
/// A larger `BLOCK_N` (the `_large` config) only means fewer tiles, never
/// more, so this is a correct upper bound regardless of which config the
/// device actually selects.
fn max_k_tiles(head_dim: usize, seq_len_k: usize) -> usize {
    let block_n_small = match head_dim {
        32 | 64 => 64,
        96 | 128 => 32,
        192 | 256 => 16,
        other => {
            unimplemented!("max_k_tiles: no bwd_block_config_small entry for head_dim {other}")
        }
    };
    seq_len_k.div_ceil(block_n_small)
}

/// Backward tolerance, derived from first principles — same construction as
/// `flash_v3_bwd_parity_cuda.rs::flash_v3_bwd_tol`, because
/// `flash_v2_bwd.cu`'s accumulation shape is identical: dQ gets one
/// `atomicAdd` per (q_row, d) per K TILE (`flash_v2_bwd.cu` lines ~180-268,
/// the `for (int q_block ...)` loop issues exactly one atomic per K-tile grid
/// cell per Q row it visits), while dK/dV accumulate in FP32 registers for
/// the WHOLE kernel — one accumulator per K row, summed over every Q block —
/// and are written once with a plain, non-atomic store.
///
/// Base pair is the quantization-only backward error from rounding Q/K/V/dO
/// to `dtype` before the kernel runs (`f16`: atol 6e-3, rtol 3e-2; `bf16` has
/// 3 fewer mantissa bits than f16, so its rtol is wider).
///
/// - dK/dV: `n_contrib = None`, base pair is the whole tolerance (single
///   rounding).
/// - dQ: `n_contrib = max_k_tiles(head_dim, seq_len_k)`. Higham's classical
///   recursive-summation bound adds `(n - 1) * u` relative error for `n`
///   sequential dtype-rounded additions (`u` = unit roundoff: `2^-11` f16,
///   `2^-8` bf16).
fn flash_v2_bwd_tol(dtype: DType, n_contrib: Option<usize>) -> (f32, f32) {
    let (atol, rtol_base) = match dtype {
        DType::F16 => (6e-3, 3e-2),
        DType::BF16 => (6e-3, 2.4e-1),
        other => unimplemented!(
            "flash_v2_bwd_tol: unsupported dtype {other:?} (this file covers F16/BF16 only)"
        ),
    };
    let u: f32 = match dtype {
        DType::F16 => 2f32.powi(-11),
        DType::BF16 => 2f32.powi(-8),
        other => unimplemented!(
            "flash_v2_bwd_tol: unsupported dtype {other:?} (this file covers F16/BF16 only)"
        ),
    };
    let rtol = match n_contrib {
        None => rtol_base,
        Some(n) => rtol_base + (n.saturating_sub(1) as f32) * u,
    };
    (atol, rtol)
}

/// Compares against the reference, printing an always-on `FLASH_BWD_DIAG`
/// line (pass or fail, flat `key=value` pairs) so the measured deviation is
/// visible even on a green run. Returns the normalized error
/// (`max_abs_diff / ref_rms`) for the dK/dV-to-dQ ratio guard.
#[allow(clippy::too_many_arguments)]
fn assert_flash_bwd_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    tensor: &str,
    dtype: DType,
    head_dim: usize,
    window_size: usize,
    seq_len: usize,
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
            "{label}: kernel produced non-finite value {a} at index {i} (reference {e}) — an \
             out-of-bounds shared-memory write can produce exactly this"
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
        "FLASH_BWD_DIAG tensor={tensor} dtype={:?} head_dim={head_dim} window_size={window_size} \
         seq_len={seq_len} max_abs={max_abs:.6e} ref_rms={rms:.6e} label=\"{label}\"",
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

/// Core comparison: runs the CPU reference in F32, runs the CUDA
/// `flash_attention_fwd`/`flash_attention_bwd` trait methods at `dtype`
/// (casting inputs down and results back up through F32), and checks
/// dQ/dK/dV against the reference with the dK/dV-to-dQ ratio guard. `causal`
/// is fixed `true`: it is the production configuration and lets the window
/// case exercise both masks at once.
#[allow(clippy::too_many_arguments)]
fn assert_flash_v2_bwd_halfprec_parity(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
    dtype: DType,
    label: &str,
) {
    if !cfg!(feature = "f16") {
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

    let q_shape = [BATCH, num_heads, SEQ, head_dim];
    let kv_shape = [BATCH, num_kv_heads, SEQ, head_dim];
    let q_n = BATCH * num_heads * SEQ * head_dim;
    let kv_n = BATCH * num_kv_heads * SEQ * head_dim;
    let q_data = values(q_n, 0.1);
    let k_data = values(kv_n, 1.3);
    let v_data = values(kv_n, 2.7);
    let do_data = values(q_n, 3.9);

    // CPU reference, always F32.
    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu = Tensor::<CpuRuntime>::from_slice(&q_data, &q_shape, &cpu_dev).unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(&k_data, &kv_shape, &cpu_dev).unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(&v_data, &kv_shape, &cpu_dev).unwrap();
    let do_cpu = Tensor::<CpuRuntime>::from_slice(&do_data, &q_shape, &cpu_dev).unwrap();

    let (out_cpu, lse_cpu) = cpu_client
        .flash_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            window_size,
            None,
        )
        .expect("CPU reference flash_attention_fwd failed");
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_client
        .flash_attention_bwd(
            &do_cpu,
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &out_cpu,
            &lse_cpu,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            window_size,
        )
        .expect("CPU reference flash_attention_bwd failed");
    let dq_cpu_vec = dq_cpu.to_vec::<f32>();
    let dk_cpu_vec = dk_cpu.to_vec::<f32>();
    let dv_cpu_vec = dv_cpu.to_vec::<f32>();

    // CUDA at `dtype`, through the SAME public dispatch a real caller uses —
    // this is what makes the shape choice (GQA + non-{32,64,128} head_dim, or
    // a nonzero window) load-bearing: it is what forces the call through
    // `flash_bwd::flash_attention_bwd_impl` instead of Flash v3 or the
    // dedicated MQA/GQA kernel.
    let (cuda_client, cuda_dev) = cuda_setup();
    let q_c = cast_to_dtype(&q_data, &q_shape, &cuda_dev, dtype);
    let k_c = cast_to_dtype(&k_data, &kv_shape, &cuda_dev, dtype);
    let v_c = cast_to_dtype(&v_data, &kv_shape, &cuda_dev, dtype);
    let do_c = cast_to_dtype(&do_data, &q_shape, &cuda_dev, dtype);

    let (out_c, lse_c) = cuda_client
        .flash_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            window_size,
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: CUDA flash_attention_fwd failed: {e}"));

    let (dq_c, dk_c, dv_c) = cuda_client
        .flash_attention_bwd(
            &do_c,
            &q_c,
            &k_c,
            &v_c,
            &out_c,
            &lse_c,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            window_size,
        )
        .unwrap_or_else(|e| panic!("{label}: CUDA flash_attention_bwd failed: {e}"));

    assert_eq!(dq_c.shape(), &q_shape, "{label}: dQ shape is wrong");
    assert_eq!(dk_c.shape(), &kv_shape, "{label}: dK shape is wrong");
    assert_eq!(dv_c.shape(), &kv_shape, "{label}: dV shape is wrong");

    let (dq_atol, dq_rtol) = flash_v2_bwd_tol(dtype, Some(max_k_tiles(head_dim, SEQ)));
    let (dkv_atol, dkv_rtol) = flash_v2_bwd_tol(dtype, None);

    let dq_norm = assert_flash_bwd_diff(
        &read_back_f32(&dq_c),
        &dq_cpu_vec,
        dq_atol,
        dq_rtol,
        &format!("{label} dQ CUDA vs CPU [{dtype:?}]"),
        "dQ",
        dtype,
        head_dim,
        window_size,
        SEQ,
    );
    let dk_norm = assert_flash_bwd_diff(
        &read_back_f32(&dk_c),
        &dk_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{label} dK CUDA vs CPU [{dtype:?}]"),
        "dK",
        dtype,
        head_dim,
        window_size,
        SEQ,
    );
    let dv_norm = assert_flash_bwd_diff(
        &read_back_f32(&dv_c),
        &dv_cpu_vec,
        dkv_atol,
        dkv_rtol,
        &format!("{label} dV CUDA vs CPU [{dtype:?}]"),
        "dV",
        dtype,
        head_dim,
        window_size,
        SEQ,
    );

    // dQ is the control: it is the gradient still accumulated through
    // atomics into a rounded intermediate, so its normalized error is an
    // upper envelope for what this kernel's arithmetic costs at this shape
    // and dtype. dK/dV are FP32-accumulated end to end and rounded once, so
    // their normalized error must sit at or below dQ's, up to noise. A
    // defect that attributed dK/dV to the wrong K position, or an
    // out-of-bounds shared-memory read/write corrupting the K/V tile, would
    // put them at O(1) relative error, orders of magnitude above dQ's. Same
    // 25x limit as `flash_v3_bwd_parity_cuda.rs` (same defect class): far
    // above real post-fix noise, far below a corruption-scale error.
    const DKV_TO_DQ_RATIO_LIMIT: f32 = 25.0;
    for (name, norm) in [("dK", dk_norm), ("dV", dv_norm)] {
        let ratio = norm / dq_norm;
        assert!(
            ratio <= DKV_TO_DQ_RATIO_LIMIT,
            "{label} {name} CUDA vs CPU [{dtype:?}]: normalized error (max_abs_diff / \
             ref_rms) is {norm:.4e}, {ratio:.1}x dQ's {dq_norm:.4e} (limit \
             {DKV_TO_DQ_RATIO_LIMIT}x) — this most likely means the shared-memory rewrite of \
             flash_v2_bwd.cu corrupted {name}'s K/V tile (wrong element type/size for this \
             dtype) or mis-attributed it to the wrong K position"
        );
    }
}

// ============================================================================
// head_dim 96/192/256, GQA — head_dim alone excludes the MQA/GQA kernel
// (needs 32/64/128) and num_kv_heads != num_heads excludes Flash v3 (MHA
// only), on any GPU including Hopper. window_size = 0, causal = true.
// ============================================================================

#[test]
fn test_flash_v2_bwd_hd96_gqa_f16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 96, 0, DType::F16, "flash_v2_bwd hd96 gqa f16");
}

#[test]
fn test_flash_v2_bwd_hd96_gqa_bf16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 96, 0, DType::BF16, "flash_v2_bwd hd96 gqa bf16");
}

#[test]
fn test_flash_v2_bwd_hd192_gqa_f16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 192, 0, DType::F16, "flash_v2_bwd hd192 gqa f16");
}

#[test]
fn test_flash_v2_bwd_hd192_gqa_bf16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 192, 0, DType::BF16, "flash_v2_bwd hd192 gqa bf16");
}

#[test]
fn test_flash_v2_bwd_hd256_gqa_f16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 256, 0, DType::F16, "flash_v2_bwd hd256 gqa f16");
}

#[test]
fn test_flash_v2_bwd_hd256_gqa_bf16() {
    assert_flash_v2_bwd_halfprec_parity(8, 2, 256, 0, DType::BF16, "flash_v2_bwd hd256 gqa bf16");
}

// ============================================================================
// head_dim 64, GQA, window_size != 0 — a shape the MQA/GQA gate WOULD accept
// (head_dim in {32,64,128}, num_heads % num_kv_heads == 0) if not for the
// nonzero window, which both Flash v3 and the MQA/GQA kernel refuse.
// ============================================================================

#[test]
fn test_flash_v2_bwd_hd64_gqa_window_f16() {
    assert_flash_v2_bwd_halfprec_parity(
        8,
        2,
        64,
        48,
        DType::F16,
        "flash_v2_bwd hd64 gqa window=48 f16",
    );
}

#[test]
fn test_flash_v2_bwd_hd64_gqa_window_bf16() {
    assert_flash_v2_bwd_halfprec_parity(
        8,
        2,
        64,
        48,
        DType::BF16,
        "flash_v2_bwd hd64 gqa window=48 bf16",
    );
}
