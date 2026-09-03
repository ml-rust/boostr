//! Flash Attention v2 forward, SMALL-BLOCK (`_sm`) kernel variant
//! (`kernels/attention/flash_v2.cu`) at F16 and BF16, vs the CPU reference.
//!
//! # Why this file exists
//!
//! `flash_v2.cu` was deduplicated onto a `template<typename T, int HEAD_DIM,
//! int BLOCK_M, int BLOCK_N>`. Diffed against the pre-migration file, the
//! `_sm` (small block config) forward entry points existed for F32 ONLY:
//! `flash_attention_fwd_{96,128,192,256}_sm_fp32`. `flash_fwd.rs` builds the
//! kernel name unconditionally as
//! `flash_attention_fwd_{head_dim}{_sm}_{dtype_suffix}`, so an F16 or BF16
//! call that resolved to the small tile requested a symbol the pre-migration
//! `.cu` never compiled — a kernel-lookup failure. The migration added the
//! missing 8 instantiations (`_sm` x {fp16, bf16} x head_dim {96, 128, 192,
//! 256}); this file is the coverage that was missing for them.
//!
//! # Forcing the `_sm` kernel deliberately, not by accident
//!
//! `block_config` in `src/ops/cuda/attention/flash_utils.rs` picks the small
//! config via two independent gates: a hard shared-memory CAPABILITY gate
//! (the large config doesn't fit this GPU), and a soft `seq_len_q <=
//! small_block_m` PERFORMANCE gate that downgrades an otherwise-fitting large
//! config to small. This file relies on gate 2 only, so the coverage does not
//! depend on which GPU it runs on:
//!
//! - `block_config_small(head_dim)` bounds used here: 96 -> BLOCK_M=32,
//!   192 -> BLOCK_M=32, 256 -> BLOCK_M=16. `seq_len_q` is chosen strictly
//!   below each bound, so `seq_len_q <= small_block_m` holds unconditionally.
//! - Gate 2 additionally requires the small config's own shared-memory
//!   requirement to fit. At these head_dims the small config needs
//!   `(head_dim + 1) * (BLOCK_M + 2*BLOCK_N) * dtype_bytes`: 96 ->
//!   97*96*2 = 18624B, 192 -> 193*64*2 = 24704B, 256 -> 257*48*2 = 24672B
//!   (F16; BF16 is identical, same element size). All are under 25KB, far
//!   below even the pre-opt-in 48KB default cap every supported GPU
//!   guarantees, so gate 2's fit check always passes here — the `_sm` kernel
//!   is selected on any real device, not conditionally on this one.
//! - If gate 1 alone already forces small (large doesn't fit on this GPU),
//!   the same call still lands on `_sm` — the two gates only ever agree on
//!   "use small" for this file's shapes, never disagree.
//!
//! head_dim is restricted to {96, 192, 256}: `CudaClient::flash_attention_fwd`
//! (`src/ops/cuda/attention/flash.rs`) routes to the dedicated MQA/GQA kernel
//! when `window_size == 0` and `head_dim` in {32, 64, 128}
//! (`mqa_gqa::should_use_mqa_gqa`), bypassing `flash_v2.cu` entirely — head_dim
//! 128 is deliberately excluded from this file for that reason. 96/192/256
//! fall through to `flash_fwd::flash_attention_fwd_impl` (this file's target)
//! regardless of head counts. GQA (`num_kv_heads != num_heads`) is used
//! anyway, matching `flash_v2_bwd_halfprec_parity_cuda.rs`, so this also stays
//! off Flash v3 (MHA-only) independent of whether v3 dispatch is enabled.
//!
//! # Why SHORT `seq_len_q` is load-bearing
//!
//! `seq_len_q <= small_block_m` is the entire mechanism that selects `_sm`
//! here (see above). A future reader enlarging `seq_len_q` "for realism"
//! would silently move the call onto the LARGE config and destroy this
//! file's coverage while it keeps reporting green — `assert_sm_precondition`
//! below turns that into a loud panic instead of a silent pass.
//!
//! # Reference
//!
//! `CpuClient::flash_attention_fwd` (`src/ops/cpu/attention/flash.rs`),
//! always F32. It shares no indexing or tiling code with the CUDA kernel.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_v2_fwd_sm_halfprec_parity_cuda -- --nocapture

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
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;

/// `seq_len_k`: exceeds every `BLOCK_N` the small config uses at the
/// head_dims tested here (32 at 96, 16 at 192/256), so the kernel's K-tile
/// loop runs more than once within the single Q tile, and 137 is not a
/// multiple of 32/16 so the tail K tile is partial too.
const SEQ_K: usize = 137;

/// Deliberately loud: a silently skipped half-precision case would report
/// green while verifying nothing, which is exactly how the missing `_sm`
/// coverage this file adds went unnoticed.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! FLASH_V2_FWD_SM_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT running\n\
         !! the flash_v2 forward `_sm` kernel at this dtype/shape. Treat it\n\
         !! as UNTESTED, not as green.\n\
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
/// generator shape as the other `flash_v2*_cuda.rs` parity files.
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

/// `block_config_small`'s `BLOCK_M`, duplicated from
/// `src/ops/cuda/attention/flash_utils.rs` — the value this file's whole
/// `_sm`-selection argument depends on. If that table changes, this bound
/// must be re-derived, not silently widened.
fn small_block_m(head_dim: usize) -> usize {
    match head_dim {
        96 => 32,
        128 => 64,
        192 => 32,
        256 => 16,
        other => unimplemented!("small_block_m: no block_config_small entry for head_dim {other}"),
    }
}

/// Fails loudly if `seq_len_q` no longer forces `block_config`'s `seq_len_q
/// <= small_block_m` downgrade — see the module docs' "Why SHORT `seq_len_q`
/// is load-bearing" section. Without this, enlarging `seq_len_q` would
/// silently move the call onto the LARGE config and the test would keep
/// passing while testing nothing this file claims to test.
fn assert_sm_precondition(head_dim: usize, seq_len_q: usize) {
    let bound = small_block_m(head_dim);
    assert!(
        seq_len_q <= bound,
        "test bug: seq_len_q={seq_len_q} exceeds small_block_m={bound} for head_dim={head_dim}, \
         so block_config's seq_len_q<=small_block_m downgrade no longer fires and this shape no \
         longer reaches the `_sm` kernel this file exists to cover — shrink seq_len_q back down, \
         do not raise this bound"
    );
}

/// Forward tolerance `(atol, rtol)` against the F32 CPU reference, derived
/// from dtype unit roundoff `u`.
///
/// The forward output is a softmax-weighted average of V; the kernel
/// accumulates the running sum and max in FP32 registers regardless of
/// storage dtype (only Q/K/V on the way in, and O on the way out, round to
/// `dtype`), so the error is dominated by two single-rounding steps, not by
/// accumulation depth over `head_dim` or `seq_len_k`. Softmax weights form a
/// convex combination, so they don't amplify the input rounding error — the
/// output error stays within a modest constant multiple of `u`.
///
/// `u_f16 = 2^-11 ≈ 4.88e-4`, `u_bf16 = 2^-8 ≈ 3.91e-3`. The margins below
/// (rtol ≈ 61*u_f16, ≈ 31*u_bf16) match the already-vetted forward tolerance
/// for this SAME production `flash_attention_fwd` entry point at the LARGE
/// block config (`flash_v2_window_multitile_cuda.rs::flash_fwd_tol`) — block
/// config changes the tiling, not the arithmetic's rounding behavior, so the
/// same multiplier applies here.
fn flash_fwd_sm_tol(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::F16 => (4e-3, 3e-2),
        DType::BF16 => (2e-2, 1.2e-1),
        other => unimplemented!("flash_fwd_sm_tol: unsupported dtype {other:?} (F16/BF16 only)"),
    }
}

/// Compares against the reference, printing an always-on
/// `FLASH_FWD_SM_DIAG` line (pass or fail, flat `key=value` pairs) so the
/// measured deviation is visible even on a green run.
#[allow(clippy::too_many_arguments)]
fn assert_fwd_sm_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    dtype: DType,
    head_dim: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    causal: bool,
) {
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
            "{label}: kernel produced non-finite value {a} at index {i} (reference {e}) — a \
             missing/miscompiled `_sm` symbol, or a wrong shared-memory element type/size for \
             this dtype, can both produce exactly this"
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
        "FLASH_FWD_SM_DIAG dtype={dtype:?} head_dim={head_dim} seq_len_q={seq_len_q} \
         seq_len_k={seq_len_k} causal={causal} max_abs={max_abs:.6e} ref_rms={rms:.6e}"
    );

    assert!(
        rms > 1e-6,
        "{label}: reference RMS is {rms:.4e} — the fixture is degenerate, so agreement \
         would prove nothing. Fix the fixture, not the tolerance."
    );
    assert!(
        max_abs <= tol,
        "{label}: max_abs_diff {max_abs:.4e} at index {max_abs_idx} (max_rel_diff {max_rel:.4e} \
         at index {max_rel_idx}) exceeds tol {tol:.4e} (ref_rms {rms:.4e}); kernel={} \
         reference={}",
        actual[max_abs_idx],
        expected[max_abs_idx]
    );
}

/// Runs the CPU F32 reference and the CUDA `_sm` kernel at `dtype` for one
/// shape, and checks output values and the LSE.
fn assert_fwd_sm_parity(head_dim: usize, seq_len_q: usize, dtype: DType, label: &str) {
    assert_sm_precondition(head_dim, seq_len_q);

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

    let causal = true;
    let q_shape = [BATCH, NUM_HEADS, seq_len_q, head_dim];
    let kv_shape = [BATCH, NUM_KV_HEADS, SEQ_K, head_dim];
    let q_n = BATCH * NUM_HEADS * seq_len_q * head_dim;
    let kv_n = BATCH * NUM_KV_HEADS * SEQ_K * head_dim;
    let q_data = values(q_n, 0.1);
    let k_data = values(kv_n, 1.3);
    let v_data = values(kv_n, 2.7);

    // CPU reference, always F32.
    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu = Tensor::<CpuRuntime>::from_slice(&q_data, &q_shape, &cpu_dev).unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(&k_data, &kv_shape, &cpu_dev).unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(&v_data, &kv_shape, &cpu_dev).unwrap();
    let (out_cpu, lse_cpu) = cpu_client
        .flash_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            NUM_HEADS,
            NUM_KV_HEADS,
            head_dim,
            causal,
            0,
            None,
        )
        .expect("CPU reference flash_attention_fwd failed");
    let out_cpu_vec = out_cpu.to_vec::<f32>();
    let lse_cpu_vec = lse_cpu.to_vec::<f32>();

    // CUDA at `dtype`, through the SAME public dispatch a real caller uses —
    // this is what makes the shape choice (GQA + head_dim outside
    // {32,64,128}, seq_len_q <= small_block_m) load-bearing: it is what
    // forces the call through `flash_fwd::flash_attention_fwd_impl` and onto
    // the `_sm` kernel specifically.
    let (cuda_client, cuda_dev) = cuda_setup();
    let q_c = cast_to_dtype(&q_data, &q_shape, &cuda_dev, dtype);
    let k_c = cast_to_dtype(&k_data, &kv_shape, &cuda_dev, dtype);
    let v_c = cast_to_dtype(&v_data, &kv_shape, &cuda_dev, dtype);

    let (out_c, lse_c) = cuda_client
        .flash_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            NUM_HEADS,
            NUM_KV_HEADS,
            head_dim,
            causal,
            0,
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: CUDA flash_attention_fwd failed: {e}"));

    assert_eq!(out_c.shape(), &q_shape, "{label}: output shape is wrong");
    assert_eq!(
        lse_c.shape(),
        &[BATCH, NUM_HEADS, seq_len_q],
        "{label}: LSE shape is wrong"
    );

    let (atol, rtol) = flash_fwd_sm_tol(dtype);
    assert_fwd_sm_diff(
        &read_back_f32(&out_c),
        &out_cpu_vec,
        atol,
        rtol,
        &format!("{label} out"),
        dtype,
        head_dim,
        seq_len_q,
        SEQ_K,
        causal,
    );
    // LSE is F32 storage regardless of `dtype` (see flash_fwd.rs), but its
    // VALUE still depends on the Q/K rounding and the `_sm` kernel's own
    // softmax accumulation, so it is checked independently of the output.
    assert_fwd_sm_diff(
        &read_back_f32(&lse_c),
        &lse_cpu_vec,
        atol,
        rtol,
        &format!("{label} lse"),
        dtype,
        head_dim,
        seq_len_q,
        SEQ_K,
        causal,
    );
}

// ============================================================================
// head_dim 96, small_block_m=32 — seq_len_q=24 is strictly below it, so
// block_config's downgrade fires unconditionally. GQA keeps this off Flash
// v3 independent of head_dim (which already excludes MQA/GQA and v3 needs
// MHA anyway).
// ============================================================================

#[test]
fn test_flash_v2_fwd_sm_hd96_f16() {
    assert_fwd_sm_parity(96, 24, DType::F16, "flash_v2_fwd_sm hd96 f16");
}

#[test]
fn test_flash_v2_fwd_sm_hd96_bf16() {
    assert_fwd_sm_parity(96, 24, DType::BF16, "flash_v2_fwd_sm hd96 bf16");
}

// ============================================================================
// head_dim 192, small_block_m=32 — seq_len_q=24, same reasoning as hd96.
// ============================================================================

#[test]
fn test_flash_v2_fwd_sm_hd192_f16() {
    assert_fwd_sm_parity(192, 24, DType::F16, "flash_v2_fwd_sm hd192 f16");
}

#[test]
fn test_flash_v2_fwd_sm_hd192_bf16() {
    assert_fwd_sm_parity(192, 24, DType::BF16, "flash_v2_fwd_sm hd192 bf16");
}

// ============================================================================
// head_dim 256, small_block_m=16 — seq_len_q=11, strictly below the tighter
// bound at this head_dim.
// ============================================================================

#[test]
fn test_flash_v2_fwd_sm_hd256_f16() {
    assert_fwd_sm_parity(256, 11, DType::F16, "flash_v2_fwd_sm hd256 f16");
}

#[test]
fn test_flash_v2_fwd_sm_hd256_bf16() {
    assert_fwd_sm_parity(256, 11, DType::BF16, "flash_v2_fwd_sm hd256 bf16");
}
