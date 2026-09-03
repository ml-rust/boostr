//! Sliding-window Flash Attention v2 (`kernels/attention/flash_v2.cu`) across
//! MULTIPLE query tiles, vs the CPU reference.
//!
//! # Why this file exists
//!
//! `flash_v2.cu` skips whole K blocks when they fall outside the sliding
//! window. A K block is only safe to skip when it is outside the window for
//! EVERY query row in the Q tile, which is governed by the FIRST query row —
//! its window reaches furthest back. The kernel used the LAST query row
//! instead, whose window reaches back the least, so the skip was maximally
//! aggressive and discarded K blocks that earlier rows of the same tile
//! legitimately attend to. The forward output and the LSE were both wrong;
//! the backward, which masks correctly per position, then amplified the bad
//! LSE into gradients orders of magnitude too large.
//!
//! # Why the shapes here are load-bearing — do NOT shrink them
//!
//! The bug is unreachable when `seq_len_q <= BLOCK_M`, because a single Q tile
//! has `q_start == first query == last query` and both rules agree. Every
//! pre-existing windowed CUDA test used `seq_len` 12 or 16, or decode
//! (`seq_len_q == 1`), so all of them were single-tile and none could observe
//! it.
//!
//! `SEQ_MULTI = 260` and `SEQ_MULTI_WIDE = 384` both exceed the largest
//! `BLOCK_M` the forward path can select (128, at head_dim 32/64/128 — see
//! `block_config_large`/`block_config_small` in
//! `src/ops/cuda/attention/flash_utils.rs`), so at least three Q tiles exist
//! and tiles after the first exercise the skip decision. 260 is deliberately
//! not a multiple of 128/64/32/16, so the tail Q tile and tail K tile are
//! partial as well.
//!
//! The windows (48, 96) are small relative to those sequence lengths on
//! purpose: the skip only engages when whole K blocks fall before the
//! earliest query's window. A window at or above `seq_len` makes the skip
//! dead code and the coverage vanishes.
//!
//! Shrinking `SEQ_MULTI*` below 129, or widening the windows toward `seq_len`,
//! silently destroys this file's entire purpose while leaving it green.
//!
//! # Reaching `flash_v2.cu` deliberately
//!
//! `CudaClient::flash_attention_fwd` (`src/ops/cuda/attention/flash.rs`) gates
//! the decode kernel, Flash v3, and the dedicated MQA/GQA kernel all on
//! `window_size == 0`. A nonzero window therefore always falls through to
//! `flash_fwd::flash_attention_fwd_impl` — this file's target — on every GPU.
//!
//! # Reference
//!
//! `CpuClient::flash_attention_fwd` / `flash_attention_bwd`
//! (`src/ops/cpu/attention/flash.rs`), always in F32. It shares no indexing,
//! tiling, or block-skipping code with the CUDA kernel.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_v2_window_multitile_cuda -- --nocapture

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

const BATCH: usize = 1;
const HEADS: usize = 2;

/// Three Q tiles at BLOCK_M=128, with a partial tail tile and a partial tail
/// K tile. See the module docs before changing this.
const SEQ_MULTI: usize = 260;

/// Exactly three Q tiles at BLOCK_M=128, six at BLOCK_M=64. See the module
/// docs before changing this.
const SEQ_MULTI_WIDE: usize = 384;

/// Deliberately loud: a silently skipped case reports green while verifying
/// nothing, which is exactly how the single-tile coverage gap this file
/// closes went unnoticed.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! FLASH_V2_WINDOW_MULTITILE_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT running\n\
         !! the flash_v2 sliding-window kernel at this dtype/shape. Treat it\n\
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

/// Deterministic pseudo-random values, distinct per index and per seed.
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

/// Forward tolerance `(atol, rtol)` against the F32 CPU reference. The
/// forward output is a softmax-weighted average of V, so the reduced-precision
/// error is dominated by rounding the inputs, not by accumulation depth (the
/// kernel accumulates the running sum in FP32 registers at every dtype).
///
/// These are far tighter than the block-skip bug's signature: dropping a K
/// block loses most of a query row's window, which moves the output by
/// O(ref_rms), not by a rounding step.
fn flash_fwd_tol(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::F32 => (1e-5, 1e-3),
        DType::F16 => (4e-3, 3e-2),
        DType::BF16 => (2e-2, 1.2e-1),
        other => unimplemented!("flash_fwd_tol: unsupported dtype {other:?}"),
    }
}

/// Compares against the reference, printing an always-on
/// `FLASH_FWD_WINDOW_DIAG` line (pass or fail, flat `key=value`) so the
/// measured deviation is visible even on a green run.
#[allow(clippy::too_many_arguments)]
fn assert_window_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    dtype: DType,
    head_dim: usize,
    window_size: usize,
    seq_len: usize,
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
        sq_sum += (*e as f64) * (*e as f64);
    }
    let rms = (sq_sum / expected.len() as f64).sqrt() as f32;
    let tol = atol + rtol * rms;

    println!(
        "FLASH_FWD_WINDOW_DIAG dtype={dtype:?} head_dim={head_dim} window_size={window_size} \
         seq_len={seq_len} causal={causal} max_abs={max_abs:.6e} ref_rms={rms:.6e} \
         label=\"{label}\""
    );

    assert!(
        rms > 1e-6,
        "{label}: reference RMS is {rms:.4e} — the fixture is degenerate, so agreement \
         would prove nothing. Fix the fixture, not the tolerance."
    );
    assert!(
        max_abs <= tol,
        "{label}: max_abs_diff {max_abs:.4e} at index {max_abs_idx} exceeds tol {tol:.4e} \
         (ref_rms {rms:.4e}); kernel={} reference={}. A K block dropped by the sliding-window \
         skip for queries that legitimately need it looks exactly like this.",
        actual[max_abs_idx],
        expected[max_abs_idx]
    );
}

/// Runs the CPU F32 reference and the CUDA kernel at `dtype` for one windowed
/// multi-tile forward shape, and checks output values and the LSE.
fn assert_window_fwd_parity(
    head_dim: usize,
    seq_len: usize,
    window_size: usize,
    causal: bool,
    dtype: DType,
    label: &str,
) {
    assert!(
        seq_len > 128,
        "{label}: seq_len {seq_len} is at or below the largest BLOCK_M (128), so this shape \
         is single-Q-tile and cannot exercise the sliding-window K-block skip at all. See \
         the module docs."
    );
    assert!(
        window_size < seq_len / 2,
        "{label}: window {window_size} is too wide relative to seq_len {seq_len} for whole \
         K blocks to fall outside it, so the skip never engages."
    );

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

    let shape = [BATCH, HEADS, seq_len, head_dim];
    let n = BATCH * HEADS * seq_len * head_dim;
    let q_data = values(n, 0.1);
    let k_data = values(n, 1.3);
    let v_data = values(n, 2.7);

    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu = Tensor::<CpuRuntime>::from_slice(&q_data, &shape, &cpu_dev).unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(&k_data, &shape, &cpu_dev).unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(&v_data, &shape, &cpu_dev).unwrap();
    let (out_cpu, lse_cpu) = cpu_client
        .flash_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            HEADS,
            HEADS,
            head_dim,
            causal,
            window_size,
            None,
        )
        .expect("CPU reference flash_attention_fwd failed");
    let out_cpu_vec = out_cpu.to_vec::<f32>();
    let lse_cpu_vec = lse_cpu.to_vec::<f32>();

    let (cuda_client, cuda_dev) = cuda_setup();
    let q_c = cast_to_dtype(&q_data, &shape, &cuda_dev, dtype);
    let k_c = cast_to_dtype(&k_data, &shape, &cuda_dev, dtype);
    let v_c = cast_to_dtype(&v_data, &shape, &cuda_dev, dtype);
    let (out_c, lse_c) = cuda_client
        .flash_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            HEADS,
            HEADS,
            head_dim,
            causal,
            window_size,
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: CUDA flash_attention_fwd failed: {e}"));

    let (atol, rtol) = flash_fwd_tol(dtype);
    assert_window_diff(
        &read_back_f32(&out_c),
        &out_cpu_vec,
        atol,
        rtol,
        &format!("{label} out"),
        dtype,
        head_dim,
        window_size,
        seq_len,
        causal,
    );
    // The LSE is where the bug was loudest: a dropped K block removes terms
    // from the log-sum-exp, so it is checked on its own rather than only via
    // the normalized output.
    assert_window_diff(
        &read_back_f32(&lse_c),
        &lse_cpu_vec,
        atol,
        rtol,
        &format!("{label} lse"),
        dtype,
        head_dim,
        window_size,
        seq_len,
        causal,
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f32_hd64_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI,
        48,
        true,
        DType::F32,
        "fwd f32 hd64 seq260 win48 causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f32_hd64_non_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI,
        48,
        false,
        DType::F32,
        "fwd f32 hd64 seq260 win48 non-causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f32_hd64_wide_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI_WIDE,
        96,
        true,
        DType::F32,
        "fwd f32 hd64 seq384 win96 causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f32_hd64_wide_non_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI_WIDE,
        96,
        false,
        DType::F32,
        "fwd f32 hd64 seq384 win96 non-causal",
    );
}

/// head_dim 128 selects BLOCK_M=128 with BLOCK_N=64, so the K blocks are half
/// the width of the head_dim-64 case and a different set of them lands on the
/// wrong side of the skip decision.
#[test]
fn flash_v2_window_multitile_fwd_f32_hd128_causal() {
    assert_window_fwd_parity(
        128,
        SEQ_MULTI,
        48,
        true,
        DType::F32,
        "fwd f32 hd128 seq260 win48 causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f16_hd64_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI,
        48,
        true,
        DType::F16,
        "fwd f16 hd64 seq260 win48 causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_bf16_hd64_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI,
        48,
        true,
        DType::BF16,
        "fwd bf16 hd64 seq260 win48 causal",
    );
}

#[test]
fn flash_v2_window_multitile_fwd_f16_hd64_non_causal() {
    assert_window_fwd_parity(
        64,
        SEQ_MULTI,
        48,
        false,
        DType::F16,
        "fwd f16 hd64 seq260 win48 non-causal",
    );
}

/// Backward at the same multi-tile windowed shape. The backward kernel masks
/// correctly per position, so it consumes the forward's LSE as given: a
/// forward that dropped K blocks yields `exp(score - too_small_lse)`, which
/// blows the gradients up by orders of magnitude. This is how the forward bug
/// was first observed, so it is pinned here too.
#[test]
fn flash_v2_window_multitile_bwd_f32_hd64_causal() {
    let (head_dim, seq_len, window_size, causal) = (64usize, SEQ_MULTI, 48usize, true);
    let label = "bwd f32 hd64 seq260 win48 causal";

    assert!(
        seq_len > 128,
        "{label}: seq_len must exceed the largest BLOCK_M (128) or the forward is \
         single-Q-tile and the skip logic is never exercised. See the module docs."
    );

    if !cuda_available() {
        loud_skip(label, "CUDA is not available on this machine");
        return;
    }
    let _lock = cuda_lock();

    let shape = [BATCH, HEADS, seq_len, head_dim];
    let n = BATCH * HEADS * seq_len * head_dim;
    let q_data = values(n, 0.1);
    let k_data = values(n, 1.3);
    let v_data = values(n, 2.7);
    let do_data = values(n, 3.9);

    let (cpu_client, cpu_dev) = cpu_setup();
    let q_cpu = Tensor::<CpuRuntime>::from_slice(&q_data, &shape, &cpu_dev).unwrap();
    let k_cpu = Tensor::<CpuRuntime>::from_slice(&k_data, &shape, &cpu_dev).unwrap();
    let v_cpu = Tensor::<CpuRuntime>::from_slice(&v_data, &shape, &cpu_dev).unwrap();
    let do_cpu = Tensor::<CpuRuntime>::from_slice(&do_data, &shape, &cpu_dev).unwrap();
    let (out_cpu, lse_cpu) = cpu_client
        .flash_attention_fwd(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            HEADS,
            HEADS,
            head_dim,
            causal,
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
            HEADS,
            HEADS,
            head_dim,
            causal,
            window_size,
        )
        .expect("CPU reference flash_attention_bwd failed");
    let dq_cpu_vec = dq_cpu.to_vec::<f32>();
    let dk_cpu_vec = dk_cpu.to_vec::<f32>();
    let dv_cpu_vec = dv_cpu.to_vec::<f32>();

    let (cuda_client, cuda_dev) = cuda_setup();
    let q_c = Tensor::<CudaRuntime>::from_slice(&q_data, &shape, &cuda_dev).unwrap();
    let k_c = Tensor::<CudaRuntime>::from_slice(&k_data, &shape, &cuda_dev).unwrap();
    let v_c = Tensor::<CudaRuntime>::from_slice(&v_data, &shape, &cuda_dev).unwrap();
    let do_c = Tensor::<CudaRuntime>::from_slice(&do_data, &shape, &cuda_dev).unwrap();
    let (out_c, lse_c) = cuda_client
        .flash_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            HEADS,
            HEADS,
            head_dim,
            causal,
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
            HEADS,
            HEADS,
            head_dim,
            causal,
            window_size,
        )
        .unwrap_or_else(|e| panic!("{label}: CUDA flash_attention_bwd failed: {e}"));

    // dQ is accumulated with atomicAdd across K tiles, so its tolerance is
    // wider than the forward's; dK/dV accumulate in FP32 registers.
    for (actual, expected, tensor, rtol) in [
        (read_back_f32(&dq_c), &dq_cpu_vec, "dQ", 2e-3f32),
        (read_back_f32(&dk_c), &dk_cpu_vec, "dK", 2e-3f32),
        (read_back_f32(&dv_c), &dv_cpu_vec, "dV", 2e-3f32),
    ] {
        assert_window_diff(
            &actual,
            expected,
            1e-4,
            rtol,
            &format!("{label} {tensor}"),
            DType::F32,
            head_dim,
            window_size,
            seq_len,
            causal,
        );
    }
}
