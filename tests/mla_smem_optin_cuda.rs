//! Regression tests for the MLA SDPA shared-memory opt-in in
//! `src/ops/cuda/attention/mla.rs`.
//!
//! Two defects lived in that launcher:
//!
//! 1. It set `LaunchConfig::shared_mem_bytes` above the 48KB CUDA default
//!    without ever calling `set_smem_attribute`, so every configuration
//!    needing more than the default limit failed at launch with
//!    `CUDA_ERROR_INVALID_VALUE` ("invalid argument") on a device that would
//!    have granted the request. Every other smem-using attention launcher in
//!    the crate opts in; this one did not.
//! 2. It sized shared memory with the *input* dtype's element size. All three
//!    kernels in `sdpa.cu` stage Q/K/V as `float` and convert F16/BF16 on
//!    load, so an F16/BF16 launch requested half the bytes the kernel indexes
//!    and read/wrote past the end of the allocation.
//!
//! Both are fixed by `sdpa_smem_size` (always `sizeof(float)` per element) plus
//! a `device_max_smem()` gate and a `set_smem_attribute` call.
//!
//! `sdpa.cu` hardcodes `BLOCK_M = BLOCK_N = 128` in `#define`s and exports
//! exactly three `extern "C"` kernels, so there is no smaller-tile variant to
//! fall back to. A shape that does not fit after opt-in must therefore produce
//! a clear error naming the requirement, the device's real limit, and the
//! shape — never a launch crash and never a baked-in ceiling.
//!
//! Shared memory for one launch is
//!   `(BLOCK_M * head_dim_k + BLOCK_N * head_dim_k + BLOCK_N * head_dim_v) * 4`
//! = `512 * (2 * head_dim_k + head_dim_v)` bytes, independent of dtype.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test mla_smem_optin_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::traits::attention::mla::MlaOps;
use numr::autograd::Var;
#[cfg(feature = "f16")]
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Deterministic pseudo-random values, distinct per index and per seed.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.5 + (x * 2.3).cos() * 0.2
        })
        .collect()
}

/// Shared memory the SDPA kernel indexes, mirroring `sdpa_smem_size`.
fn smem_bytes(head_dim_k: usize, head_dim_v: usize) -> usize {
    (128 * head_dim_k + 128 * head_dim_k + 128 * head_dim_v) * 4
}

/// CPU-vs-CUDA tolerance for the SDPA forward. Both run the same math in
/// `f32`, but the CUDA kernel rescales its running max/sum once per 128-column
/// K tile while the CPU reference materializes the full score row, so the two
/// accumulate rounding in a different order. With `seq_len_k` in the low
/// hundreds that is a handful of rescalings per row; 1e-4 relative is a
/// generous bound with headroom, not a widened-to-pass number.
fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: non-finite value {a} at index {i} (kernel launched but computed garbage)"
        );
        let tol = 1e-5 + 1e-4 * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: {a} vs {e} (diff={}, tol={tol})",
            (a - e).abs()
        );
    }
}

/// Run MLA SDPA on both backends for one shape and return
/// `(cuda_result_or_error, cpu_reference)`.
fn run_both(
    b: usize,
    h: usize,
    s: usize,
    head_dim_k: usize,
    head_dim_v: usize,
    causal: bool,
) -> (boostr::error::Result<Vec<f32>>, Vec<f32>) {
    let (cpu_client, cpu_device) = cpu_setup();
    let (cuda_client, cuda_device) = cuda_setup();
    let scale = (head_dim_k as f64).sqrt().recip();

    let q_vals = values(b * h * s * head_dim_k, 0.11);
    let k_vals = values(b * h * s * head_dim_k, 0.37);
    let v_vals = values(b * h * s * head_dim_v, 0.73);

    let q_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&q_vals, &[b, h, s, head_dim_k], &cpu_device).unwrap(),
        false,
    );
    let k_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&k_vals, &[b, h, s, head_dim_k], &cpu_device).unwrap(),
        false,
    );
    let v_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&v_vals, &[b, h, s, head_dim_v], &cpu_device).unwrap(),
        false,
    );
    let expected = cpu_client
        .scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, scale, causal)
        .unwrap()
        .tensor()
        .to_vec::<f32>();

    let q_gpu = Var::<CudaRuntime>::new(
        Tensor::from_slice(&q_vals, &[b, h, s, head_dim_k], &cuda_device).unwrap(),
        false,
    );
    let k_gpu = Var::<CudaRuntime>::new(
        Tensor::from_slice(&k_vals, &[b, h, s, head_dim_k], &cuda_device).unwrap(),
        false,
    );
    let v_gpu = Var::<CudaRuntime>::new(
        Tensor::from_slice(&v_vals, &[b, h, s, head_dim_v], &cuda_device).unwrap(),
        false,
    );
    let actual = cuda_client
        .scaled_dot_product_attention(&q_gpu, &k_gpu, &v_gpu, scale, causal)
        .map(|out| {
            let v = out.tensor().to_vec::<f32>();
            cuda_client.synchronize();
            v
        });

    (actual, expected)
}

/// Assert an error is the graceful pre-launch refusal, not a driver launch
/// failure, and that it names the requirement, the device limit, and the shape.
fn assert_graceful_smem_error(
    err: &boostr::error::Error,
    smem: usize,
    head_dim_k: usize,
    head_dim_v: usize,
) {
    let msg = err.to_string();
    assert!(
        !msg.contains("launch failed") && !msg.contains("INVALID_VALUE"),
        "expected a pre-launch refusal, got a driver launch failure: {msg}"
    );
    assert!(
        msg.contains("SDPA shared memory requirement"),
        "error does not name the shared memory requirement: {msg}"
    );
    assert!(
        msg.contains(&smem.to_string()),
        "error does not report the actual requirement of {smem} bytes: {msg}"
    );
    assert!(
        msg.contains("device opt-in limit"),
        "error does not report the device's real limit: {msg}"
    );
    assert!(
        msg.contains(&format!("head_dim_k={head_dim_k}"))
            && msg.contains(&format!("head_dim_v={head_dim_v}")),
        "error does not name the shape: {msg}"
    );
}

/// 96KB of shared memory: over the 48KB default, under the opt-in limit of
/// every GPU this code path supports. Before the fix this failed at launch
/// with `CUDA_ERROR_INVALID_VALUE` because `set_smem_attribute` was never
/// called.
#[test]
fn head_dim_64_needs_the_smem_opt_in_and_matches_cpu() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED head_dim_64_needs_the_smem_opt_in_and_matches_cpu: CUDA runtime unavailable"
        );
        return;
    }
    assert_eq!(smem_bytes(64, 64), 98304);

    let (actual, expected) = run_both(1, 2, 192, 64, 64, false);
    let actual = actual.expect("head_dim_k=64, head_dim_v=64 (96KB smem) must launch after opt-in");
    assert_close(&actual, &expected, "mla_sdpa hdk=64 hdv=64");
}

/// Exactly 48KB: the boundary at which the opt-in path must engage. A request
/// of the full 48KB of *dynamic* shared memory is not reliably granted without
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, since the static
/// allocation shares that budget.
#[test]
fn head_dim_32_at_the_48kb_boundary_matches_cpu() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!("SKIPPED head_dim_32_at_the_48kb_boundary_matches_cpu: CUDA runtime unavailable");
        return;
    }
    assert_eq!(smem_bytes(32, 32), 49152);

    let (actual, expected) = run_both(1, 2, 192, 32, 32, true);
    let actual = actual.expect("head_dim_k=32, head_dim_v=32 (48KB smem) must launch after opt-in");
    assert_close(&actual, &expected, "mla_sdpa hdk=32 hdv=32 causal");
}

/// 128KB. Above the opt-in limit of some supported GPUs and within it on
/// others, so both outcomes are checked rather than skipped: either it
/// computes the right answer, or it refuses with the graceful error.
#[test]
fn head_dim_k_96_head_dim_v_64_either_computes_or_refuses_cleanly() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED head_dim_k_96_head_dim_v_64_either_computes_or_refuses_cleanly: \
             CUDA runtime unavailable"
        );
        return;
    }
    let smem = smem_bytes(96, 64);
    assert_eq!(smem, 131072);

    let (actual, expected) = run_both(1, 2, 192, 96, 64, false);
    match actual {
        Ok(out) => {
            eprintln!(
                "head_dim_k=96, head_dim_v=64 ({smem} bytes) fits this device; checking parity"
            );
            assert_close(&out, &expected, "mla_sdpa hdk=96 hdv=64");
        }
        Err(e) => {
            eprintln!("head_dim_k=96, head_dim_v=64 ({smem} bytes) exceeds this device: {e}");
            assert_graceful_smem_error(&e, smem, 96, 64);
        }
    }
}

/// DeepSeek-V2/V3-shaped MLA: `head_dim_k = head_dim + rope_head_dim = 192`,
/// `head_dim_v = 128`, i.e. 256KB with the single 128x128 tile `sdpa.cu`
/// compiles. That is above the opt-in limit of every shipping GPU, so it must
/// produce the graceful error naming the device's real limit — not a launch
/// crash, and not a refusal quoting a hardcoded ceiling.
#[test]
fn deepseek_shaped_mla_reports_the_device_limit_not_a_launch_crash() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED deepseek_shaped_mla_reports_the_device_limit_not_a_launch_crash: \
             CUDA runtime unavailable"
        );
        return;
    }
    let smem = smem_bytes(192, 128);
    assert_eq!(smem, 262144);

    let (actual, _expected) = run_both(1, 2, 128, 192, 128, true);
    let err = match actual {
        Ok(_) => panic!(
            "head_dim_k=192, head_dim_v=128 needs {smem} bytes of shared memory; \
             a device granting that would be new — re-check this test's assumption"
        ),
        Err(e) => e,
    };
    assert_graceful_smem_error(&err, smem, 192, 128);
    assert!(
        !err.to_string().contains("96 KB"),
        "error still quotes the removed hardcoded ceiling: {err}"
    );
}

/// `head_dim_v` indexes a fixed 256-element per-thread array in `sdpa.cu`.
/// A larger value must be refused, not left to corrupt the stack.
#[test]
fn head_dim_v_above_the_accumulator_length_is_refused() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED head_dim_v_above_the_accumulator_length_is_refused: CUDA runtime unavailable"
        );
        return;
    }
    let (cuda_client, cuda_device) = cuda_setup();
    let (b, h, s, head_dim_k, head_dim_v) = (1, 1, 8, 8, 320);
    let q = Var::<CudaRuntime>::new(
        Tensor::from_slice(
            &values(b * h * s * head_dim_k, 0.1),
            &[b, h, s, head_dim_k],
            &cuda_device,
        )
        .unwrap(),
        false,
    );
    let k = Var::<CudaRuntime>::new(
        Tensor::from_slice(
            &values(b * h * s * head_dim_k, 0.2),
            &[b, h, s, head_dim_k],
            &cuda_device,
        )
        .unwrap(),
        false,
    );
    let v = Var::<CudaRuntime>::new(
        Tensor::from_slice(
            &values(b * h * s * head_dim_v, 0.3),
            &[b, h, s, head_dim_v],
            &cuda_device,
        )
        .unwrap(),
        false,
    );

    let err = cuda_client
        .scaled_dot_product_attention(&q, &k, &v, 0.5, false)
        .expect_err(
            "head_dim_v=320 exceeds the kernel's 256-element accumulator and must be refused",
        );
    let msg = err.to_string();
    assert!(
        msg.contains("256") && msg.contains("320"),
        "error does not name the limit and the offending value: {msg}"
    );
}

// ===========================================================================
// F16 / BF16 coverage
//
// The element-size half of the defect was invisible to every test that
// existed: `sdpa.cu` stages Q/K/V as `float*` in all three kernels and
// converts on load, but the Rust helper multiplied by
// `dtype.size_in_bytes()`. F32 was therefore sized correctly and F16/BF16
// requested exactly HALF the bytes the kernel indexes, reading and writing
// past the end of the dynamic shared-memory allocation. Those launches
// succeeded — that is precisely why this survived. Returning `Ok` proves
// nothing here; only comparing values does.
//
// All half tests below use `seq_len = 192` so the grid spans two Q tiles
// (128 + 64) and the K loop spans two K/V tiles. The second K/V tile is what
// forces the `__half2float` / `__bfloat162float` store paths into `K_smem`
// and `V_smem` to run past the first block, which is where an undersized
// allocation is overrun.
// ===========================================================================

/// Machine epsilon of the storage format: `2^-10` for F16 (10 explicit
/// mantissa bits), `2^-7` for BF16 (7). Used to derive tolerances rather than
/// picking a number that happens to pass.
#[cfg(feature = "f16")]
fn half_eps(dtype: DType) -> f32 {
    match dtype {
        DType::F16 => 2f32.powi(-10),
        DType::BF16 => 2f32.powi(-7),
        other => panic!("half_eps called with non-half dtype {other:?}"),
    }
}

#[cfg(feature = "f16")]
fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Run MLA SDPA at `dtype` on CUDA and build the CPU reference from inputs
/// that were rounded to `dtype` and back to F32 first.
///
/// Rounding the fixtures once, up front, and feeding the SAME values to both
/// backends removes input quantization from the comparison entirely. What is
/// left is only what the kernel itself does: an `f32` online-softmax
/// accumulation out of `f32` shared-memory tiles, then one rounding of the
/// output to `dtype` on store. That is what makes a tight tolerance legitimate
/// here, and a tight tolerance is what makes shared-memory corruption
/// detectable — a loose "half precision is fuzzy" bound would hide it.
///
/// Returns `(cuda_result_as_f32_or_error, cpu_reference)`.
#[cfg(feature = "f16")]
fn run_both_typed(
    dtype: DType,
    b: usize,
    h: usize,
    s: usize,
    head_dim_k: usize,
    head_dim_v: usize,
    causal: bool,
) -> (boostr::error::Result<Vec<f32>>, Vec<f32>) {
    let (cpu_client, cpu_device) = cpu_setup();
    let (cuda_client, cuda_device) = cuda_setup();
    let scale = (head_dim_k as f64).sqrt().recip();

    let make = |vals: &[f32], shape: &[usize]| {
        let gpu = Tensor::<CudaRuntime>::from_slice(vals, shape, &cuda_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap_or_else(|e| panic!("cast fixture to {:?}: {e}", dtype));
        let rounded = gpu
            .to_dtype(DType::F32)
            .expect("cast rounded fixture back to F32")
            .to_vec::<f32>();
        (gpu, rounded)
    };

    let (q_gpu, q_round) = make(
        &values(b * h * s * head_dim_k, 0.11),
        &[b, h, s, head_dim_k],
    );
    let (k_gpu, k_round) = make(
        &values(b * h * s * head_dim_k, 0.37),
        &[b, h, s, head_dim_k],
    );
    let (v_gpu, v_round) = make(
        &values(b * h * s * head_dim_v, 0.73),
        &[b, h, s, head_dim_v],
    );

    let q_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&q_round, &[b, h, s, head_dim_k], &cpu_device).unwrap(),
        false,
    );
    let k_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&k_round, &[b, h, s, head_dim_k], &cpu_device).unwrap(),
        false,
    );
    let v_cpu = Var::<CpuRuntime>::new(
        Tensor::from_slice(&v_round, &[b, h, s, head_dim_v], &cpu_device).unwrap(),
        false,
    );
    let expected = cpu_client
        .scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, scale, causal)
        .unwrap()
        .tensor()
        .to_vec::<f32>();

    let actual = cuda_client
        .scaled_dot_product_attention(
            &Var::<CudaRuntime>::new(q_gpu, false),
            &Var::<CudaRuntime>::new(k_gpu, false),
            &Var::<CudaRuntime>::new(v_gpu, false),
            scale,
            causal,
        )
        .map(|out| {
            let v = out
                .tensor()
                .to_dtype(DType::F32)
                .expect("cast half output back to F32 for comparison")
                .to_vec::<f32>();
            cuda_client.synchronize();
            v
        });

    (actual, expected)
}

/// Tolerance for a half-dtype run against the F32 reference built from
/// identically rounded inputs. Two error terms, each bounded:
///
/// 1. The kernel stores the output once in `dtype`, so element `i` carries at
///    most `half_eps/2 * |o_i|` of rounding. `4 * half_eps` relative gives 8
///    ULP of headroom over that bound.
/// 2. The CUDA online softmax rescales its running max/sum once per 128-column
///    K tile while the CPU reference materializes the whole score row, so the
///    two `f32` accumulations round in a different order. With `seq_len_k` in
///    the low hundreds that is a couple of rescalings per row, bounded well
///    inside `1e3 * f32::EPSILON` relative to the accumulation's scale — hence
///    an absolute floor of `1e3 * f32::EPSILON * max|expected|`.
///
/// Concretely: F16 gives rtol 3.9e-3, BF16 3.1e-2. BF16 keeps only 8 mantissa
/// bits, so that is the format's real resolution, not a widened bound.
/// Shared-memory corruption produces wrong tiles, not sub-ULP drift, so it
/// lands far outside either.
#[cfg(feature = "f16")]
fn assert_close_half(actual: &[f32], expected: &[f32], dtype: DType, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    let max_abs = expected.iter().fold(0.0f32, |m, e| m.max(e.abs()));
    let floor = 1e3 * f32::EPSILON * max_abs;
    let rtol = 4.0 * half_eps(dtype);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: non-finite value {a} at index {i} — the kernel launched but \
             computed garbage, the signature of an undersized shared-memory tile"
        );
        let tol = floor + rtol * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: {a} vs {e} (diff={}, tol={tol}, rtol={rtol}, floor={floor})",
            (a - e).abs()
        );
    }
}

/// Pull the byte figure out of the graceful shared-memory refusal.
#[cfg(feature = "f16")]
fn reported_smem_bytes(err: &boostr::error::Error) -> usize {
    let msg = err.to_string();
    const HEAD: &str = "SDPA shared memory requirement (";
    let start = msg
        .find(HEAD)
        .unwrap_or_else(|| panic!("not a shared-memory refusal: {msg}"))
        + HEAD.len();
    let rest = &msg[start..];
    let end = rest
        .find(" bytes)")
        .unwrap_or_else(|| panic!("malformed shared-memory refusal: {msg}"));
    rest[..end]
        .parse()
        .unwrap_or_else(|e| panic!("unparseable byte count in {msg}: {e}"))
}

/// F16 at 48KB. Under the old sizing this asked for 24576 bytes and indexed
/// 49152, so both K/V tiles ran off the end of the allocation.
#[test]
fn f16_head_dim_32_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED f16_head_dim_32_matches_cpu: built without the `f16` feature; \
         run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!("SKIPPED f16_head_dim_32_matches_cpu: CUDA runtime unavailable");
            return;
        }
        assert_eq!(smem_bytes(32, 32), 49152);

        let (actual, expected) = run_both_typed(DType::F16, 1, 2, 192, 32, 32, false);
        let actual = actual.expect("f16 head_dim 32/32 (48KB smem) must launch after opt-in");
        assert_close_half(&actual, &expected, DType::F16, "mla_sdpa f16 hdk=32 hdv=32");
    }
}

/// F16 at 96KB, causal. Under the old sizing this asked for 49152 bytes and
/// indexed 98304 — a full 48KB overrun.
#[test]
fn f16_head_dim_64_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED f16_head_dim_64_matches_cpu: built without the `f16` feature; \
         run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!("SKIPPED f16_head_dim_64_matches_cpu: CUDA runtime unavailable");
            return;
        }
        assert_eq!(smem_bytes(64, 64), 98304);

        let (actual, expected) = run_both_typed(DType::F16, 1, 2, 192, 64, 64, true);
        let actual = actual.expect("f16 head_dim 64/64 (96KB smem) must launch after opt-in");
        assert_close_half(
            &actual,
            &expected,
            DType::F16,
            "mla_sdpa f16 hdk=64 hdv=64 causal",
        );
    }
}

/// BF16 at 48KB. Same overrun as the F16 case, through the
/// `__bfloat162float` load path.
#[test]
fn bf16_head_dim_32_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED bf16_head_dim_32_matches_cpu: built without the `f16` feature; \
         run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!("SKIPPED bf16_head_dim_32_matches_cpu: CUDA runtime unavailable");
            return;
        }
        assert_eq!(smem_bytes(32, 32), 49152);

        let (actual, expected) = run_both_typed(DType::BF16, 1, 2, 192, 32, 32, false);
        let actual = actual.expect("bf16 head_dim 32/32 (48KB smem) must launch after opt-in");
        assert_close_half(
            &actual,
            &expected,
            DType::BF16,
            "mla_sdpa bf16 hdk=32 hdv=32",
        );
    }
}

/// BF16 at 96KB, causal — the shape and dtype closest to a real MLA layer
/// that still fits the single 128x128 tile.
#[test]
fn bf16_head_dim_64_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED bf16_head_dim_64_matches_cpu: built without the `f16` feature; \
         run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!("SKIPPED bf16_head_dim_64_matches_cpu: CUDA runtime unavailable");
            return;
        }
        assert_eq!(smem_bytes(64, 64), 98304);

        let (actual, expected) = run_both_typed(DType::BF16, 1, 2, 192, 64, 64, true);
        let actual = actual.expect("bf16 head_dim 64/64 (96KB smem) must launch after opt-in");
        assert_close_half(
            &actual,
            &expected,
            DType::BF16,
            "mla_sdpa bf16 hdk=64 hdv=64 causal",
        );
    }
}

/// The direct regression guard for the element-size bug: the shared memory
/// requested for a given shape must be IDENTICAL across F32, F16 and BF16.
///
/// `sdpa.cu` stages every tile as `float` and converts F16/BF16 on load, so
/// the requirement depends only on the shape. Reintroducing
/// `* dtype.size_in_bytes()` makes the half dtypes ask for exactly half the
/// bytes the kernel indexes; this test fails the moment that happens.
///
/// `head_dim_k=96, head_dim_v=64` is 131072 bytes, above some supported
/// devices' opt-in limit and within others', so both outcomes are checked
/// rather than skipped. Either every dtype is refused with the same byte
/// figure, or every dtype computes the right answer — dtype-independence
/// holds on both paths. A split outcome is itself the bug: under the old
/// sizing F32 was refused at 131072 while F16/BF16 sailed through at 65536.
#[test]
fn requested_smem_is_identical_across_f32_f16_and_bf16() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED requested_smem_is_identical_across_f32_f16_and_bf16: built without the \
         `f16` feature; run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!(
                "SKIPPED requested_smem_is_identical_across_f32_f16_and_bf16: \
                 CUDA runtime unavailable"
            );
            return;
        }
        let smem = smem_bytes(96, 64);
        assert_eq!(smem, 131072);

        let dtypes = [DType::F32, DType::F16, DType::BF16];
        let mut outcomes = Vec::new();
        for dtype in dtypes {
            let (actual, expected) = run_both_typed(dtype, 1, 2, 192, 96, 64, false);
            match actual {
                Ok(out) => {
                    let label = format!("mla_sdpa {} hdk=96 hdv=64", dtype_label(dtype));
                    if dtype == DType::F32 {
                        assert_close(&out, &expected, &label);
                    } else {
                        assert_close_half(&out, &expected, dtype, &label);
                    }
                    outcomes.push((dtype, None));
                }
                Err(e) => {
                    assert_graceful_smem_error(&e, smem, 96, 64);
                    outcomes.push((dtype, Some(reported_smem_bytes(&e))));
                }
            }
        }

        let first = outcomes[0].1;
        for (dtype, reported) in &outcomes {
            assert_eq!(
                *reported,
                first,
                "SDPA shared memory is dtype-dependent: {} reported {:?} where {} reported {:?}. \
                 sdpa.cu stages every tile as `float` and converts F16/BF16 on load, so the \
                 requirement is a function of the SHAPE alone. A size scaled by \
                 `dtype.size_in_bytes()` asks for half the bytes the half kernels index and \
                 overruns the allocation — that was the original defect.",
                dtype_label(*dtype),
                reported,
                dtype_label(outcomes[0].0),
                first
            );
        }

        match first {
            Some(bytes) => {
                assert_eq!(
                    bytes, smem,
                    "refusal must report the shape-derived requirement of {smem} bytes"
                );
                eprintln!(
                    "head_dim_k=96, head_dim_v=64 exceeds this device for all three dtypes; \
                     all reported {bytes} bytes"
                );
            }
            None => eprintln!(
                "head_dim_k=96, head_dim_v=64 ({smem} bytes) fits this device for all three \
                 dtypes; parity checked for each"
            ),
        }
    }
}
