//! Regression tests for the MLA SDPA shared-memory opt-in and tile selection
//! in `src/ops/cuda/attention/mla.rs` + `mla_block_config.rs`.
//!
//! Three defects lived in that launcher:
//!
//! 1. It set `LaunchConfig::shared_mem_bytes` above the 48KB CUDA default
//!    without ever calling `set_smem_attribute`, so every configuration
//!    needing more than the default limit failed at launch with
//!    `CUDA_ERROR_INVALID_VALUE` ("invalid argument") on a device that would
//!    have granted the request. Every other smem-using attention launcher in
//!    the crate opts in; this one did not.
//! 2. It sized shared memory with the *input* dtype's element size. Every
//!    kernel in `sdpa.cu` stages Q/K/V as `float` and converts F16/BF16 on
//!    load, so an F16/BF16 launch requested half the bytes the kernel indexes
//!    and read/wrote past the end of the allocation.
//! 3. `sdpa.cu` hardcoded `BLOCK_M = BLOCK_N = 128` and exported exactly three
//!    kernels, so there was no smaller tile to fall back to. A DeepSeek-V2/V3
//!    -shaped MLA (`head_dim_k = head_dim + rope_head_dim = 192`,
//!    `head_dim_v = 128`) needs 256KB at that tile and was refused outright on
//!    every device.
//!
//! The fix: `sdpa_impl<T, BLOCK_M, BLOCK_N>` in `sdpa.cu`, instantiated at
//! `(128, 128)` under the unsuffixed names and at `(64, 32)` under
//! `sdpa_{f32,f16,bf16}_small`, plus `mla_block_config::block_config`, which
//! picks the largest tile that fits `device_max_smem()` and returns the
//! matching kernel name, `block_dim.x` and smem request together.
//!
//! Shared memory for one launch is
//!   `(BLOCK_M * head_dim_k + BLOCK_N * head_dim_k + BLOCK_N * head_dim_v) * 4`
//! bytes, independent of dtype — every tile is staged as `float`.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test mla_smem_optin_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::mla_block_config::mla_tile_for_test;
use boostr::ops::traits::attention::mla::MlaOps;
use numr::autograd::Var;
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

/// `(BLOCK_M, BLOCK_N)` of the two tiles `sdpa.cu` instantiates.
const LARGE_TILE: (usize, usize) = (128, 128);
const SMALL_TILE: (usize, usize) = (64, 32);

/// Shared memory the SDPA kernel indexes for a tile, mirroring
/// `mla_block_config::sdpa_smem_size`. Dtype-independent by construction.
fn smem_bytes(tile: (usize, usize), head_dim_k: usize, head_dim_v: usize) -> usize {
    let (block_m, block_n) = tile;
    (block_m * head_dim_k + block_n * head_dim_k + block_n * head_dim_v) * 4
}

/// Base kernel name for a dtype, without the tile suffix.
fn base_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "sdpa_f32",
        DType::F16 => "sdpa_f16",
        DType::BF16 => "sdpa_bf16",
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Resolve the tile this device picks for a shape and assert the whole
/// resolution is self-consistent: the kernel name's suffix, `block_m`,
/// `block_n` and the smem figure must all describe the SAME tile, and that
/// tile must be the largest one that fits. A name that disagrees with the
/// tile it was selected for is the exact failure `TileVariant::suffix` exists
/// to prevent — it produced a symbol that was never compiled.
///
/// Returns `(kernel_name, block_m, block_n, smem, device_max_smem)`.
fn resolved_tile(
    dtype: DType,
    head_dim_k: usize,
    head_dim_v: usize,
) -> (String, usize, usize, usize, usize) {
    let (name, block_m, block_n, smem, max_smem) = mla_tile_for_test(dtype, head_dim_k, head_dim_v)
        .unwrap_or_else(|e| panic!("no tile for {head_dim_k}/{head_dim_v} {dtype:?}: {e}"));

    let expected_tile = if smem_bytes(LARGE_TILE, head_dim_k, head_dim_v) <= max_smem {
        LARGE_TILE
    } else {
        SMALL_TILE
    };
    let expected_suffix = if expected_tile == LARGE_TILE {
        ""
    } else {
        "_small"
    };
    assert_eq!(
        name,
        format!("{}{}", base_name(dtype), expected_suffix),
        "resolved kernel name does not match the largest tile that fits \
         (device limit {max_smem} bytes)"
    );
    assert_eq!(
        (block_m, block_n),
        expected_tile,
        "resolved block dims do not match the kernel name {name}"
    );
    assert_eq!(
        smem,
        smem_bytes(expected_tile, head_dim_k, head_dim_v),
        "resolved smem does not match the resolved tile {expected_tile:?}"
    );
    assert!(
        smem <= max_smem,
        "resolved tile needs {smem} bytes, above this device's {max_smem} byte limit"
    );
    (name, block_m, block_n, smem, max_smem)
}

/// CPU-vs-CUDA tolerance for the SDPA forward. Both run the same math in
/// `f32`, but the CUDA kernel rescales its running max/sum once per K tile
/// (`BLOCK_N` columns) while the CPU reference materializes the full score row,
/// so the two
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
    assert_eq!(smem_bytes(LARGE_TILE, 64, 64), 98304);
    resolved_tile(DType::F32, 64, 64);

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
    assert_eq!(smem_bytes(LARGE_TILE, 32, 32), 49152);

    let (actual, expected) = run_both(1, 2, 192, 32, 32, true);
    let actual = actual.expect("head_dim_k=32, head_dim_v=32 (48KB smem) must launch after opt-in");
    assert_close(&actual, &expected, "mla_sdpa hdk=32 hdv=32 causal");
}

/// 131072 bytes at the large tile — above the opt-in limit of some supported
/// GPUs and within it on others. Either way it must now COMPUTE: the small
/// tile needs only 45056 bytes for this shape, which fits every device the
/// opt-in path targets. Before the tile fallback existed, half the supported
/// devices refused this shape outright.
#[test]
fn head_dim_k_96_head_dim_v_64_computes_on_whichever_tile_fits() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED head_dim_k_96_head_dim_v_64_computes_on_whichever_tile_fits: \
             CUDA runtime unavailable"
        );
        return;
    }
    assert_eq!(smem_bytes(LARGE_TILE, 96, 64), 131072);
    assert_eq!(smem_bytes(SMALL_TILE, 96, 64), 45056);
    let (name, _, _, smem, _) = resolved_tile(DType::F32, 96, 64);
    eprintln!("head_dim_k=96, head_dim_v=64 resolved to {name} ({smem} bytes)");

    let (actual, expected) = run_both(1, 2, 192, 96, 64, false);
    let actual = actual.expect("head_dim_k=96, head_dim_v=64 must run on one of the two tiles");
    assert_close(&actual, &expected, "mla_sdpa hdk=96 hdv=64");
}

/// DeepSeek-V2/V3-shaped MLA: `head_dim_k = head_dim + rope_head_dim = 192`,
/// `head_dim_v = 128`. The large 128x128 tile needs 262144 bytes, above the
/// opt-in limit of every shipping GPU, so before the fallback existed this
/// shape could not run anywhere. The small 64x32 tile needs 90112 bytes and
/// fits a ~99KB device, so it must now COMPUTE — and compute the right
/// answer, which is why this checks values against the CPU reference rather
/// than settling for `Ok`.
#[test]
fn deepseek_shaped_mla_runs_on_the_small_tile_and_matches_cpu() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED deepseek_shaped_mla_runs_on_the_small_tile_and_matches_cpu: \
             CUDA runtime unavailable"
        );
        return;
    }
    assert_eq!(smem_bytes(LARGE_TILE, 192, 128), 262144);
    assert_eq!(smem_bytes(SMALL_TILE, 192, 128), 90112);

    let (name, block_m, block_n, smem, max_smem) = resolved_tile(DType::F32, 192, 128);
    assert!(
        max_smem < 262144,
        "this device grants {max_smem} bytes, enough for the large tile at 192/128; \
         a GPU that big would be new — re-check this test's assumption"
    );
    assert_eq!(name, "sdpa_f32_small");
    assert_eq!((block_m, block_n), SMALL_TILE);
    assert_eq!(smem, 90112);

    let (actual, expected) = run_both(1, 2, 128, 192, 128, true);
    let actual = actual
        .expect("DeepSeek-shaped MLA (192/128) must run on the small tile after the fallback");
    assert_close(&actual, &expected, "mla_sdpa hdk=192 hdv=128 causal");
}

/// The fallback must not have become unconditional. A shape whose LARGE tile
/// fits this device must resolve to the unsuffixed kernel and `BLOCK_M=128`,
/// and a shape whose large tile does not fit must resolve to `_small` and
/// `BLOCK_M=64` — asserted on the resolved tile, not inferred from a launch
/// succeeding. `resolved_tile` checks name, block dims and smem describe one
/// and the same tile, which is what keeps `block_dim.x` in step with the
/// kernel `sdpa.cu` compiled.
#[test]
fn tile_selection_tracks_the_device_limit_in_both_directions() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED tile_selection_tracks_the_device_limit_in_both_directions: \
             CUDA runtime unavailable"
        );
        return;
    }
    // 32/32 needs 49152 bytes large — inside every opt-in limit, so the large
    // tile must be selected here on any device this path supports.
    let (name, block_m, block_n, _, _) = resolved_tile(DType::F32, 32, 32);
    assert_eq!(name, "sdpa_f32", "large tile must still win when it fits");
    assert_eq!((block_m, block_n), LARGE_TILE);

    // 192/128 needs 262144 bytes large — outside every opt-in limit.
    let (name, block_m, block_n, _, _) = resolved_tile(DType::F32, 192, 128);
    assert_eq!(
        name, "sdpa_f32_small",
        "small tile must take over when the large one does not fit"
    );
    assert_eq!((block_m, block_n), SMALL_TILE);
}

/// The refusal path still has to exist and still has to be graceful. The
/// small tile pushes the ceiling far out, but not to infinity:
/// `head_dim_k=4096, head_dim_v=256` needs 1605632 bytes even at 64x32, well
/// past any device. That must be a pre-launch error naming the requirement,
/// the device's real limit and the shape — never a driver launch failure.
#[test]
fn a_shape_too_large_for_even_the_small_tile_is_refused_gracefully() {
    let _guard = cuda_lock();
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED a_shape_too_large_for_even_the_small_tile_is_refused_gracefully: \
             CUDA runtime unavailable"
        );
        return;
    }
    let (head_dim_k, head_dim_v) = (4096, 256);
    let smem = smem_bytes(SMALL_TILE, head_dim_k, head_dim_v);
    assert_eq!(smem, 1605632);
    assert!(mla_tile_for_test(DType::F32, head_dim_k, head_dim_v).is_err());

    let (cuda_client, cuda_device) = cuda_setup();
    let (b, h, s) = (1, 1, 8);
    let make = |seed: f32, dim: usize| {
        Var::<CudaRuntime>::new(
            Tensor::from_slice(
                &values(b * h * s * dim, seed),
                &[b, h, s, dim],
                &cuda_device,
            )
            .unwrap(),
            false,
        )
    };
    let err = cuda_client
        .scaled_dot_product_attention(
            &make(0.1, head_dim_k),
            &make(0.2, head_dim_k),
            &make(0.3, head_dim_v),
            0.5,
            false,
        )
        .expect_err("a shape needing 1.5MB of shared memory must be refused");
    assert_graceful_smem_error(&err, smem, head_dim_k, head_dim_v);
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
// All half tests below use `seq_len = 192` so the grid spans more than one Q
// tile and the K loop spans more than one K/V tile. The second K/V tile is what
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
/// 2. The CUDA online softmax rescales its running max/sum once per K tile
///    (`BLOCK_N` columns) while the CPU reference materializes the whole score row, so the
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
        assert_eq!(smem_bytes(LARGE_TILE, 32, 32), 49152);

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
        assert_eq!(smem_bytes(LARGE_TILE, 64, 64), 98304);

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
        assert_eq!(smem_bytes(LARGE_TILE, 32, 32), 49152);

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
        assert_eq!(smem_bytes(LARGE_TILE, 64, 64), 98304);

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

/// The direct regression guard for the element-size bug: the tile resolved
/// for a given shape — its kernel, its block dims and its shared-memory
/// request — must be IDENTICAL across F32, F16 and BF16 apart from the dtype
/// in the kernel name.
///
/// `sdpa.cu` stages every tile as `float` and converts F16/BF16 on load, so
/// the requirement depends only on the SHAPE and the TILE. Reintroducing
/// `* dtype.size_in_bytes()` makes the half dtypes ask for exactly half the
/// bytes the kernel indexes; this test fails the moment that happens — and it
/// would also let a half dtype resolve to a larger tile than F32 for the same
/// shape, which is the same bug wearing the tile selector's clothes.
#[test]
fn resolved_tile_is_identical_across_f32_f16_and_bf16() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED resolved_tile_is_identical_across_f32_f16_and_bf16: built without the \
         `f16` feature; run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!(
                "SKIPPED resolved_tile_is_identical_across_f32_f16_and_bf16: \
                 CUDA runtime unavailable"
            );
            return;
        }

        // Both a shape that takes the large tile and one that takes the small
        // one, so dtype-independence is checked on each branch of the selector.
        for (head_dim_k, head_dim_v) in [(64usize, 64usize), (192, 128)] {
            let mut seen: Option<(usize, usize, usize)> = None;
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let (name, block_m, block_n, smem, _) =
                    resolved_tile(dtype, head_dim_k, head_dim_v);
                assert!(
                    name.starts_with(base_name(dtype)),
                    "{name} is not the kernel for {}",
                    dtype_label(dtype)
                );
                match seen {
                    None => seen = Some((block_m, block_n, smem)),
                    Some(first) => assert_eq!(
                        (block_m, block_n, smem),
                        first,
                        "SDPA tile selection is dtype-dependent at {head_dim_k}/{head_dim_v}: \
                         {} resolved {:?} where the first dtype resolved {:?}. sdpa.cu stages \
                         every tile as `float` and converts F16/BF16 on load, so the \
                         requirement is a function of the SHAPE and the TILE alone. A size \
                         scaled by `dtype.size_in_bytes()` asks for half the bytes the half \
                         kernels index and overruns the allocation — that was the original \
                         defect.",
                        dtype_label(dtype),
                        (block_m, block_n, smem),
                        first
                    ),
                }
            }
        }
    }
}

/// F16 on the SMALL tile, DeepSeek-shaped. The small tile is new code, and
/// F32-only coverage of it would repeat exactly the gap that let the
/// element-size defect survive: an undersized or mis-strided `_small`
/// allocation still launches, and only comparing values catches it.
#[test]
fn f16_deepseek_shaped_small_tile_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED f16_deepseek_shaped_small_tile_matches_cpu: built without the `f16` \
         feature; run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!(
                "SKIPPED f16_deepseek_shaped_small_tile_matches_cpu: CUDA runtime unavailable"
            );
            return;
        }
        let (name, block_m, block_n, smem, _) = resolved_tile(DType::F16, 192, 128);
        assert_eq!(name, "sdpa_f16_small");
        assert_eq!((block_m, block_n), SMALL_TILE);
        assert_eq!(smem, 90112);

        let (actual, expected) = run_both_typed(DType::F16, 1, 2, 192, 192, 128, false);
        let actual = actual.expect("f16 192/128 must run on the small tile");
        assert_close_half(
            &actual,
            &expected,
            DType::F16,
            "mla_sdpa f16 small hdk=192 hdv=128",
        );
    }
}

/// BF16 on the SMALL tile, DeepSeek-shaped and causal. Same reason as the
/// F16 case, through the `__bfloat162float` load path — and causal so the
/// small tile's masking runs against a `BLOCK_N` of 32 rather than 128.
#[test]
fn bf16_deepseek_shaped_small_tile_matches_cpu() {
    #[cfg(not(feature = "f16"))]
    eprintln!(
        "SKIPPED bf16_deepseek_shaped_small_tile_matches_cpu: built without the `f16` \
         feature; run with --features cuda,f16"
    );
    #[cfg(feature = "f16")]
    {
        let _guard = cuda_lock();
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!(
                "SKIPPED bf16_deepseek_shaped_small_tile_matches_cpu: CUDA runtime unavailable"
            );
            return;
        }
        let (name, block_m, block_n, smem, _) = resolved_tile(DType::BF16, 192, 128);
        assert_eq!(name, "sdpa_bf16_small");
        assert_eq!((block_m, block_n), SMALL_TILE);
        assert_eq!(smem, 90112);

        let (actual, expected) = run_both_typed(DType::BF16, 1, 2, 192, 192, 128, true);
        let actual = actual.expect("bf16 192/128 must run on the small tile");
        assert_close_half(
            &actual,
            &expected,
            DType::BF16,
            "mla_sdpa bf16 small hdk=192 hdv=128 causal",
        );
    }
}
