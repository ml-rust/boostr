//! Flash Attention v2 forward (`kernels/attention/flash_v2.cu`) at F32, F16
//! and BF16, head_dim 96/192/256, vs the CPU reference.
//!
//! # What this file covers
//!
//! `flash_v2.cu` is the general forward: it serves every head_dim no
//! dedicated kernel takes (96, 192, 256), sliding window at every head_dim,
//! and head_dim 32/64/128 on devices below sm_80. It maps a query row to a
//! group of lanes and emits two symbols per (head_dim, dtype): a four-warp
//! block (`flash_attention_fwd_{head_dim}_{dtype}`) and a two-warp one
//! (`flash_attention_fwd_{head_dim}_sm_{dtype}`). `flash_fwd_tile` in
//! `src/ops/cuda/attention/flash/flash_block_config.rs` picks between them by
//! device fill at F32, and always takes the four-warp symbol at F16/BF16.
//!
//! Query lengths per (head_dim, dtype):
//!
//! - `SEQ_Q_SHORT` (24 at 96/192, 11 at 256): one query block with a partial
//!   row tail.
//! - `SEQ_Q_MULTI` (150): several query blocks at every head_dim (64 rows per
//!   four-warp block at 96, 32 at 192/256, half that for two-warp) with a
//!   partial last block.
//! - `seq_q_filled` (F32 only, derived from the device's compute-unit count):
//!   enough query blocks to put the grid past the picker's fill threshold, so
//!   the F32 four-warp symbol runs. The short and multi F32 cases select the
//!   two-warp symbol on any device with more than a handful of compute units.
//!
//! `assert_expected_symbol` checks each case against the picker itself, so a
//! threshold change that moves a case onto the other symbol fails loudly
//! instead of silently dropping that symbol's coverage.
//!
//! `SEQ_K = 137` is not a multiple of the 16-key tile, so the K loop runs
//! several tiles and the last one is partial; with `SEQ_Q_MULTI > SEQ_K`
//! the causal mask also leaves whole query rows past every key, which the
//! kernel must treat as an exact no-op rather than exp(-inf - -inf).
//!
//! # Reaching `flash_v2.cu` deliberately
//!
//! `CudaClient::flash_attention_fwd` routes `head_dim` in {32, 64, 128} with
//! `window_size == 0` to the dedicated MQA/GQA kernel, so 128 is deliberately
//! absent here. 96/192/256 fall through to `flash_fwd::flash_attention_fwd_impl`
//! regardless of head counts. GQA (`num_kv_heads != num_heads`) is used
//! anyway, matching `flash_v2_bwd_halfprec_parity_cuda.rs`, so this also stays
//! off Flash v3 (MHA-only) independent of whether v3 dispatch is enabled.
//!
//! # Reference
//!
//! `CpuClient::flash_attention_fwd` (`src/ops/cpu/attention/flash.rs`),
//! always F32. It shares no indexing or tiling code with the CUDA kernel.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_v2_fwd_parity_cuda -- --nocapture

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::flash::flash_block_config::flash_fwd_tile_for_test;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Device, Runtime};
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

/// Key length: several 16-key tiles with a partial last one.
const SEQ_K: usize = 137;

/// Query length that spans several query blocks at every head_dim here,
/// with a partial last block, and exceeds `SEQ_K` so causal masking leaves
/// whole rows with no visible key.
const SEQ_Q_MULTI: usize = 150;

/// Deliberately loud: a silently skipped case would report green while
/// verifying nothing.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! FLASH_V2_FWD_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT running\n\
         !! the flash_v2 forward kernel at this dtype/shape. Treat it as\n\
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

/// Short query length per head_dim: one query block with a partial row tail
/// under both symbols (64 or 32 rows for four warps, 32 or 16 for two).
fn seq_q_short(head_dim: usize) -> usize {
    match head_dim {
        96 | 192 => 24,
        256 => 11,
        other => unimplemented!("seq_q_short: head_dim {other} is not covered by this file"),
    }
}

/// Mirrors `FLASH_FWD_F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT` in
/// `flash_block_config.rs`: F32 grids with fewer four-warp blocks per compute
/// unit than this take the two-warp symbol.
const F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT: usize = 8;

fn compute_units() -> usize {
    CudaDevice::new(0).profile().compute_units as usize
}

/// `(rows, small)` of the tile the picker selects for one case.
fn picked_tile(head_dim: usize, seq_len_q: usize, dtype: DType) -> (usize, bool) {
    let (rows, _threads, small) = flash_fwd_tile_for_test(
        head_dim,
        dtype.size_in_bytes(),
        seq_len_q,
        BATCH * NUM_HEADS,
        compute_units(),
    )
    .unwrap_or_else(|| panic!("flash_fwd_tile has no entry for head_dim={head_dim}"));
    (rows, small)
}

/// Rows per four-warp block at `head_dim`: with zero compute units no grid
/// counts as underfilled, so the picker returns the four-warp tile.
fn four_warp_rows(head_dim: usize) -> usize {
    let (rows, _threads, small) = flash_fwd_tile_for_test(head_dim, 4, 1, BATCH * NUM_HEADS, 0)
        .unwrap_or_else(|| panic!("flash_fwd_tile has no entry for head_dim={head_dim}"));
    assert!(
        !small,
        "test bug: zero compute units still picked the two-warp tile"
    );
    rows
}

/// F32 query length whose grid fills the device past the picker's threshold:
/// `BATCH * NUM_HEADS * blocks_per_head >= compute_units * threshold`, plus a
/// partial last block.
fn seq_q_filled(head_dim: usize) -> usize {
    let rows = four_warp_rows(head_dim);
    let blocks_per_head =
        (compute_units() * F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT).div_ceil(BATCH * NUM_HEADS);
    rows * blocks_per_head + 5
}

/// Fails loudly when the picker no longer selects the symbol a case exists
/// to cover. Returns false, after a loud skip, on a device with so few
/// compute units that the short and multi F32 grids already fill it; the
/// parity check still runs there.
fn assert_expected_symbol(
    head_dim: usize,
    seq_len_q: usize,
    dtype: DType,
    want_small: bool,
    label: &str,
) -> bool {
    let (rows, small) = picked_tile(head_dim, seq_len_q, dtype);
    if want_small && seq_len_q <= SEQ_Q_MULTI {
        let blocks = BATCH * NUM_HEADS * seq_len_q.div_ceil(four_warp_rows(head_dim));
        if blocks >= compute_units() * F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT {
            loud_skip(
                label,
                "this device has too few compute units for the F32 short/multi grid to                  select the two-warp symbol; parity still runs on the four-warp one",
            );
            return false;
        }
    }
    assert_eq!(
        small,
        want_small,
        "test bug: head_dim={head_dim} seq_len_q={seq_len_q} dtype={dtype:?} selects the          {} symbol, but this case exists to cover the {} one — re-derive the shape from          `flash_fwd_tile`, do not drop the case",
        if small { "two-warp" } else { "four-warp" },
        if want_small { "two-warp" } else { "four-warp" },
    );
    if seq_len_q > seq_q_short(head_dim) {
        assert!(
            seq_len_q > rows && !seq_len_q.is_multiple_of(rows),
            "test bug: seq_len_q={seq_len_q} no longer spans several query blocks of              {rows} rows with a partial tail at head_dim={head_dim}"
        );
    }
    true
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
/// `u_f16 = 2^-11 ≈ 4.88e-4`, `u_bf16 = 2^-8 ≈ 3.91e-3`. The F32 bound
/// covers the kernel's fast-math `__expf` and its G-way split dot products
/// against the reference's serial sums.
fn flash_fwd_tol(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::F32 => (1e-4, 1e-3),
        DType::F16 => (4e-3, 3e-2),
        DType::BF16 => (2e-2, 1.2e-1),
        other => unimplemented!("flash_fwd_tol: unsupported dtype {other:?}"),
    }
}

/// Compares against the reference, printing an always-on
/// `FLASH_FWD_DIAG` line (pass or fail, flat `key=value` pairs) so the
/// measured deviation is visible even on a green run.
#[allow(clippy::too_many_arguments)]
fn assert_fwd_diff(
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
             fully masked row or tile that was not treated as a no-op produces exactly this"
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
        "FLASH_FWD_DIAG dtype={dtype:?} head_dim={head_dim} seq_len_q={seq_len_q} \
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

/// Runs the CPU F32 reference and the CUDA kernel at `dtype` for one shape,
/// and checks output values and the LSE. `want_small` names the symbol the
/// case exists to cover.
fn assert_fwd_parity(
    head_dim: usize,
    seq_len_q: usize,
    dtype: DType,
    want_small: bool,
    label: &str,
) {
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
    assert_expected_symbol(head_dim, seq_len_q, dtype, want_small, label);

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

    // CUDA at `dtype`, through the SAME public dispatch a real caller uses:
    // GQA plus head_dim outside {32, 64, 128} is what lands the call in
    // `flash_fwd::flash_attention_fwd_impl`.
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

    let (atol, rtol) = flash_fwd_tol(dtype);
    assert_fwd_diff(
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
    // VALUE still depends on the Q/K rounding and the kernel's own softmax
    // accumulation, so it is checked independently of the output.
    assert_fwd_diff(
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

#[derive(Clone, Copy)]
enum QueryLen {
    Short,
    Multi,
    Filled,
}

macro_rules! fwd_parity_cases {
    ($($name:ident: $head_dim:expr, $dtype:expr, $len:expr;)*) => {
        $(
            #[test]
            fn $name() {
                let seq_len_q = match $len {
                    QueryLen::Short => seq_q_short($head_dim),
                    QueryLen::Multi => SEQ_Q_MULTI,
                    QueryLen::Filled => seq_q_filled($head_dim),
                };
                // F32 takes the two-warp symbol until the grid fills the
                // device; half precision always takes the four-warp one.
                let want_small = $dtype == DType::F32 && !matches!($len, QueryLen::Filled);
                assert_fwd_parity(
                    $head_dim,
                    seq_len_q,
                    $dtype,
                    want_small,
                    concat!("flash_v2_fwd ", stringify!($name)),
                );
            }
        )*
    };
}

fwd_parity_cases! {
    hd96_f32_short: 96, DType::F32, QueryLen::Short;
    hd96_f32_multi: 96, DType::F32, QueryLen::Multi;
    hd96_f32_filled: 96, DType::F32, QueryLen::Filled;
    hd96_f16_short: 96, DType::F16, QueryLen::Short;
    hd96_f16_multi: 96, DType::F16, QueryLen::Multi;
    hd96_bf16_short: 96, DType::BF16, QueryLen::Short;
    hd96_bf16_multi: 96, DType::BF16, QueryLen::Multi;
    hd192_f32_short: 192, DType::F32, QueryLen::Short;
    hd192_f32_multi: 192, DType::F32, QueryLen::Multi;
    hd192_f32_filled: 192, DType::F32, QueryLen::Filled;
    hd192_f16_short: 192, DType::F16, QueryLen::Short;
    hd192_f16_multi: 192, DType::F16, QueryLen::Multi;
    hd192_bf16_short: 192, DType::BF16, QueryLen::Short;
    hd192_bf16_multi: 192, DType::BF16, QueryLen::Multi;
    hd256_f32_short: 256, DType::F32, QueryLen::Short;
    hd256_f32_multi: 256, DType::F32, QueryLen::Multi;
    hd256_f32_filled: 256, DType::F32, QueryLen::Filled;
    hd256_f16_short: 256, DType::F16, QueryLen::Short;
    hd256_f16_multi: 256, DType::F16, QueryLen::Multi;
    hd256_bf16_short: 256, DType::BF16, QueryLen::Short;
    hd256_bf16_multi: 256, DType::BF16, QueryLen::Multi;
}
