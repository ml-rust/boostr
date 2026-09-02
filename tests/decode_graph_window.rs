//! Sliding-window tests for the CUDA graph decode attention kernel.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test decode_graph_window
//!
//! The whole file is gated on the `cuda` feature so it compiles away on
//! CPU-only builds.
//!
//! Reference choice: decode is single-token, so the query sits at absolute
//! position `seq_len_k - 1` and a window keeps exactly the contiguous key suffix
//! `j >= seq_len_k - window`. The reference therefore runs the trusted non-graph
//! `flash_attention_fwd` decode over that narrowed suffix with `window_size = 0`,
//! which is mathematically the windowed result.
//!
//! It does NOT pass the window to `flash_attention_fwd` directly: a narrowed
//! key range needs no masking at all, so the reference stays independent of the
//! mask builders it is checking. Those builders
//! (`ops/impl_generic/attention/flash_standard.rs::build_attention_mask` and
//! `kernels/attention/flash_v2.cu`) now read the query index as the absolute
//! position `seq_len_k - seq_len_q + i`; `tests/window_mask_decode.rs` checks
//! them against the same suffix reference on CPU.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::flash::decode_attention_graph_fwd;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use numr::runtime::Runtime;
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

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

const BATCH: usize = 1;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
/// Full KV-cache capacity — larger than `SEQ_LEN_K`, as in graph mode.
const CAPACITY: usize = 48;
/// Live cache length for the decode step.
const SEQ_LEN_K: usize = 20;
/// Window shorter than the live cache, so it must change the result.
const WINDOW: usize = 6;

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Q `[B, H, 1, D]`, plus K/V caches `[B, KVH, CAPACITY, D]` whose slots at or
/// beyond `SEQ_LEN_K` hold large garbage — any kernel that reads past the live
/// length blows the comparison apart instead of drifting quietly.
fn inputs(
    device: &CudaDevice,
) -> (
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
) {
    let q = Tensor::<CudaRuntime>::from_slice(
        &values(BATCH * NUM_HEADS * HEAD_DIM, 0.3),
        &[BATCH, NUM_HEADS, 1, HEAD_DIM],
        device,
    )
    .unwrap();

    let cache_len = BATCH * NUM_KV_HEADS * CAPACITY * HEAD_DIM;
    let mut k_data = values(cache_len, 1.1);
    let mut v_data = values(cache_len, 2.7);
    for kv_h in 0..NUM_KV_HEADS {
        for pos in SEQ_LEN_K..CAPACITY {
            for d in 0..HEAD_DIM {
                let idx = (kv_h * CAPACITY + pos) * HEAD_DIM + d;
                k_data[idx] = 50.0;
                v_data[idx] = -50.0;
            }
        }
    }

    let shape = [BATCH, NUM_KV_HEADS, CAPACITY, HEAD_DIM];
    let k = Tensor::<CudaRuntime>::from_slice(&k_data, &shape, device).unwrap();
    let v = Tensor::<CudaRuntime>::from_slice(&v_data, &shape, device).unwrap();
    (q, k, v)
}

/// Run the graph decode kernel with `window_size`, `seq_len_k` on the device.
fn graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    window_size: usize,
) -> Vec<f32> {
    let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[SEQ_LEN_K as i32], &[1], device).unwrap();
    let (out, _lse) = decode_attention_graph_fwd(
        client,
        q,
        k,
        v,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        seq_len_k.ptr(),
        CAPACITY,
        window_size,
    )
    .expect("graph decode attention failed");
    out.to_vec::<f32>()
}

/// Trusted reference: unwindowed non-graph decode over keys `[start, SEQ_LEN_K)`.
fn reference_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    start: usize,
) -> Vec<f32> {
    let len = SEQ_LEN_K - start;
    let k_slice = k.narrow(2, start, len).unwrap().contiguous().unwrap();
    let v_slice = v.narrow(2, start, len).unwrap().contiguous().unwrap();
    let (out, _lse) = client
        .flash_attention_fwd(
            q,
            &k_slice,
            &v_slice,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            false,
            0,
            None,
        )
        .expect("reference flash_attention_fwd failed");
    out.to_vec::<f32>()
}

fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let threshold = tol + tol * y.abs();
        assert!(
            diff <= threshold,
            "{label} at index {i}: {x} vs {y} (diff={diff}, tol={threshold})"
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The test that matters: windowed graph decode == unwindowed decode over the
/// key suffix the window selects.
#[test]
fn graph_decode_window_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_decode_window_matches_non_graph_reference");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = inputs(&device);

    let windowed = graph_decode(&client, &device, &q, &k, &v, WINDOW);
    let reference = reference_decode(&client, &q, &k, &v, SEQ_LEN_K - WINDOW);

    assert_close(&windowed, &reference, "graph decode window vs suffix", 1e-5);
}

/// Regression guard: `window_size = 0` still attends the whole live cache.
#[test]
fn graph_decode_zero_window_is_unchanged() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_decode_zero_window_is_unchanged");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = inputs(&device);

    let unwindowed = graph_decode(&client, &device, &q, &k, &v, 0);
    let reference = reference_decode(&client, &q, &k, &v, 0);

    assert_close(&unwindowed, &reference, "graph decode window 0", 1e-5);
}

/// Without this, the parity test above could pass with the window ignored.
#[test]
fn graph_decode_window_changes_the_result() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_decode_window_changes_the_result");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = inputs(&device);

    let windowed = graph_decode(&client, &device, &q, &k, &v, WINDOW);
    let unwindowed = graph_decode(&client, &device, &q, &k, &v, 0);

    let diff = max_abs_diff(&windowed, &unwindowed);
    assert!(
        diff > 1e-3,
        "window {WINDOW} over {SEQ_LEN_K} keys left the result unchanged (max diff {diff})"
    );
}

/// A window at least as long as the live cache masks nothing, so it equals
/// `window_size = 0` — the boundary of the inclusive-window rule.
#[test]
fn graph_decode_window_wider_than_cache_equals_zero_window() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping graph_decode_window_wider_than_cache_equals_zero_window"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = inputs(&device);

    let wide = graph_decode(&client, &device, &q, &k, &v, SEQ_LEN_K);
    let unwindowed = graph_decode(&client, &device, &q, &k, &v, 0);

    assert_close(&wide, &unwindowed, "graph decode wide window", 1e-6);
}

// ---------------------------------------------------------------------------
// Split-KV graph path (`decode_attention_graph_fwd` with `splits > 1`).
//
// `decode_split_count(device_index, base_blocks, kv_len)` in
// `src/ops/cuda/attention/decode_split.rs` returns more than one once BOTH:
//   - `base_blocks < compute_units * DECODE_BLOCKS_PER_UNIT (8)` — the
//     whole-sequence grid underfills the device, and
//   - `kv_len / DECODE_MIN_CHUNK (32) >= 2` — the sequence has at least two
//     minimum-sized chunks to cut.
// The tests above all use `CAPACITY = 48`, so `48 / 32 = 1 < 2` and they never
// leave the whole-sequence kernel. The tests below use a larger capacity, big
// enough to split, without touching `CAPACITY`, `SEQ_LEN_K`, or `WINDOW` above
// so the whole-sequence coverage stays intact.
// ---------------------------------------------------------------------------

/// KV-cache capacity for the split-KV graph tests.
///
/// The graph path sizes its split count from this static capacity, not the
/// live `seq_len_k` (see the comment on `decode_attention_graph_fwd`), so
/// every test below shares it and only varies the live length.
///
/// `base_blocks = BATCH * NUM_HEADS = 4`. Any real CUDA device profile
/// reports `compute_units >= 1`, so `target_blocks = compute_units * 8 >= 8 >
/// 4` and the whole-sequence grid is always underfilled. `SPLIT_CAPACITY / 32
/// = 64 >= 2`, so the minimum-chunk floor is cleared with a wide margin —
/// comfortably enough that the split path fires even on a device with very
/// few compute units, while the resulting split count still lands inside
/// `DECODE_MAX_SPLITS (32)`.
const SPLIT_CAPACITY: usize = 2048;
/// Live length for the full-cache split test: equal to `SPLIT_CAPACITY`, so
/// every split slice is entirely populated.
const SPLIT_SEQ_LEN_K_FULL: usize = SPLIT_CAPACITY;
/// Live length for the mostly-empty split test: far below `SPLIT_CAPACITY`,
/// so most of the statically-sized split grid falls past the live cache and
/// must be skipped by the split kernel's `begin >= end` guard, then dropped
/// by the combine kernel's `l <= 0` guard.
const SPLIT_SEQ_LEN_K_EMPTY: usize = 96;
/// Live length for the windowed split test.
const SPLIT_SEQ_LEN_K_WINDOW: usize = 300;
/// Window narrower than `SPLIT_SEQ_LEN_K_WINDOW`, so it must change the
/// result and still interact correctly with the split slicing.
const SPLIT_WINDOW: usize = 40;

/// Q `[B, NUM_HEADS, 1, D]`, plus K/V caches `[B, KVH, capacity, D]`
/// whose slots at or beyond `live_len` hold large garbage. Mirrors `inputs()`
/// above but takes `capacity`/`live_len` as parameters so the split tests can
/// use a capacity large enough to make `decode_split_count` return more than
/// one, without disturbing the whole-sequence tests' fixed constants.
fn split_inputs(
    device: &CudaDevice,
    capacity: usize,
    live_len: usize,
) -> (
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
) {
    let q = Tensor::<CudaRuntime>::from_slice(
        &values(BATCH * NUM_HEADS * HEAD_DIM, 0.3),
        &[BATCH, NUM_HEADS, 1, HEAD_DIM],
        device,
    )
    .unwrap();

    let cache_len = BATCH * NUM_KV_HEADS * capacity * HEAD_DIM;
    let mut k_data = values(cache_len, 1.1);
    let mut v_data = values(cache_len, 2.7);
    for kv_h in 0..NUM_KV_HEADS {
        for pos in live_len..capacity {
            for d in 0..HEAD_DIM {
                let idx = (kv_h * capacity + pos) * HEAD_DIM + d;
                k_data[idx] = 50.0;
                v_data[idx] = -50.0;
            }
        }
    }

    let shape = [BATCH, NUM_KV_HEADS, capacity, HEAD_DIM];
    let k = Tensor::<CudaRuntime>::from_slice(&k_data, &shape, device).unwrap();
    let v = Tensor::<CudaRuntime>::from_slice(&v_data, &shape, device).unwrap();
    (q, k, v)
}

/// Run the graph decode kernel with `window_size` over a cache of `capacity`,
/// live length `live_len`. Mirrors `graph_decode()` above but parameterized
/// instead of pinned to the whole-sequence tests' constants. Returns
/// `(output, lse)` since the full-cache split test checks both.
#[allow(clippy::too_many_arguments)]
fn split_graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    live_len: usize,
    capacity: usize,
    window_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[live_len as i32], &[1], device).unwrap();
    let (out, lse) = decode_attention_graph_fwd(
        client,
        q,
        k,
        v,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        seq_len_k.ptr(),
        capacity,
        window_size,
    )
    .expect("split-KV graph decode attention failed");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Trusted reference: unwindowed non-graph decode over keys `[start,
/// live_len)`. Mirrors `reference_decode()` above but parameterized.
fn split_reference_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    live_len: usize,
    start: usize,
) -> (Vec<f32>, Vec<f32>) {
    let len = live_len - start;
    let k_slice = k.narrow(2, start, len).unwrap().contiguous().unwrap();
    let v_slice = v.narrow(2, start, len).unwrap().contiguous().unwrap();
    let (out, lse) = client
        .flash_attention_fwd(
            q,
            &k_slice,
            &v_slice,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            false,
            0,
            None,
        )
        .expect("reference flash_attention_fwd failed");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Exercises `decode_attention_graph_fwd`'s split-KV path with a fully live
/// cache (`live_len == SPLIT_CAPACITY`): every slice the split kernel walks
/// is entirely populated, so this is the "every slice does real work" case.
/// Fires because `SPLIT_CAPACITY` clears both conditions in
/// `decode_split_count` (see the comment on `SPLIT_CAPACITY` above). Checks
/// both the combined output and the combined log-sum-exp against the
/// non-graph whole-sequence reference.
#[test]
fn graph_decode_split_full_cache_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping graph_decode_split_full_cache_matches_non_graph_reference"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = split_inputs(&device, SPLIT_CAPACITY, SPLIT_SEQ_LEN_K_FULL);

    let (split_out, split_lse) = split_graph_decode(
        &client,
        &device,
        &q,
        &k,
        &v,
        SPLIT_SEQ_LEN_K_FULL,
        SPLIT_CAPACITY,
        0,
    );
    let (ref_out, ref_lse) = split_reference_decode(&client, &q, &k, &v, SPLIT_SEQ_LEN_K_FULL, 0);

    assert_close(
        &split_out,
        &ref_out,
        "graph decode split full-cache output",
        1e-5,
    );
    assert_close(
        &split_lse,
        &ref_lse,
        "graph decode split full-cache lse",
        1e-4,
    );
}

/// Exercises `decode_attention_graph_fwd`'s split-KV path with a mostly-empty
/// cache: the split grid is still sized from `SPLIT_CAPACITY` (the split
/// count is fixed at capture time), but `live_len` is a small fraction of it.
/// Most slices fall entirely past the live length at replay, so this is the
/// risky case where the split kernel's `begin >= end` guard and the combine
/// kernel's `l <= 0` guard must both correctly skip empty slices instead of
/// reading the garbage cache tail or polluting the combined softmax stats.
#[test]
fn graph_decode_split_mostly_empty_cache_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping graph_decode_split_mostly_empty_cache_matches_non_graph_reference"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = split_inputs(&device, SPLIT_CAPACITY, SPLIT_SEQ_LEN_K_EMPTY);

    let (split_out, _split_lse) = split_graph_decode(
        &client,
        &device,
        &q,
        &k,
        &v,
        SPLIT_SEQ_LEN_K_EMPTY,
        SPLIT_CAPACITY,
        0,
    );
    let (ref_out, _ref_lse) = split_reference_decode(&client, &q, &k, &v, SPLIT_SEQ_LEN_K_EMPTY, 0);

    assert_close(
        &split_out,
        &ref_out,
        "graph decode split mostly-empty cache",
        1e-5,
    );
}

/// Exercises `decode_attention_graph_fwd`'s split-KV path together with
/// `window_size`: the window changes `pos_start`, which feeds the same slice
/// `[begin, end)` arithmetic the split kernel uses, so this checks the two
/// features compose correctly instead of only being covered in isolation.
#[test]
fn graph_decode_split_with_window_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping graph_decode_split_with_window_matches_non_graph_reference"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = split_inputs(&device, SPLIT_CAPACITY, SPLIT_SEQ_LEN_K_WINDOW);

    let (split_out, _split_lse) = split_graph_decode(
        &client,
        &device,
        &q,
        &k,
        &v,
        SPLIT_SEQ_LEN_K_WINDOW,
        SPLIT_CAPACITY,
        SPLIT_WINDOW,
    );
    let (ref_out, _ref_lse) = split_reference_decode(
        &client,
        &q,
        &k,
        &v,
        SPLIT_SEQ_LEN_K_WINDOW,
        SPLIT_SEQ_LEN_K_WINDOW - SPLIT_WINDOW,
    );

    assert_close(&split_out, &ref_out, "graph decode split window", 1e-5);
}
