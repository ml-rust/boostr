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
