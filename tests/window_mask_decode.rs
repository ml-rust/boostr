//! Sliding-window correctness on the non-graph CPU attention path.
//!
//! Run with:
//!   cd boostr && cargo test --test window_mask_decode
//!
//! Query row `i` is at ABSOLUTE sequence position `key_offset + i`, where
//! `key_offset = seq_len_k - seq_len_q`. A KV-cached decode passes one query
//! against the whole cache, so its query sits at `seq_len_k - 1` and a window
//! keeps exactly the contiguous key suffix. Before that convention landed, the
//! mask read `i` as a position inside the query tensor, so `seq_len_q == 1`
//! made every window term false and `sliding_window` was silently ignored.
//!
//! References are built by running the SAME entry point over the narrowed key
//! range the window selects, with `window_size = 0` — a single query over a
//! key slice is mathematically the windowed result, and it goes through no
//! masking at all.

use boostr::ops::traits::attention::flash::FlashAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const BATCH: usize = 1;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
/// Key/cache length for the decode step.
const SEQ_LEN_K: usize = 20;
/// Window shorter than the cache, so it must change the result.
const WINDOW: usize = 6;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

fn tensor(shape: &[usize], seed: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&values(n, seed), shape, device).unwrap()
}

/// Q `[B, H, seq_len_q, D]` plus K/V `[B, KVH, SEQ_LEN_K, D]`.
fn inputs(
    seq_len_q: usize,
    device: &CpuDevice,
) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
    let q = tensor(&[BATCH, NUM_HEADS, seq_len_q, HEAD_DIM], 0.3, device);
    let kv_shape = [BATCH, NUM_KV_HEADS, SEQ_LEN_K, HEAD_DIM];
    let k = tensor(&kv_shape, 1.1, device);
    let v = tensor(&kv_shape, 2.7, device);
    (q, k, v)
}

fn attend(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    causal: bool,
    window_size: usize,
) -> Vec<f32> {
    let (out, _lse) = client
        .flash_attention_fwd(
            q,
            k,
            v,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal,
            window_size,
            None,
        )
        .expect("flash_attention_fwd failed");
    out.to_vec::<f32>()
}

/// Unwindowed attention of `q` over keys `[start, start + len)`.
fn reference_over_slice(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    start: usize,
    len: usize,
) -> Vec<f32> {
    let k_slice = k.narrow(2, start, len).unwrap().contiguous().unwrap();
    let v_slice = v.narrow(2, start, len).unwrap().contiguous().unwrap();
    attend(client, q, &k_slice, &v_slice, false, 0)
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

/// The test that matters: windowed single-token decode == unwindowed decode
/// over the key suffix the window selects. This FAILS before the fix, where a
/// windowed decode returns the unwindowed answer.
#[test]
fn windowed_decode_matches_key_suffix() {
    let (client, device) = cpu_setup();
    let (q, k, v) = inputs(1, &device);

    let windowed = attend(&client, &q, &k, &v, false, WINDOW);
    let reference = reference_over_slice(&client, &q, &k, &v, SEQ_LEN_K - WINDOW, WINDOW);

    assert_close(&windowed, &reference, "decode window vs suffix", 1e-5);
}

/// Without this, the test above could pass with the window still ignored.
#[test]
fn windowed_decode_differs_from_unwindowed() {
    let (client, device) = cpu_setup();
    let (q, k, v) = inputs(1, &device);

    let windowed = attend(&client, &q, &k, &v, false, WINDOW);
    let unwindowed = attend(&client, &q, &k, &v, false, 0);

    let diff = max_abs_diff(&windowed, &unwindowed);
    assert!(
        diff > 1e-3,
        "window {WINDOW} over {SEQ_LEN_K} keys left the result unchanged (max diff {diff})"
    );
}

/// A window at least as long as the cache masks nothing, so it equals
/// `window_size = 0` — the boundary of the inclusive-window rule.
#[test]
fn decode_window_wider_than_cache_equals_zero_window() {
    let (client, device) = cpu_setup();
    let (q, k, v) = inputs(1, &device);

    let wide = attend(&client, &q, &k, &v, false, SEQ_LEN_K);
    let unwindowed = attend(&client, &q, &k, &v, false, 0);

    assert_close(&wide, &unwindowed, "decode wide window", 1e-6);
}

/// Chunked prefill, `1 < seq_len_q < seq_len_k`: row `i` is at absolute
/// position `key_offset + i` and keeps the `WINDOW` keys ending there. Each row
/// is checked against a single-query run over exactly those keys.
#[test]
fn chunked_window_matches_per_row_reference() {
    let seq_len_q = 3usize;
    let (client, device) = cpu_setup();
    let (q, k, v) = inputs(seq_len_q, &device);
    let key_offset = SEQ_LEN_K - seq_len_q;

    let got = attend(&client, &q, &k, &v, true, WINDOW);

    for i in 0..seq_len_q {
        let q_pos = key_offset + i;
        let q_row = q.narrow(2, i, 1).unwrap().contiguous().unwrap();
        let want = reference_over_slice(&client, &q_row, &k, &v, q_pos + 1 - WINDOW, WINDOW);

        // Row `i` of `got` for every (batch, head): [B, H, S_q, D] layout.
        let mut row = Vec::with_capacity(BATCH * NUM_HEADS * HEAD_DIM);
        for bh in 0..BATCH * NUM_HEADS {
            let base = (bh * seq_len_q + i) * HEAD_DIM;
            row.extend_from_slice(&got[base..base + HEAD_DIM]);
        }
        assert_close(&row, &want, &format!("chunked row {i}"), 1e-5);
    }
}

/// Prefill (`seq_len_q == seq_len_k`) has `key_offset == 0`. Windowed prefill
/// keeps working as the plain relative rule: row `i` sees keys
/// `(i - WINDOW, i]`.
#[test]
fn prefill_window_is_unchanged() {
    let (client, device) = cpu_setup();
    let q = tensor(&[BATCH, NUM_HEADS, SEQ_LEN_K, HEAD_DIM], 0.3, &device);
    let kv_shape = [BATCH, NUM_KV_HEADS, SEQ_LEN_K, HEAD_DIM];
    let k = tensor(&kv_shape, 1.1, &device);
    let v = tensor(&kv_shape, 2.7, &device);

    let got = attend(&client, &q, &k, &v, true, WINDOW);

    for i in 0..SEQ_LEN_K {
        let start = (i + 1).saturating_sub(WINDOW);
        let len = i + 1 - start;
        let q_row = q.narrow(2, i, 1).unwrap().contiguous().unwrap();
        let want = reference_over_slice(&client, &q_row, &k, &v, start, len);

        let mut row = Vec::with_capacity(BATCH * NUM_HEADS * HEAD_DIM);
        for bh in 0..BATCH * NUM_HEADS {
            let base = (bh * SEQ_LEN_K + i) * HEAD_DIM;
            row.extend_from_slice(&got[base..base + HEAD_DIM]);
        }
        assert_close(&row, &want, &format!("prefill row {i}"), 1e-5);
    }
}
