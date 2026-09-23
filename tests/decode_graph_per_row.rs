//! Per-row key bounds on the CUDA graph decode attention entries.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test decode_graph_per_row
//!
//! The whole file is gated on the `cuda` feature so it compiles away on
//! CPU-only builds.
//!
//! No graph is captured here: `decode_attention_graph_fwd` is called directly
//! with ordinary tensors, which is what exercises the `_graph` and
//! `_split_graph` kernel entries. Capture only freezes the grid, and the grid
//! comes from the static capacity either way.
//!
//! Reference: the non-graph `flash_attention_fwd` decode path with
//! `KvSeqLen::PerRow` and `kv_start`, over the SAME full-capacity K/V buffer.
//! Both paths then size their split grid from that capacity and pass the same
//! `fill`, so every row is cut into the same slices and merged in the same
//! order. The comparison is therefore bitwise, not approximate.
//!
//! `*seq_len_k_ptr` is the row-independent capacity clamp in
//! `row_end = min(kv_end[b], seq_len_k)`. Passing the capacity makes it a
//! no-op, which is what lets the two paths agree bit for bit;
//! `graph_capacity_clamp_caps_per_row_end` covers a live value below it by
//! clamping the reference's ends on the host instead.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::flash::impl_ops::decode_attention_graph_fwd;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use boostr::ops::{AttnOutLayout, KvSeqLen};
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

const BATCH: usize = 3;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;

/// Capacity below the split threshold, so the whole-sequence `_graph` entry
/// runs. `decode_slices` cuts a span into slices of at least 32 positions and
/// keeps one slice below two of them, so 48 never splits.
const WHOLE_CAPACITY: usize = 48;
/// Capacity above the split threshold, so the `_split_graph` entry runs.
/// `NUM_HEADS = 4` at `HEAD_DIM = 64` asks for slices of at most 1024
/// positions on any device, so 2048 is at least two slices.
const SPLIT_CAPACITY: usize = 2048;

/// Per-row `[start, end)` key ranges at `WHOLE_CAPACITY`. Row 2 is empty.
const WHOLE_ROWS: [(i32, i32); BATCH] = [(0, 40), (7, 31), (19, 19)];
/// Per-row `[start, end)` key ranges at `SPLIT_CAPACITY`. Row 2 is empty, and
/// the lengths straddle several slice boundaries.
const SPLIT_ROWS: [(i32, i32); BATCH] = [(0, 2048), (137, 1611), (640, 640)];

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Q `[B, H, 1, D]` plus K/V caches `[B, KVH, capacity, D]`, every slot filled.
/// Nothing is poisoned: a row's bounds are the only thing keeping it off the
/// slots outside its range, which is exactly what these tests check.
fn inputs(
    device: &CudaDevice,
    capacity: usize,
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
    let shape = [BATCH, NUM_KV_HEADS, capacity, HEAD_DIM];
    let k = Tensor::<CudaRuntime>::from_slice(&values(cache_len, 1.1), &shape, device).unwrap();
    let v = Tensor::<CudaRuntime>::from_slice(&values(cache_len, 2.7), &shape, device).unwrap();
    (q, k, v)
}

fn bounds(
    device: &CudaDevice,
    rows: &[(i32, i32); BATCH],
) -> (Tensor<CudaRuntime>, Tensor<CudaRuntime>) {
    let starts: Vec<i32> = rows.iter().map(|r| r.0).collect();
    let ends: Vec<i32> = rows.iter().map(|r| r.1).collect();
    (
        Tensor::<CudaRuntime>::from_slice(&starts, &[BATCH], device).unwrap(),
        Tensor::<CudaRuntime>::from_slice(&ends, &[BATCH], device).unwrap(),
    )
}

/// Device pointers to a `[B]` i32 per-row `kv_start`/`kv_end`, or `0` for
/// either when the call has no per-row bound.
struct RowBoundPtrs {
    kv_start: u64,
    kv_end: u64,
}

/// Q/K/V tensors for one `graph_decode` call.
struct QkvRefs<'a> {
    q: &'a Tensor<CudaRuntime>,
    k: &'a Tensor<CudaRuntime>,
    v: &'a Tensor<CudaRuntime>,
}

/// `decode_attention_graph_fwd` over `capacity` slots with `seq_len_k` on the
/// device and optional per-row bound pointers. Returns `(output, lse)`.
fn graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    qkv: QkvRefs<'_>,
    seq_len_k: usize,
    bounds: RowBoundPtrs,
    capacity: usize,
) -> (Vec<f32>, Vec<f32>) {
    let QkvRefs { q, k, v } = qkv;
    let RowBoundPtrs {
        kv_start: kv_start_ptr,
        kv_end: kv_end_ptr,
    } = bounds;
    let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[seq_len_k as i32], &[1], device).unwrap();
    let (out, lse) = decode_attention_graph_fwd(
        client,
        q,
        k,
        v,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        seq_len_k.ptr(),
        kv_start_ptr,
        kv_end_ptr,
        capacity,
        0,
    )
    .expect("graph decode attention failed");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Trusted reference: the non-graph decode path over the same full-capacity
/// buffer, with the same per-row start and end.
fn reference_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    kv_start: &Tensor<CudaRuntime>,
    kv_end: &Tensor<CudaRuntime>,
) -> (Vec<f32>, Vec<f32>) {
    let (out, lse) = client
        .flash_attention_fwd(
            q,
            k,
            v,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            false,
            0,
            Some(KvSeqLen::PerRow(kv_end)),
            Some(kv_start),
            AttnOutLayout::HeadMajor,
        )
        .expect("reference flash_attention_fwd failed");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Bitwise equality, reported per index so a single drifting lane is named.
fn assert_bitwise(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label} at index {i}: graph {x} vs reference {y}"
        );
    }
}

/// Elements of row `b` in an `[B, H, 1, D]` output buffer.
fn out_row(out: &[f32], b: usize) -> &[f32] {
    let per_row = NUM_HEADS * HEAD_DIM;
    &out[b * per_row..(b + 1) * per_row]
}

/// Elements of row `b` in a `[B, H, 1]` LSE buffer.
fn lse_row(lse: &[f32], b: usize) -> &[f32] {
    &lse[b * NUM_HEADS..(b + 1) * NUM_HEADS]
}

/// One capacity's per-row check: every row of the graph entry equals the same
/// row through the non-graph path, bit for bit.
fn per_row_matches_reference(capacity: usize, rows: &[(i32, i32); BATCH], label: &str) {
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v) = inputs(&device, capacity);
    let (kv_start, kv_end) = bounds(&device, rows);

    let (out, lse) = graph_decode(
        &client,
        &device,
        QkvRefs {
            q: &q,
            k: &k,
            v: &v,
        },
        capacity,
        RowBoundPtrs {
            kv_start: kv_start.ptr(),
            kv_end: kv_end.ptr(),
        },
        capacity,
    );
    let (ref_out, ref_lse) = reference_decode(&client, &q, &k, &v, &kv_start, &kv_end);

    for b in 0..BATCH {
        assert_bitwise(
            out_row(&out, b),
            out_row(&ref_out, b),
            &format!("{label} out row {b}"),
        );
        assert_bitwise(
            lse_row(&lse, b),
            lse_row(&ref_lse, b),
            &format!("{label} lse row {b}"),
        );
    }
}

/// Whole-sequence grid: distinct per-row ranges in one launch.
#[test]
fn graph_per_row_bounds_match_non_graph_whole_sequence() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping graph_per_row_bounds_match_non_graph_whole_sequence"
        );
        return;
    }
    per_row_matches_reference(WHOLE_CAPACITY, &WHOLE_ROWS, "whole-sequence");
}

/// Split-KV grid: distinct per-row ranges straddling slice boundaries.
#[test]
fn graph_per_row_bounds_match_non_graph_split() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_per_row_bounds_match_non_graph_split");
        return;
    }
    per_row_matches_reference(SPLIT_CAPACITY, &SPLIT_ROWS, "split-KV");
}

/// A row whose start equals its end shares the launch with populated rows: it
/// stores zeros and an LSE of `-inf`, and the rows beside it are untouched
/// (the bitwise checks above cover those, on the same fixtures).
#[test]
fn graph_per_row_empty_row_stores_zeros_and_neg_inf() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_per_row_empty_row_stores_zeros_and_neg_inf");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    for (capacity, rows, label) in [
        (WHOLE_CAPACITY, &WHOLE_ROWS, "whole-sequence"),
        (SPLIT_CAPACITY, &SPLIT_ROWS, "split-KV"),
    ] {
        let empty = rows
            .iter()
            .position(|&(s, e)| s == e)
            .expect("fixture must hold one empty row");
        let (q, k, v) = inputs(&device, capacity);
        let (kv_start, kv_end) = bounds(&device, rows);
        let (out, lse) = graph_decode(
            &client,
            &device,
            QkvRefs {
                q: &q,
                k: &k,
                v: &v,
            },
            capacity,
            RowBoundPtrs {
                kv_start: kv_start.ptr(),
                kv_end: kv_end.ptr(),
            },
            capacity,
        );

        for (i, &x) in out_row(&out, empty).iter().enumerate() {
            assert_eq!(x, 0.0, "{label} empty row {empty} out index {i}: {x}");
        }
        for (h, &x) in lse_row(&lse, empty).iter().enumerate() {
            assert_eq!(
                x,
                f32::NEG_INFINITY,
                "{label} empty row {empty} lse head {h}: {x}"
            );
        }
        // A populated row beside it still attends: its LSE is finite.
        let populated = rows
            .iter()
            .position(|&(s, e)| s != e)
            .expect("fixture must hold one populated row");
        for (h, &x) in lse_row(&lse, populated).iter().enumerate() {
            assert!(x.is_finite(), "{label} row {populated} lse head {h}: {x}");
        }
    }
}

/// Null bounds keep the arithmetic they had: the same bits as explicit bounds
/// that select the whole live span of every row.
#[test]
fn graph_null_bounds_match_explicit_full_bounds() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_null_bounds_match_explicit_full_bounds");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    for (capacity, live, label) in [
        (WHOLE_CAPACITY, 31usize, "whole-sequence"),
        (SPLIT_CAPACITY, 1611usize, "split-KV"),
    ] {
        let (q, k, v) = inputs(&device, capacity);
        let full: [(i32, i32); BATCH] = [(0, live as i32); BATCH];
        let (kv_start, kv_end) = bounds(&device, &full);

        let (null_out, null_lse) = graph_decode(
            &client,
            &device,
            QkvRefs {
                q: &q,
                k: &k,
                v: &v,
            },
            live,
            RowBoundPtrs {
                kv_start: 0,
                kv_end: 0,
            },
            capacity,
        );
        let (explicit_out, explicit_lse) = graph_decode(
            &client,
            &device,
            QkvRefs {
                q: &q,
                k: &k,
                v: &v,
            },
            capacity,
            RowBoundPtrs {
                kv_start: kv_start.ptr(),
                kv_end: kv_end.ptr(),
            },
            capacity,
        );

        assert_bitwise(
            &null_out,
            &explicit_out,
            &format!("{label} null-vs-explicit out"),
        );
        assert_bitwise(
            &null_lse,
            &explicit_lse,
            &format!("{label} null-vs-explicit lse"),
        );
    }
}

/// `seq_len_k` below the capacity clamps every row's end without touching its
/// start, matching the non-graph path fed the host-clamped ends.
#[test]
fn graph_capacity_clamp_caps_per_row_end() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping graph_capacity_clamp_caps_per_row_end");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    // Live length below two of the three rows' ends, so the clamp bites.
    const LIVE: i32 = 25;
    let (q, k, v) = inputs(&device, WHOLE_CAPACITY);
    let (kv_start, kv_end) = bounds(&device, &WHOLE_ROWS);
    let clamped: [(i32, i32); BATCH] = [
        (WHOLE_ROWS[0].0, WHOLE_ROWS[0].1.min(LIVE)),
        (WHOLE_ROWS[1].0, WHOLE_ROWS[1].1.min(LIVE)),
        (WHOLE_ROWS[2].0, WHOLE_ROWS[2].1.min(LIVE)),
    ];
    let (ref_start, ref_end) = bounds(&device, &clamped);

    let (out, lse) = graph_decode(
        &client,
        &device,
        QkvRefs {
            q: &q,
            k: &k,
            v: &v,
        },
        LIVE as usize,
        RowBoundPtrs {
            kv_start: kv_start.ptr(),
            kv_end: kv_end.ptr(),
        },
        WHOLE_CAPACITY,
    );
    let (ref_out, ref_lse) = reference_decode(&client, &q, &k, &v, &ref_start, &ref_end);

    assert_bitwise(&out, &ref_out, "capacity clamp out");
    assert_bitwise(&lse, &ref_lse, "capacity clamp lse");
}
