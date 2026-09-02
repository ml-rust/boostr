//! Split-KV tests for the CUDA graph paged decode attention path.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test paged_decode_graph_split
//!
//! The whole file is gated on the `cuda` feature so it compiles away on
//! CPU-only builds.
//!
//! Before this file, `paged_decode_attention_fwd_graph` in
//! `src/ops/cuda/attention/paged_decode.rs` had zero test coverage — no
//! existing test file called it at all, split or not. `tests/decode_graph_window.rs`
//! covers the equivalent *contiguous* graph decode path
//! (`decode_attention_graph_fwd`); this file mirrors its style for the paged
//! path, and `tests/backend_parity/paged_decode_split.rs` covers the
//! *non-graph* paged split path, which serves as the trusted reference here.
//!
//! `decode_split_count(device_index, base_blocks, kv_len)` in
//! `src/ops/cuda/attention/decode_split.rs` returns more than one once BOTH:
//!   - `base_blocks < compute_units * DECODE_BLOCKS_PER_UNIT (8)` — the
//!     whole-sequence grid underfills the device, and
//!   - `kv_len / DECODE_MIN_CHUNK (32) >= 2` — the sequence has at least two
//!     minimum-sized chunks to cut.
//!
//! The graph path sizes `kv_len` from `max_num_blocks * block_size` (the
//! static capacity), not the live `seq_len_k` — see the comment on
//! `paged_decode_attention_fwd_graph`.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::paged_decode::paged_decode_attention_fwd_graph;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;
use numr::dtype::DType;
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
const BLOCK_SIZE: usize = 16;
/// Blocks in the static, capture-time paged cache.
///
/// `base_blocks = BATCH * NUM_HEADS = 4`. Any real CUDA device profile
/// reports `compute_units >= 1`, so `target_blocks = compute_units * 8 >= 8 >
/// 4` and the whole-sequence grid is always underfilled. `CAPACITY / 32 =
/// 2048 / 32 = 64 >= 2`, so the minimum-chunk floor is cleared with a wide
/// margin — comfortably enough that the split path fires even on a device
/// with very few compute units, while the resulting split count still lands
/// inside `DECODE_MAX_SPLITS (32)`.
const MAX_NUM_BLOCKS: usize = 128;
const CAPACITY: usize = MAX_NUM_BLOCKS * BLOCK_SIZE;
/// Live length for the full-cache split test: equal to `CAPACITY`, so every
/// split slice is entirely populated.
const SEQ_LEN_K_FULL: usize = CAPACITY;
/// Live length for the mostly-empty split test: far below `CAPACITY`, a whole
/// number of blocks so the reference's own `blocks_per_seq` divides evenly.
/// Most of the statically-sized split grid falls past the live cache and must
/// be skipped by the split kernel's `blk_begin >= blk_end` guard, then
/// dropped by the combine kernel's `l <= 0` guard.
const SEQ_LEN_K_EMPTY: usize = 6 * BLOCK_SIZE;

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Physical block holding logical block `logical` of `total`. Reversing the
/// order is a bijection, so every page is used exactly once but no logical
/// block sits at its own physical index — a kernel that ignores the block
/// table and reads pages in order diverges from the reference.
fn scrambled_page(logical: usize, total: usize) -> i32 {
    (total - 1 - logical) as i32
}

/// Q `[B, H, 1, D]`, K/V blocks `[MAX_NUM_BLOCKS, BLOCK_SIZE, KVH, D]`, and a
/// scrambled `block_table [B, MAX_NUM_BLOCKS]` sized for the full static
/// capacity. The physical slots that logical positions at or beyond
/// `live_len` map to — walked through the same indirection the kernel uses —
/// hold large garbage, so a kernel that reads past the live length blows the
/// comparison apart instead of drifting quietly.
fn paged_split_inputs(
    device: &CudaDevice,
    live_len: usize,
) -> (
    Tensor<CudaRuntime>,
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

    let cache_shape = [MAX_NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM];
    let cache_len = MAX_NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let mut k_data = values(cache_len, 1.1);
    let mut v_data = values(cache_len, 2.7);

    let bt_data: Vec<i32> = (0..MAX_NUM_BLOCKS)
        .map(|logical| scrambled_page(logical, MAX_NUM_BLOCKS))
        .collect();

    for pos in live_len..CAPACITY {
        let logical_block = pos / BLOCK_SIZE;
        let offset = pos % BLOCK_SIZE;
        let physical_block = bt_data[logical_block] as usize;
        for kv_h in 0..NUM_KV_HEADS {
            for d in 0..HEAD_DIM {
                let idx =
                    ((physical_block * BLOCK_SIZE + offset) * NUM_KV_HEADS + kv_h) * HEAD_DIM + d;
                k_data[idx] = 50.0;
                v_data[idx] = -50.0;
            }
        }
    }

    let k = Tensor::<CudaRuntime>::from_slice(&k_data, &cache_shape, device).unwrap();
    let v = Tensor::<CudaRuntime>::from_slice(&v_data, &cache_shape, device).unwrap();
    let block_table =
        Tensor::<CudaRuntime>::from_slice(&bt_data, &[BATCH, MAX_NUM_BLOCKS], device).unwrap();
    (q, k, v, block_table)
}

/// Run `paged_decode_attention_fwd_graph` with the static `MAX_NUM_BLOCKS` /
/// `BLOCK_SIZE` capacity and a device-resident `live_len`.
#[allow(clippy::too_many_arguments)]
fn paged_split_graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    live_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[live_len as i32], &[1], device).unwrap();
    let output =
        Tensor::<CudaRuntime>::empty(&[BATCH, NUM_HEADS, 1, HEAD_DIM], DType::F32, device).unwrap();
    let lse = Tensor::<CudaRuntime>::empty(&[BATCH, NUM_HEADS, 1], DType::F32, device).unwrap();

    paged_decode_attention_fwd_graph(
        client,
        q,
        k,
        v,
        block_table,
        &output,
        &lse,
        BATCH,
        NUM_HEADS,
        NUM_KV_HEADS,
        seq_len_k.ptr(),
        HEAD_DIM,
        BLOCK_SIZE,
        MAX_NUM_BLOCKS,
    )
    .expect("paged split-KV graph decode attention failed");

    (output.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Trusted reference: the non-graph `paged_attention_fwd` decode fast path,
/// over the same full-width `block_table`. It sizes its own split count from
/// the live `seq_len_k`, independently of the graph path under test, and is
/// already checked against CPU in `tests/backend_parity/paged_decode_split.rs`.
fn paged_split_reference_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    live_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (out, lse) = client
        .paged_attention_fwd(
            q,
            k,
            v,
            block_table,
            NUM_HEADS,
            NUM_KV_HEADS,
            1,
            live_len,
            HEAD_DIM,
            BLOCK_SIZE,
            false,
        )
        .expect("reference paged_attention_fwd failed");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
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

/// Exercises `paged_decode_attention_fwd_graph`'s split-KV path with a fully
/// live cache (`live_len == CAPACITY`): every slice the split kernel walks
/// is entirely populated. Fires because `CAPACITY` clears both conditions in
/// `decode_split_count` (see the comment on `MAX_NUM_BLOCKS` above). Checks
/// both the combined output and the combined log-sum-exp against the
/// non-graph reference.
#[test]
fn paged_graph_decode_split_full_cache_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping paged_graph_decode_split_full_cache_matches_non_graph_reference"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v, block_table) = paged_split_inputs(&device, SEQ_LEN_K_FULL);

    let (split_out, split_lse) =
        paged_split_graph_decode(&client, &device, &q, &k, &v, &block_table, SEQ_LEN_K_FULL);
    let (ref_out, ref_lse) =
        paged_split_reference_decode(&client, &q, &k, &v, &block_table, SEQ_LEN_K_FULL);

    assert_close(
        &split_out,
        &ref_out,
        "paged graph decode split full-cache output",
        1e-5,
    );
    assert_close(
        &split_lse,
        &ref_lse,
        "paged graph decode split full-cache lse",
        1e-4,
    );
}

/// Exercises `paged_decode_attention_fwd_graph`'s split-KV path with a
/// mostly-empty cache: the split grid is still sized from the static
/// `MAX_NUM_BLOCKS * BLOCK_SIZE` capacity (the split count is fixed at
/// capture time), but `live_len` is a small fraction of it. Most slices fall
/// entirely past the live length at replay, so this is the risky case where
/// the split kernel's block-range guard and the combine kernel's `l <= 0`
/// guard must both correctly skip empty slices instead of reading the
/// garbage cache tail (reached through the scrambled block table) or
/// polluting the combined softmax stats.
#[test]
fn paged_graph_decode_split_mostly_empty_cache_matches_non_graph_reference() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping paged_graph_decode_split_mostly_empty_cache_matches_non_graph_reference"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let (q, k, v, block_table) = paged_split_inputs(&device, SEQ_LEN_K_EMPTY);

    let (split_out, _split_lse) =
        paged_split_graph_decode(&client, &device, &q, &k, &v, &block_table, SEQ_LEN_K_EMPTY);
    let (ref_out, _ref_lse) =
        paged_split_reference_decode(&client, &q, &k, &v, &block_table, SEQ_LEN_K_EMPTY);

    assert_close(
        &split_out,
        &ref_out,
        "paged graph decode split mostly-empty cache",
        1e-5,
    );
}
