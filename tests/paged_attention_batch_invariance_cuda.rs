//! A row's CUDA paged-decode result does not depend on how many rows share
//! the batched `paged_attention_fwd` call, and never reads another row's pages.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test paged_attention_batch_invariance_cuda
//!
//! `paged_attention_fwd` takes one `seq_len_k` for the whole batch and a
//! per-row `KvSeqLen::PerRow(ends)` saying where each row's own keys stop.
//! Row `b` of a batched call must be bit-identical to the same row computed
//! alone, with its own block table and its own `seq_len_k`.
//!
//! One row in each fixture is short enough that its block table is zero-padded
//! past its real pages, and the padding entries point at physical block 0 —
//! which holds a longer row's live keys and values. Without the per-row end
//! clamping the last, partly filled page and the page loop to the row's own
//! keys, that row's output would depend on data it does not own.
//!
//! Both decode grid shapes run here. A block-table capacity of fewer than two
//! slice chunks keeps one block per `(batch, Q head)`; a wider one cuts each
//! row's page run into slices merged by the combine pass.
//! `src/ops/cuda/attention/decode_split.rs` owns that choice. Rather than
//! assume which grid a fixture lands on, each case below calls
//! `paged_decode_splits` — the same sizing `paged_decode_attention_fwd`
//! applies to the block-table capacity — and asserts the grid it got: `1`
//! slice for `WHOLE_LENS`, more than one for `SPLIT_LENS`. A mismatch fails
//! the test loudly instead of letting the wrong kernel run unnoticed.
//!
//! The alone call passes its own single-element `PerRow` end, so both sides
//! take the per-row slicing rule and the comparison is of batching alone.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::decode_split::paged_decode_splits;
use boostr::ops::traits::KvSeqLen;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;
use numr::runtime::Device;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 16;

/// Per-row key counts for the whole-sequence grid: three pages of capacity,
/// one slice chunk, so no shape here splits. Every row's last page is partly
/// filled.
const WHOLE_LENS: [usize; 3] = [45, 5, 20];

/// Per-row key counts for the split grid: the widest row needs 63 pages, far
/// past the two-chunk threshold, and the shortest needs three.
const SPLIT_LENS: [usize; 4] = [1000, 37, 520, 129];

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

/// Deterministic values, distinct per index so a row that reads the wrong slot
/// changes the answer.
fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * ((i as f32) * 0.731 + seed).sin() + 0.1 * ((i as f32) * 0.017).cos())
        .collect()
}

/// Physical pages each row owns, handed out consecutively so no row's real
/// data overlaps another's, plus the total page count.
///
/// Row 0 owns physical block 0, the value every short row's spare block-table
/// slots carry, so a leak past a row's own keys lands in row 0's live data.
fn page_layout(lens: &[usize]) -> (Vec<Vec<i32>>, usize) {
    let mut next = 0i32;
    let mut rows = Vec::with_capacity(lens.len());
    for &len in lens {
        let pages = len.div_ceil(BLOCK_SIZE);
        rows.push((next..next + pages as i32).collect::<Vec<i32>>());
        next += pages as i32;
    }
    (rows, next as usize)
}

/// The batch's block table, zero-padded past each row's real pages up to the
/// widest row. The padding value `0` is a real physical block, chosen
/// deliberately, not an out-of-range sentinel the kernel would trip over.
fn block_table_data(rows: &[Vec<i32>], max_num_blocks: usize) -> Vec<i32> {
    let mut data = vec![0i32; rows.len() * max_num_blocks];
    for (b, row) in rows.iter().enumerate() {
        for (i, &phys) in row.iter().enumerate() {
            data[b * max_num_blocks + i] = phys;
        }
    }
    data
}

/// One paged decode call. `ends` is `None` for the shared-bound call.
#[allow(clippy::too_many_arguments)]
fn paged_decode(
    client: &CudaClient,
    device: &CudaDevice,
    q_data: &[f32],
    batch: usize,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    bt_data: &[i32],
    max_num_blocks: usize,
    seq_len_k: usize,
    ends: Option<&[i32]>,
) -> (Vec<f32>, Vec<f32>) {
    let q = Tensor::<CudaRuntime>::from_slice(q_data, &[batch, NUM_HEADS, 1, HEAD_DIM], device)
        .expect("q");
    let block_table =
        Tensor::<CudaRuntime>::from_slice(bt_data, &[batch, max_num_blocks], device).expect("bt");
    let ends_t = ends
        .map(|e| Tensor::<CudaRuntime>::from_slice(e, &[batch], device).expect("per-row key ends"));

    let (out, lse) = client
        .paged_attention_fwd(
            &q,
            k_blocks,
            v_blocks,
            &block_table,
            NUM_HEADS,
            NUM_KV_HEADS,
            1,
            seq_len_k,
            HEAD_DIM,
            BLOCK_SIZE,
            false,
            ends_t.as_ref().map(KvSeqLen::PerRow),
        )
        .expect("paged_attention_fwd");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// The shared paged cache, `[total_blocks, BLOCK_SIZE, NUM_KV_HEADS,
/// HEAD_DIM]`, every page's bytes distinct so a wrong physical index shows in
/// the output.
fn build_cache(
    device: &CudaDevice,
    total_blocks: usize,
) -> (Tensor<CudaRuntime>, Tensor<CudaRuntime>) {
    let cache_len = total_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let shape = [total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM];
    let k = Tensor::<CudaRuntime>::from_slice(&values(cache_len, 1.1), &shape, device)
        .expect("k cache");
    let v = Tensor::<CudaRuntime>::from_slice(&values(cache_len, 2.7), &shape, device)
        .expect("v cache");
    (k, v)
}

/// Runs one mixed-length batch and asserts every row matches its own
/// single-row decode bit for bit, on both the output and the log-sum-exp.
///
/// Before touching a single tensor, asserts the batched call actually takes
/// the grid `label` is meant to exercise: `expect_split` false requires
/// exactly one slice (the whole-sequence kernel), true requires more than
/// one (the split kernel merged by the combine pass). A device on which the
/// fixture lands on the other grid fails here, loudly, naming the computed
/// slice count, instead of silently leaving that kernel untested.
fn assert_rows_match_alone(label: &str, lens: &[usize], expect_split: bool) {
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let batch = lens.len();
    let seq_len_k = *lens.iter().max().expect("non-empty lens");
    let (rows, total_blocks) = page_layout(lens);
    let max_num_blocks = rows.iter().map(|r| r.len()).max().expect("non-empty rows");

    let splits = paged_decode_splits(device.id(), NUM_HEADS, max_num_blocks, BLOCK_SIZE, HEAD_DIM);
    if expect_split {
        assert!(
            splits > 1,
            "{label}: expected the split grid but paged_decode_splits computed {splits} \
             slice(s) for max_num_blocks={max_num_blocks} (SPLIT_LENS no longer produces a \
             split on this device — widen SPLIT_LENS)"
        );
    } else {
        assert_eq!(
            splits, 1,
            "{label}: expected the whole-sequence grid but paged_decode_splits computed \
             {splits} slices for max_num_blocks={max_num_blocks} (WHOLE_LENS now produces a \
             split on this device — shrink WHOLE_LENS)"
        );
    }

    let (k_blocks, v_blocks) = build_cache(&device, total_blocks);
    let bt_data = block_table_data(&rows, max_num_blocks);

    let q_row = NUM_HEADS * HEAD_DIM;
    let q_data = values(batch * q_row, 0.3);
    let ends: Vec<i32> = lens.iter().map(|&l| l as i32).collect();

    let (out, lse) = paged_decode(
        &client,
        &device,
        &q_data,
        batch,
        &k_blocks,
        &v_blocks,
        &bt_data,
        max_num_blocks,
        seq_len_k,
        Some(&ends),
    );

    for (b, &own_len) in lens.iter().enumerate() {
        let (out_b, lse_b) = paged_decode(
            &client,
            &device,
            &q_data[b * q_row..(b + 1) * q_row],
            1,
            &k_blocks,
            &v_blocks,
            &rows[b],
            rows[b].len(),
            own_len,
            Some(&[own_len as i32]),
        );

        for (i, (got, want)) in out[b * q_row..(b + 1) * q_row]
            .iter()
            .zip(&out_b)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "{label} row {b} (len {own_len}) output element {i}: batched {got:e}, alone {want:e}"
            );
        }
        for (h, (got, want)) in lse[b * NUM_HEADS..(b + 1) * NUM_HEADS]
            .iter()
            .zip(&lse_b)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "{label} row {b} (len {own_len}) head {h} LSE: batched {got:e}, alone {want:e}"
            );
        }
    }
}

/// Whole-sequence grid: asserts `paged_decode_splits` computes exactly one
/// slice for `WHOLE_LENS`'s capacity, then checks one block per
/// `(batch, Q head)` walks each row's own page run.
#[test]
fn batched_row_matches_its_own_alone_decode_whole_sequence() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping batched_row_matches_its_own_alone_decode_whole_sequence"
        );
        return;
    }
    assert_rows_match_alone("whole-sequence", &WHOLE_LENS, false);
}

/// Split grid: asserts `paged_decode_splits` computes more than one slice for
/// `SPLIT_LENS`'s capacity, then checks each row's page run is cut into
/// slices, the slices past its end coming out empty, so the real folds stay
/// in the same order after the combine pass.
#[test]
fn batched_row_matches_its_own_alone_decode_split() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping batched_row_matches_its_own_alone_decode_split");
        return;
    }
    assert_rows_match_alone("split", &SPLIT_LENS, true);
}

/// Same batch with `kv_seq_len: None`, the shared-bound call: every row reads
/// out to the batch maximum, so the short row's output MUST differ from its
/// own alone decode. Without this the tests above could pass for a reason
/// unrelated to the per-row end.
#[test]
fn without_per_row_ends_the_short_row_disagrees_with_its_own_decode() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping without_per_row_ends_the_short_row_disagrees_with_its_own_decode"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let lens = WHOLE_LENS;
    let batch = lens.len();
    let seq_len_k = *lens.iter().max().expect("non-empty lens");
    let (rows, total_blocks) = page_layout(&lens);
    let max_num_blocks = rows.iter().map(|r| r.len()).max().expect("non-empty rows");
    let (k_blocks, v_blocks) = build_cache(&device, total_blocks);
    let bt_data = block_table_data(&rows, max_num_blocks);

    let q_row = NUM_HEADS * HEAD_DIM;
    let q_data = values(batch * q_row, 0.3);

    let (out, _) = paged_decode(
        &client,
        &device,
        &q_data,
        batch,
        &k_blocks,
        &v_blocks,
        &bt_data,
        max_num_blocks,
        seq_len_k,
        None,
    );

    // Row 1 is the short row, whose block table is zero-padded into block 0.
    let b = 1;
    let (out_b, _) = paged_decode(
        &client,
        &device,
        &q_data[b * q_row..(b + 1) * q_row],
        1,
        &k_blocks,
        &v_blocks,
        &rows[b],
        rows[b].len(),
        lens[b],
        None,
    );

    let differs = out[b * q_row..(b + 1) * q_row]
        .iter()
        .zip(&out_b)
        .any(|(got, want)| got.to_bits() != want.to_bits());
    assert!(
        differs,
        "fixture is degenerate: without per-row ends the short row should read padding into \
         block 0 and disagree with its own decode, but it matched"
    );
}

/// The padding target is not an accidentally all-zero page: physical block 0
/// holds real, distinct, nonzero keys and values, so a leak into it is visible.
#[test]
fn the_padding_target_block_holds_real_nonzero_data() {
    let (_, total_blocks) = page_layout(&WHOLE_LENS);
    let cache_len = total_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let page_elems = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k_page0 = values(cache_len, 1.1)[..page_elems].to_vec();
    let v_page0 = values(cache_len, 2.7)[..page_elems].to_vec();
    assert!(
        k_page0.iter().any(|&x| x != 0.0) && v_page0.iter().any(|&x| x != 0.0),
        "physical block 0 fixture data must be nonzero for the leak tests to be meaningful"
    );
}
