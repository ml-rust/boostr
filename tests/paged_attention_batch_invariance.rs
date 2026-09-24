//! A row's paged-decode result does not depend on how many rows share the
//! batched `paged_attention_fwd` call, and never reads another row's blocks.
//!
//! `paged_attention_fwd` takes one `seq_len_k` for the whole batch to size
//! the gathered contiguous KV buffer, and a per-row `KvSeqLen::PerRow(ends)`
//! to say where each row's own keys stop inside that buffer. Row `b` of a
//! batched call must be bit-identical to
//! the same row computed alone, with its own block table and its own
//! `seq_len_k`, at every length in a mixed-length batch.
//!
//! One row below is short enough that its block table is zero-padded past
//! its real blocks, and the padding entries point at physical block 0 —
//! which holds a different, longer row's live keys and values. Without the
//! per-row end narrowing the softmax to the row's own keys, the gather
//! would read that foreign block as extra keys and the row's output would
//! depend on data it does not own. This test pins that down: run alone, the
//! short row never touches block 0, so the batched and alone results must
//! agree bit for bit.

use boostr::ops::traits::KvSeqLen;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const BLOCK_SIZE: usize = 4;
/// Per-row sequence lengths: short, long, medium — batch max is 11.
const LENS: [usize; 3] = [3, 11, 7];
/// Physical block each row owns, in order, distinct across rows so no row's
/// real data ever overlaps another's. The long row (index 1) owns blocks
/// 0..3, so a padding entry of `0` reads straight into its live data.
const ROW_BLOCKS: [&[i32]; 3] = [&[3], &[0, 1, 2], &[4, 5]];
const TOTAL_BLOCKS: usize = 6;

fn cpu() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Deterministic values, distinct per index so a row that reads the wrong
/// slot changes the answer.
fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * ((i as f32) * 0.731 + seed).sin() + 0.1 * ((i as f32) * 0.017).cos())
        .collect()
}

/// The shared paged cache: `TOTAL_BLOCKS` physical blocks, `[num_blocks,
/// BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]`, each block's bytes distinct from
/// every other so a wrong physical index is visible in the output.
fn build_cache(device: &CpuDevice) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
    let cache_len = TOTAL_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k = Tensor::<CpuRuntime>::from_slice(
        &values(cache_len, 1.1),
        &[TOTAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM],
        device,
    )
    .expect("k cache");
    let v = Tensor::<CpuRuntime>::from_slice(
        &values(cache_len, 2.7),
        &[TOTAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM],
        device,
    )
    .expect("v cache");
    (k, v)
}

/// The batch's block table, zero-padded past each row's real blocks up to
/// the widest row (`max_num_blocks = 3`). The padding value `0` is a real
/// physical block — the long row's first one — chosen deliberately, not an
/// out-of-range sentinel the gather would trip over on its own.
fn build_block_table(device: &CpuDevice) -> Tensor<CpuRuntime> {
    let max_num_blocks = ROW_BLOCKS.iter().map(|r| r.len()).max().unwrap();
    let mut data = vec![0i32; LENS.len() * max_num_blocks];
    for (b, row) in ROW_BLOCKS.iter().enumerate() {
        for (i, &phys) in row.iter().enumerate() {
            data[b * max_num_blocks + i] = phys;
        }
        // Remaining slots stay `0`, the padding convention under test.
    }
    Tensor::<CpuRuntime>::from_slice(&data, &[LENS.len(), max_num_blocks], device)
        .expect("block table")
}

#[test]
fn batched_row_matches_its_own_alone_decode() {
    let (client, device) = cpu();
    let batch = LENS.len();
    let seq_len_k = *LENS.iter().max().unwrap();
    let (k_blocks, v_blocks) = build_cache(&device);
    let block_table = build_block_table(&device);

    let q_data = values(batch * NUM_HEADS * HEAD_DIM, 0.3);
    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[batch, NUM_HEADS, 1, HEAD_DIM], &device)
        .expect("q");

    let ends: Vec<i32> = LENS.iter().map(|&l| l as i32).collect();
    let ends_t = Tensor::<CpuRuntime>::from_slice(&ends, &[batch], &device).expect("ends");

    let (out, lse) = client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            NUM_HEADS,
            NUM_KV_HEADS,
            1,
            seq_len_k,
            HEAD_DIM,
            BLOCK_SIZE,
            false,
            Some(KvSeqLen::PerRow(&ends_t)),
        )
        .expect("batched paged_attention_fwd");
    let out_vec = out.to_vec::<f32>();
    let lse_vec = lse.to_vec::<f32>();

    let q_row = NUM_HEADS * HEAD_DIM;
    for b in 0..batch {
        let own_len = LENS[b];
        let own_blocks = ROW_BLOCKS[b];
        let q_b = Tensor::<CpuRuntime>::from_slice(
            &q_data[b * q_row..(b + 1) * q_row],
            &[1, NUM_HEADS, 1, HEAD_DIM],
            &device,
        )
        .expect("q row");
        let bt_b = Tensor::<CpuRuntime>::from_slice(own_blocks, &[1, own_blocks.len()], &device)
            .expect("block table row");

        let (out_b, lse_b) = client
            .paged_attention_fwd(
                &q_b,
                &k_blocks,
                &v_blocks,
                &bt_b,
                NUM_HEADS,
                NUM_KV_HEADS,
                1,
                own_len,
                HEAD_DIM,
                BLOCK_SIZE,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("alone paged_attention_fwd for row {b}: {e}"));
        let out_b_vec = out_b.to_vec::<f32>();
        let lse_b_vec = lse_b.to_vec::<f32>();

        for (i, (got, want)) in out_vec[b * q_row..(b + 1) * q_row]
            .iter()
            .zip(&out_b_vec)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "row {b} (len {own_len}) output element {i}: batched {got:e}, alone {want:e}"
            );
        }
        for (h, (got, want)) in lse_vec[b * NUM_HEADS..(b + 1) * NUM_HEADS]
            .iter()
            .zip(&lse_b_vec)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "row {b} (len {own_len}) head {h} LSE: batched {got:e}, alone {want:e}"
            );
        }
    }
}

/// Same batch, but with `kv_seq_len: None` (today's shared-bound call). This
/// pins the pre-existing behavior: every row is read out to the batch max,
/// so the short row's output MUST differ from its own alone decode — proving
/// the per-row test above is not vacuously equal for some unrelated reason.
#[test]
fn without_per_row_ends_the_short_row_disagrees_with_its_own_decode() {
    let (client, device) = cpu();
    let batch = LENS.len();
    let seq_len_k = *LENS.iter().max().unwrap();
    let (k_blocks, v_blocks) = build_cache(&device);
    let block_table = build_block_table(&device);

    let q_data = values(batch * NUM_HEADS * HEAD_DIM, 0.3);
    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[batch, NUM_HEADS, 1, HEAD_DIM], &device)
        .expect("q");

    let (out, _) = client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            NUM_HEADS,
            NUM_KV_HEADS,
            1,
            seq_len_k,
            HEAD_DIM,
            BLOCK_SIZE,
            false,
            None,
        )
        .expect("batched paged_attention_fwd without per-row ends");
    let out_vec = out.to_vec::<f32>();

    let b = 0; // the short row, whose block table is zero-padded into block 0
    let own_len = LENS[b];
    let own_blocks = ROW_BLOCKS[b];
    let q_row = NUM_HEADS * HEAD_DIM;
    let q_b = Tensor::<CpuRuntime>::from_slice(
        &q_data[b * q_row..(b + 1) * q_row],
        &[1, NUM_HEADS, 1, HEAD_DIM],
        &device,
    )
    .expect("q row");
    let bt_b = Tensor::<CpuRuntime>::from_slice(own_blocks, &[1, own_blocks.len()], &device)
        .expect("block table row");
    let (out_b, _) = client
        .paged_attention_fwd(
            &q_b,
            &k_blocks,
            &v_blocks,
            &bt_b,
            NUM_HEADS,
            NUM_KV_HEADS,
            1,
            own_len,
            HEAD_DIM,
            BLOCK_SIZE,
            false,
            None,
        )
        .expect("alone paged_attention_fwd");
    let out_b_vec = out_b.to_vec::<f32>();

    let differs = out_vec[b * q_row..(b + 1) * q_row]
        .iter()
        .zip(&out_b_vec)
        .any(|(got, want)| got.to_bits() != want.to_bits());
    assert!(
        differs,
        "test setup is degenerate: without per-row ends the short row's output should \
         differ from its own decode (it reads padding into block 0), but it matched"
    );
}

/// Sanity check that the padding target is not an accidentally degenerate
/// all-zero block: physical block 0 (the long row's first block, and the
/// zero-padding value every short row's spare block-table slots carry)
/// holds real, distinct, nonzero key and value bytes. If it were all zero,
/// a leak into it would be invisible and the tests above would prove
/// nothing.
#[test]
fn the_padding_target_block_holds_real_nonzero_data() {
    let cache_len = TOTAL_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let block0_elems = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k_block0 = &values(cache_len, 1.1)[..block0_elems];
    let v_block0 = &values(cache_len, 2.7)[..block0_elems];
    assert!(
        k_block0.iter().any(|&x| x != 0.0) && v_block0.iter().any(|&x| x != 0.0),
        "physical block 0 fixture data must be nonzero for the leak tests to be meaningful"
    );
}
