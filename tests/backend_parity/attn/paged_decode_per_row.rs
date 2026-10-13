//! Backend parity for a per-row key end on the CUDA paged decode path.
//!
//! `KvSeqLen::PerRow(ends)` gives row `b` its own last key, so one block table
//! serves rows of different lengths. The CPU client gathers the pages to the
//! batch's `seq_len_k` and narrows each row's softmax to `[0, ends[b])`; the
//! CUDA decode kernel clamps the row's page run and its last, partly filled
//! page to the same end and lets the slices past it fall out empty. The two
//! must agree on the output and on the log-sum-exp, which is the combine
//! pass's only other product.
//!
//! Both decode grid shapes run here: a block-table capacity of fewer than two
//! slice chunks keeps one block per `(batch, Q head)`, and a wider one cuts
//! each row's page run into slices. `src/ops/cuda/attention/decode_split.rs`
//! picks between them.
//!
//! Every block table below is scrambled rather than identity, so a kernel that
//! ignores the indirection and reads pages in order diverges from CPU.
//!
//! The per-row end reaches the paged decode kernel alone. At `seq_len_q > 1`
//! the CUDA paged path still refuses it, which the last test pins.

use super::helpers::*;
use boostr::ops::KvSeqLen;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;

/// Physical block holding logical block `logical` of the whole run.
///
/// Reversing the order is a bijection, so every page is still used exactly
/// once, but no logical block sits at its own index.
fn scrambled_page(logical: usize, total: usize) -> i32 {
    (total - 1 - logical) as i32
}

/// Runs one paged decode shape with per-row ends on CPU and CUDA and asserts
/// the output and LSE agree.
///
/// `seq_len_k` is the batch's block-table walk bound, which both backends read
/// as the capacity. `ends[b]` is row `b`'s last key, at or below it.
#[allow(clippy::too_many_arguments)]
fn assert_paged_per_row_decode_parity(
    label: &str,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_k: usize,
    block_size: usize,
    ends: &[i32],
) {
    let batch = ends.len();
    let (cpu_client, cpu_device) = setup_cpu();

    let blocks_per_seq = seq_len_k.div_ceil(block_size);
    let total_blocks = batch * blocks_per_seq;

    let q_shape = [batch, num_heads, 1, head_dim];
    let cache_shape = [total_blocks, block_size, num_kv_heads, head_dim];
    let bt_shape = [batch, blocks_per_seq];

    let q = det_tensor(&q_shape, &cpu_device);
    let k_blocks = det_tensor(&cache_shape, &cpu_device);
    let v_blocks = det_tensor(&cache_shape, &cpu_device);

    let bt_data: Vec<i32> = (0..total_blocks)
        .map(|logical| scrambled_page(logical, total_blocks))
        .collect();
    let block_table = det_i32_tensor(&bt_data, &bt_shape, &cpu_device);
    let cpu_ends = det_i32_tensor(ends, &[batch], &cpu_device);

    let (cpu_out, cpu_lse) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            num_heads,
            num_kv_heads,
            1,
            seq_len_k,
            head_dim,
            block_size,
            false,
            Some(KvSeqLen::PerRow(&cpu_ends)),
        )
        .unwrap_or_else(|e| panic!("CPU per-row paged decode failed for {label}: {e}"));
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    let cpu_lse_vec = cpu_lse.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::tensor::Tensor;

        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &q_shape, &cuda_device).unwrap();
        let kb = Tensor::from_slice(&k_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let vb = Tensor::from_slice(&v_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &bt_shape, &cuda_device).unwrap();
        let cuda_ends = Tensor::from_slice(ends, &[batch], &cuda_device).unwrap();

        let (cuda_out, cuda_lse) = cuda_client
            .paged_attention_fwd(
                &q_c,
                &kb,
                &vb,
                &bt,
                num_heads,
                num_kv_heads,
                1,
                seq_len_k,
                head_dim,
                block_size,
                false,
                Some(KvSeqLen::PerRow(&cuda_ends)),
            )
            .unwrap_or_else(|e| panic!("CUDA per-row paged decode failed for {label}: {e}"));

        assert_parity_f32_tol(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            &format!("{label} output CUDA vs CPU"),
            1e-4,
            1e-6,
        );
        assert_parity_f32_tol(
            &cuda_lse.to_vec::<f32>(),
            &cpu_lse_vec,
            &format!("{label} lse CUDA vs CPU"),
            1e-4,
            1e-6,
        );
    });
}

/// Ragged ends on the whole-sequence grid: three pages of capacity is one
/// slice chunk, so one block per `(batch, Q head)` walks each row's own pages.
/// Two ends land mid-page, so the partial-page clamp is a per-row value.
#[test]
fn paged_per_row_ends_whole_sequence_parity() {
    assert_paged_per_row_decode_parity("paged_per_row_short", 4, 4, 64, 48, 16, &[48, 5, 20, 33]);
}

/// Ragged ends on the split grid: each row cuts its own page run into slices
/// and the slices past its end come out empty.
#[test]
fn paged_per_row_ends_split_parity() {
    assert_paged_per_row_decode_parity(
        "paged_per_row_split",
        8,
        2,
        128,
        1024,
        16,
        &[1024, 777, 320, 33],
    );
}

/// Grouped-query paged decode with ragged ends: several query heads share one
/// KV head, so the `kv_h` mapping runs alongside the per-row slicing.
#[test]
fn paged_per_row_ends_grouped_query_parity() {
    assert_paged_per_row_decode_parity(
        "paged_per_row_gqa",
        8,
        2,
        64,
        768,
        32,
        &[768, 500, 129, 40],
    );
}

/// Ends far below the capacity: the grid is sized from the block table, so most
/// of its slices do no work and must not reach the combine.
#[test]
fn paged_per_row_ends_far_below_capacity_parity() {
    assert_paged_per_row_decode_parity(
        "paged_per_row_sparse_capacity",
        8,
        2,
        64,
        1024,
        16,
        &[60, 41, 17, 3],
    );
}

/// The per-row end reaches the paged decode kernel alone; the tiled paged
/// kernels read one key length for the whole batch and refuse it.
#[cfg(feature = "cuda")]
#[test]
fn paged_per_row_ends_above_one_query_row_is_refused() {
    with_cuda_backend(|client, device| {
        use numr::tensor::Tensor;

        let (_, cpu_device) = setup_cpu();
        let (batch, num_heads, head_dim, block_size) = (2usize, 4usize, 64usize, 16usize);
        let (seq_len_q, seq_len_k) = (2usize, 32usize);
        let blocks_per_seq = seq_len_k.div_ceil(block_size);
        let total_blocks = batch * blocks_per_seq;

        let q_shape = [batch, num_heads, seq_len_q, head_dim];
        let cache_shape = [total_blocks, block_size, num_heads, head_dim];
        let q_vec = det_tensor(&q_shape, &cpu_device).to_vec::<f32>();
        let cache_vec = det_tensor(&cache_shape, &cpu_device).to_vec::<f32>();
        let bt_data: Vec<i32> = (0..total_blocks)
            .map(|logical| scrambled_page(logical, total_blocks))
            .collect();

        let q = Tensor::from_slice(&q_vec, &q_shape, &device).unwrap();
        let kb = Tensor::from_slice(&cache_vec, &cache_shape, &device).unwrap();
        let vb = Tensor::from_slice(&cache_vec, &cache_shape, &device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &[batch, blocks_per_seq], &device).unwrap();
        let ends = Tensor::from_slice(&[32i32, 17], &[batch], &device).unwrap();

        let result = client.paged_attention_fwd(
            &q,
            &kb,
            &vb,
            &bt,
            num_heads,
            num_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            block_size,
            false,
            Some(KvSeqLen::PerRow(&ends)),
        );
        assert!(
            result.is_err(),
            "seq_len_q > 1 with a per-row key end must be refused on the CUDA paged path"
        );
    });
}
