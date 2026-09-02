//! Backend parity for the CUDA split-KV paged decode attention path.
//!
//! `paged_attention_fwd` routes `seq_len_q == 1` to a decode kernel that walks
//! the KV cache through the block table. Like the contiguous decode path it
//! takes either a whole-sequence grid (one block per `(batch, Q head)`) or a
//! split-KV grid whose slices are merged by a combine pass; the shapes here are
//! chosen to land on each side of that choice and are stated per test.
//!
//! Every block table below is scrambled rather than identity, so a kernel that
//! ignores the indirection and reads pages in order diverges from CPU.
//!
//! Both the output and the log-sum-exp are compared: the paged decode kernel
//! previously returned an LSE it never wrote.

use super::helpers::*;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;

/// Physical block holding logical block `logical` of the whole run.
///
/// Reversing the order is a bijection, so every page is still used exactly
/// once, but no logical block sits at its own index.
fn scrambled_page(logical: usize, total: usize) -> i32 {
    (total - 1 - logical) as i32
}

/// Runs one paged decode shape on CPU and CUDA and asserts output and LSE agree.
#[allow(clippy::too_many_arguments)]
fn assert_paged_decode_parity(
    label: &str,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_k: usize,
    block_size: usize,
) {
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
        )
        .unwrap_or_else(|e| panic!("CPU paged decode failed for {label}: {e}"));
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    let cpu_lse_vec = cpu_lse.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::tensor::Tensor;

        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &q_shape, &cuda_device).unwrap();
        let kb = Tensor::from_slice(&k_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let vb = Tensor::from_slice(&v_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &bt_shape, &cuda_device).unwrap();

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
            )
            .unwrap_or_else(|e| panic!("CUDA paged decode failed for {label}: {e}"));

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

/// Too short to split: the whole-sequence kernel runs, one block per
/// `(batch, Q head)`. This is the path that must not change.
#[test]
fn paged_decode_short_sequence_takes_whole_sequence_path() {
    assert_paged_decode_parity("paged_decode_short", 1, 4, 4, 64, 64, 16);
}

/// Long enough to split, with the KV block count an exact multiple of the slice
/// count, so every slice owns the same number of pages.
#[test]
fn paged_decode_split_with_even_slices_parity() {
    assert_paged_decode_parity("paged_decode_even", 1, 2, 2, 128, 1024, 16);
}

/// KV length not a whole number of pages, so the last page is partly filled and
/// the slice covering it must stop at `seq_len_k`.
#[test]
fn paged_decode_split_with_partial_last_page_parity() {
    assert_paged_decode_parity("paged_decode_ragged", 1, 3, 3, 64, 1000, 16);
}

/// Grouped-query paged decode: several query heads share one KV head, so the
/// `kv_h` mapping is exercised alongside the slicing and the page indirection.
#[test]
fn paged_decode_split_grouped_query_parity() {
    assert_paged_decode_parity("paged_decode_gqa", 1, 8, 2, 128, 768, 32);
}

/// Large pages, so the device-fill target asks for more slices than the
/// sequence has KV blocks and the split count is clamped to the block count.
#[test]
fn paged_decode_split_clamped_to_page_count_parity() {
    assert_paged_decode_parity("paged_decode_clamped", 1, 4, 4, 64, 512, 256);
}

/// Batch above one widens the base grid and gives each sequence its own run of
/// pages, so both the batch and the slice index address the partials.
#[test]
fn paged_decode_split_batched_parity() {
    assert_paged_decode_parity("paged_decode_batched", 3, 4, 4, 64, 512, 16);
}

/// `seq_len_q == 2`, so the general paged kernel runs instead of the decode
/// fast path — over the same scrambled block table and multi-KV-head cache.
///
/// Regression guard for the layout contract itself: the CPU gather once read
/// the cache as `[num_blocks, block_size, head_dim]`, dropping the KV-head
/// axis, which agreed with CUDA only at `num_kv_heads == 1` — the sole case the
/// other paged tests covered.
#[test]
fn paged_prefill_scrambled_pages_parity() {
    let (batch, num_heads, num_kv_heads, head_dim) = (1usize, 4usize, 4usize, 64usize);
    let (seq_len_q, seq_len_k, block_size) = (2usize, 64usize, 16usize);
    let (cpu_client, cpu_device) = setup_cpu();

    let blocks_per_seq = seq_len_k.div_ceil(block_size);
    let total_blocks = batch * blocks_per_seq;
    let q_shape = [batch, num_heads, seq_len_q, head_dim];
    let cache_shape = [total_blocks, block_size, num_kv_heads, head_dim];
    let bt_shape = [batch, blocks_per_seq];

    let q = det_tensor(&q_shape, &cpu_device);
    let k_blocks = det_tensor(&cache_shape, &cpu_device);
    let v_blocks = det_tensor(&cache_shape, &cpu_device);
    let bt_data: Vec<i32> = (0..total_blocks)
        .map(|logical| scrambled_page(logical, total_blocks))
        .collect();
    let block_table = det_i32_tensor(&bt_data, &bt_shape, &cpu_device);

    let (cpu_out, _) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            num_heads,
            num_kv_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            block_size,
            false,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &q_shape, &cuda_device).unwrap();
        let kb = Tensor::from_slice(&k_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let vb = Tensor::from_slice(&v_blocks.to_vec::<f32>(), &cache_shape, &cuda_device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &bt_shape, &cuda_device).unwrap();
        let (out, _) = cuda_client
            .paged_attention_fwd(
                &q_c,
                &kb,
                &vb,
                &bt,
                num_heads,
                num_kv_heads,
                seq_len_q,
                seq_len_k,
                head_dim,
                block_size,
                false,
            )
            .unwrap();
        assert_parity_f32_tol(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "paged prefill scrambled pages CUDA vs CPU",
            1e-4,
            1e-6,
        );
    });
}
