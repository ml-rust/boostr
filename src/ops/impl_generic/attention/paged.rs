//! Generic paged-attention backward.
//!
//! Paged KV lives in a block pool with logical shape
//! `[num_blocks, block_size, num_kv_heads, head_dim]`; a `[B, max_num_blocks]`
//! block table maps each sequence's logical blocks to physical ones. The
//! backward gathers the per-sequence KV into a dense `[B, num_kv_heads, S_k, D]`
//! tensor, runs the shared standard-attention backward, then scatters the KV
//! gradients back into the block pool. Gather/scatter run on-device via numr
//! `index_select` / `scatter_reduce`; only the small `block_table` is read
//! host-side to build the row-index map.

use super::flash_standard::{StandardAttentionClient, StandardAttnConfig, standard_attention_bwd};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScatterReduceOp};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The paged KV pool plus its block table.
pub struct PagedKv<'a, R: Runtime> {
    /// Key block pool `[num_blocks, block_size, num_kv_heads, head_dim]`.
    pub k_blocks: &'a Tensor<R>,
    /// Value block pool, same shape as `k_blocks`.
    pub v_blocks: &'a Tensor<R>,
    /// Block table `[B, max_num_blocks]` (i32).
    pub block_table: &'a Tensor<R>,
}

/// Geometry for paged attention backward.
#[derive(Debug, Clone, Copy)]
pub struct PagedAttnConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Total key/value sequence length.
    pub seq_len_k: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Tokens per KV block.
    pub block_size: usize,
    /// Apply a causal mask.
    pub causal: bool,
}

/// Row index, into the KV pool flattened to `[num_rows, head_dim]`, for every
/// `(b, h_kv, t)` in `[B, num_kv_heads, S_k]` order. A row is one `head_dim`
/// vector: `row = (physical_block · block_size + offset) · num_kv_heads + h_kv`.
fn paged_kv_row_indices(
    block_table: &[i32],
    batch_size: usize,
    num_kv_heads: usize,
    seq_len_k: usize,
    block_size: usize,
    max_num_blocks: usize,
) -> Vec<i64> {
    let mut rows = Vec::with_capacity(batch_size * num_kv_heads * seq_len_k);
    for b in 0..batch_size {
        for h_kv in 0..num_kv_heads {
            for t in 0..seq_len_k {
                let logical_block = t / block_size;
                let offset = t % block_size;
                let physical_block = block_table[b * max_num_blocks + logical_block] as usize;
                let row = (physical_block * block_size + offset) * num_kv_heads + h_kv;
                rows.push(row as i64);
            }
        }
    }
    rows
}

/// Paged attention backward. Returns `(dq, dk_blocks, dv_blocks)` where the KV
/// gradients have the same block-pool shape as `k_blocks` / `v_blocks`.
pub fn paged_attention_bwd_impl<R, C>(
    client: &C,
    dout: &Tensor<R>,
    q: &Tensor<R>,
    kv: &PagedKv<R>,
    output: &Tensor<R>,
    cfg: PagedAttnConfig,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R> + IndexingOps<R>,
{
    let batch_size = q.shape()[0];
    let max_num_blocks = kv.block_table.shape()[1];
    let d = cfg.head_dim;
    let nkv = cfg.num_kv_heads;
    let sk = cfg.seq_len_k;
    let device = q.device();

    let pool_rows: usize = kv.k_blocks.shape().iter().product::<usize>() / d;
    let n = batch_size * nkv * sk;

    // Build the gather/scatter row-index map from the (small) block table.
    let bt = kv.block_table.to_vec::<i32>();
    let rows = paged_kv_row_indices(&bt, batch_size, nkv, sk, cfg.block_size, max_num_blocks);
    let idx = Tensor::<R>::from_slice(&rows, &[n], device);

    // Gather KV blocks → dense [B, num_kv_heads, S_k, D].
    let k_pool = kv.k_blocks.reshape(&[pool_rows, d]).map_err(Error::Numr)?;
    let v_pool = kv.v_blocks.reshape(&[pool_rows, d]).map_err(Error::Numr)?;
    let k_dense = client
        .index_select(&k_pool, 0, &idx)
        .map_err(Error::Numr)?
        .reshape(&[batch_size, nkv, sk, d])
        .map_err(Error::Numr)?;
    let v_dense = client
        .index_select(&v_pool, 0, &idx)
        .map_err(Error::Numr)?
        .reshape(&[batch_size, nkv, sk, d])
        .map_err(Error::Numr)?;

    // Shared standard-attention backward (handles GQA expand/reduce).
    let (dq, dk_dense, dv_dense) = standard_attention_bwd(
        client,
        dout,
        q,
        &k_dense,
        &v_dense,
        output,
        StandardAttnConfig {
            num_heads: cfg.num_heads,
            num_kv_heads: nkv,
            causal: cfg.causal,
            window_size: 0,
        },
    )?;

    // Scatter KV gradients back into the block pool (accumulate onto zeros).
    let dk_blocks = scatter_kv_grad(
        client,
        &dk_dense,
        &idx,
        pool_rows,
        n,
        d,
        kv.k_blocks.shape(),
    )?;
    let dv_blocks = scatter_kv_grad(
        client,
        &dv_dense,
        &idx,
        pool_rows,
        n,
        d,
        kv.v_blocks.shape(),
    )?;

    Ok((dq, dk_blocks, dv_blocks))
}

/// Scatter-add a dense KV gradient `[B, num_kv_heads, S_k, D]` back into a
/// zeroed block pool shaped `pool_shape`, using the same row map as the gather.
fn scatter_kv_grad<R, C>(
    client: &C,
    grad_dense: &Tensor<R>,
    idx: &Tensor<R>,
    pool_rows: usize,
    n: usize,
    d: usize,
    pool_shape: &[usize],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R> + IndexingOps<R>,
{
    let grad_rows = grad_dense.reshape(&[n, d]).map_err(Error::Numr)?;
    // scatter_reduce is element-wise: index.shape == src.shape. Broadcast each
    // row's pool index across the `d` columns.
    let index2d = idx
        .reshape(&[n, 1])
        .map_err(Error::Numr)?
        .broadcast_to(&[n, d])
        .map_err(Error::Numr)?
        .contiguous()?;
    let dst = Tensor::<R>::zeros(&[pool_rows, d], DType::F32, grad_dense.device());
    let scattered = client
        .scatter_reduce(&dst, 0, &index2d, &grad_rows, ScatterReduceOp::Sum, true)
        .map_err(Error::Numr)?;
    scattered.reshape(pool_shape).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::attention::flash_standard::standard_attention_fwd;
    use crate::test_utils::cpu_setup;
    use numr::ops::IndexingOps;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn row_indices_match_layout() {
        // 2 batches, block_size 2, 2 kv heads, seq 3, max_num_blocks 2.
        // block_table: b0 -> [0,1], b1 -> [2,3]
        let bt = [0, 1, 2, 3];
        let rows = paged_kv_row_indices(&bt, 2, 2, 3, 2, 2);
        // For b0,h0: t0 ->(0*2+0)*2+0=0, t1->(0*2+1)*2+0=2, t2->(1*2+0)*2+0=4
        assert_eq!(&rows[0..3], &[0, 2, 4]);
        // b0,h1: +1 each -> 1,3,5
        assert_eq!(&rows[3..6], &[1, 3, 5]);
        // b1,h0: physical blocks 2,3 -> t0->(2*2+0)*2=8, t1->(2*2+1)*2=10, t2->(3*2+0)*2=12
        assert_eq!(&rows[6..9], &[8, 10, 12]);
    }

    #[test]
    fn paged_bwd_matches_dense_reference() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, d) = (2usize, 2usize, 1usize, 4usize);
        let block_size = 2usize;
        let seq_len_k = 4usize;
        let seq_len_q = 4usize;
        let max_num_blocks = 2usize;
        let num_blocks = 4usize; // b0 -> blocks 0,1 ; b1 -> blocks 2,3

        let nrows = num_blocks * block_size * nkv;
        let pool: Vec<f32> = (0..nrows * d).map(|i| (i as f32 * 0.017).sin()).collect();
        let k_blocks =
            Tensor::<CpuRuntime>::from_slice(&pool, &[num_blocks, block_size, nkv, d], &device);
        let vpool: Vec<f32> = (0..nrows * d).map(|i| (i as f32 * 0.023).cos()).collect();
        let v_blocks =
            Tensor::<CpuRuntime>::from_slice(&vpool, &[num_blocks, block_size, nkv, d], &device);
        let bt = Tensor::<CpuRuntime>::from_slice(&[0i32, 1, 2, 3], &[b, max_num_blocks], &device);

        let qd: Vec<f32> = (0..b * h * seq_len_q * d)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        let q = Tensor::<CpuRuntime>::from_slice(&qd, &[b, h, seq_len_q, d], &device);
        let dod: Vec<f32> = (0..b * h * seq_len_q * d)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let dout = Tensor::<CpuRuntime>::from_slice(&dod, &[b, h, seq_len_q, d], &device);

        let cfg = PagedAttnConfig {
            num_heads: h,
            num_kv_heads: nkv,
            seq_len_k,
            head_dim: d,
            block_size,
            causal: true,
        };

        // Reference: gather KV densely with the same index map, run standard fwd/bwd.
        let rows =
            paged_kv_row_indices(&[0, 1, 2, 3], b, nkv, seq_len_k, block_size, max_num_blocks);
        let n = rows.len();
        let idx = Tensor::<CpuRuntime>::from_slice(&rows, &[n], &device);
        let k_pool = k_blocks.reshape(&[nrows, d]).unwrap();
        let v_pool = v_blocks.reshape(&[nrows, d]).unwrap();
        let k_dense = client
            .index_select(&k_pool, 0, &idx)
            .unwrap()
            .reshape(&[b, nkv, seq_len_k, d])
            .unwrap();
        let v_dense = client
            .index_select(&v_pool, 0, &idx)
            .unwrap()
            .reshape(&[b, nkv, seq_len_k, d])
            .unwrap();
        let scfg = StandardAttnConfig {
            num_heads: h,
            num_kv_heads: nkv,
            causal: true,
            window_size: 0,
        };
        let (output, _lse) = standard_attention_fwd(&client, &q, &k_dense, &v_dense, scfg).unwrap();
        let (dq_ref, dk_ref_dense, _dv_ref_dense) =
            standard_attention_bwd(&client, &dout, &q, &k_dense, &v_dense, &output, scfg).unwrap();

        // Paged backward under test.
        let kv = PagedKv {
            k_blocks: &k_blocks,
            v_blocks: &v_blocks,
            block_table: &bt,
        };
        let (dq, dk_blocks, _dv_blocks) =
            paged_attention_bwd_impl(&client, &dout, &q, &kv, &output, cfg).unwrap();

        // dq must match the dense reference exactly.
        let dq_a: Vec<f32> = dq.to_vec();
        let dq_b: Vec<f32> = dq_ref.to_vec();
        for (x, y) in dq_a.iter().zip(dq_b.iter()) {
            assert!((x - y).abs() < 1e-5, "dq mismatch: {x} vs {y}");
        }

        // Gathering dk_blocks back with the same index map must reproduce the
        // dense dk reference (the scatter is the gather's inverse here, since the
        // block table maps each token to a distinct physical slot).
        let dk_pool = dk_blocks.reshape(&[nrows, d]).unwrap();
        let dk_regathered = client.index_select(&dk_pool, 0, &idx).unwrap();
        let dk_re: Vec<f32> = dk_regathered.to_vec();
        let dk_ref: Vec<f32> = dk_ref_dense.reshape(&[n, d]).unwrap().to_vec();
        for (x, y) in dk_re.iter().zip(dk_ref.iter()) {
            assert!((x - y).abs() < 1e-5, "dk mismatch: {x} vs {y}");
        }
    }
}
