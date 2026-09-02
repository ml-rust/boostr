//! CPU implementation of PagedAttentionOps
//!
//! Gathers paged KV blocks into contiguous tensors, then delegates to
//! the existing CPU FlashAttentionOps (standard O(N²) attention).
//!
//! The cache layout mapping lives in `paged_kv_layout`; this file owns the
//! attention itself.

use crate::error::{Error, Result};
use crate::ops::traits::PagedAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::paged_kv_layout::{expand_kv_heads, gather_paged_kv, reduce_kv_heads, scatter_to_paged};

/// Rejects a head count that is not a whole multiple of the KV head count.
///
/// Grouped-query attention maps query head `h` to KV head
/// `h / (num_heads / num_kv_heads)`, which is only well defined when the
/// division is exact.
fn validate_head_grouping(num_heads: usize, num_kv_heads: usize) -> Result<()> {
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads={num_heads} must be a multiple of num_kv_heads={num_kv_heads}"
            ),
        });
    }
    Ok(())
}

impl PagedAttentionOps<CpuRuntime> for CpuClient {
    fn paged_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k_blocks: &Tensor<CpuRuntime>,
        v_blocks: &Tensor<CpuRuntime>,
        block_table: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        _seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_head_grouping(num_heads, num_kv_heads)?;
        let batch_size = q.shape()[0];

        // Gather paged blocks into contiguous [B, num_kv_heads, S_k, D]
        let k_cont = gather_paged_kv(
            k_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let v_cont = gather_paged_kv(
            v_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        let k_expanded = expand_kv_heads(self, &k_cont, num_heads, num_kv_heads)?;
        let v_expanded = expand_kv_heads(self, &v_cont, num_heads, num_kv_heads)?;

        // Delegate to existing FlashAttentionOps
        use crate::ops::traits::FlashAttentionOps;
        self.flash_attention_fwd(
            q,
            &k_expanded,
            &v_expanded,
            num_heads,
            num_heads,
            head_dim,
            causal,
            0,
            None,
        )
    }

    fn paged_attention_fwd_fp8(
        &self,
        _q: &Tensor<CpuRuntime>,
        _k_blocks: &Tensor<CpuRuntime>,
        _v_blocks: &Tensor<CpuRuntime>,
        _block_table: &Tensor<CpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _seq_len_q: usize,
        _seq_len_k: usize,
        _head_dim: usize,
        _block_size: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _o_scale: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "FP8 paged attention is not supported on CPU".into(),
        })
    }

    fn paged_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k_blocks: &Tensor<CpuRuntime>,
        v_blocks: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        block_table: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        _seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_head_grouping(num_heads, num_kv_heads)?;
        let batch_size = q.shape()[0];

        // Gather paged blocks into contiguous [B, num_kv_heads, S_k, D]
        let k_cont = gather_paged_kv(
            k_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let v_cont = gather_paged_kv(
            v_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        let k_expanded = expand_kv_heads(self, &k_cont, num_heads, num_kv_heads)?;
        let v_expanded = expand_kv_heads(self, &v_cont, num_heads, num_kv_heads)?;

        // Delegate backward to FlashAttentionOps
        use crate::ops::traits::FlashAttentionOps;
        let (dq, dk_exp, dv_exp) = self.flash_attention_bwd(
            dout,
            q,
            &k_expanded,
            &v_expanded,
            output,
            lse,
            num_heads,
            num_heads,
            head_dim,
            causal,
            0,
        )?;

        // Sum the query heads that shared each KV head, then scatter back to pages.
        let dk_summed = reduce_kv_heads(
            self,
            &dk_exp,
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len_k,
            head_dim,
        )?;
        let dv_summed = reduce_kv_heads(
            self,
            &dv_exp,
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len_k,
            head_dim,
        )?;

        let dk_blocks = scatter_to_paged(
            &dk_summed,
            k_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let dv_blocks = scatter_to_paged(
            &dv_summed,
            v_blocks,
            block_table,
            batch_size,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        Ok((dq, dk_blocks, dv_blocks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    fn rand_tensor(
        shape: &[usize],
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    #[test]
    fn test_paged_attention_fwd_shape() {
        let (client, device) = cpu_setup();
        let (b, h, kvh, s, d, bs): (usize, usize, usize, usize, usize, usize) = (1, 4, 2, 8, 16, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);

        // Identity block table
        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table =
            Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device).unwrap();

        let (out, lse) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
                kvh,
                s,
                s,
                d,
                bs,
                false,
            )
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_paged_attention_fwd_causal() {
        let (client, device) = cpu_setup();
        let (b, h, kvh, s, d, bs): (usize, usize, usize, usize, usize, usize) = (1, 2, 2, 8, 8, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);

        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table =
            Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device).unwrap();

        let (out_causal, _) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
                kvh,
                s,
                s,
                d,
                bs,
                true,
            )
            .unwrap();
        let (out_full, _) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
                kvh,
                s,
                s,
                d,
                bs,
                false,
            )
            .unwrap();

        use numr::ops::{BinaryOps, ReduceOps, UnaryOps};
        let diff = client.sub(&out_causal, &out_full).unwrap();
        let abs_diff = client.abs(&diff).unwrap();
        let max_diff = client.max(&abs_diff, &[], false).unwrap();
        assert!(
            max_diff.to_vec::<f32>()[0] > 1e-6,
            "Causal and non-causal should differ"
        );
    }

    #[test]
    fn test_paged_attention_bwd_shapes() {
        let (client, device) = cpu_setup();
        let (b, h, kvh, s, d, bs): (usize, usize, usize, usize, usize, usize) = (1, 2, 2, 8, 8, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, kvh, d], &device);

        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table =
            Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device).unwrap();

        let (out, lse) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
                kvh,
                s,
                s,
                d,
                bs,
                false,
            )
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &device);

        let (dq, dk_blocks, dv_blocks) = client
            .paged_attention_bwd(
                &dout,
                &q,
                &k_blocks,
                &v_blocks,
                &out,
                &lse,
                &block_table,
                h,
                kvh,
                s,
                s,
                d,
                bs,
                false,
            )
            .unwrap();

        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk_blocks.shape(), k_blocks.shape());
        assert_eq!(dv_blocks.shape(), v_blocks.shape());

        // Gradients should be non-zero
        use numr::ops::{ReduceOps, UnaryOps};
        let dq_abs = client.abs(&dq).unwrap();
        let dq_sum = client.sum(&dq_abs, &[], false).unwrap();
        assert!(dq_sum.to_vec::<f32>()[0] > 1e-6, "dQ should be non-zero");
    }
}
