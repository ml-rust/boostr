//! CPU implementation of PagedAttentionOps
//!
//! Gathers paged KV blocks into contiguous tensors, then delegates to
//! the existing CPU FlashAttentionOps (standard O(N²) attention).

use crate::error::{Error, Result};
use crate::ops::traits::PagedAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Gather paged KV blocks into a contiguous [B, 1, S_k, D] tensor.
///
/// k_blocks: [num_blocks, block_size, head_dim]
/// block_table: [B, max_num_blocks] (i32)
/// Returns: [B, 1, seq_len_k, head_dim] (single head — caller repeats for GQA)
fn gather_paged_kv(
    kv_blocks: &Tensor<CpuRuntime>,
    block_table: &Tensor<CpuRuntime>,
    batch_size: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor<CpuRuntime>> {
    let kv_data = kv_blocks.to_vec::<f32>();
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];

    let mut out = vec![0.0f32; batch_size * seq_len_k * head_dim];

    for b in 0..batch_size {
        for t in 0..seq_len_k {
            let logical_block = t / block_size;
            let block_offset = t % block_size;
            let physical_block = bt_data[b * max_num_blocks + logical_block] as usize;
            let src_offset = physical_block * block_size * head_dim + block_offset * head_dim;
            let dst_offset = b * seq_len_k * head_dim + t * head_dim;
            out[dst_offset..dst_offset + head_dim]
                .copy_from_slice(&kv_data[src_offset..src_offset + head_dim]);
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        &[batch_size, 1, seq_len_k, head_dim],
        kv_blocks.device(),
    ))
}

impl PagedAttentionOps<CpuRuntime> for CpuClient {
    fn paged_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k_blocks: &Tensor<CpuRuntime>,
        v_blocks: &Tensor<CpuRuntime>,
        block_table: &Tensor<CpuRuntime>,
        num_heads: usize,
        _seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let batch_size = q.shape()[0];

        // Gather paged blocks into contiguous tensors [B, 1, S_k, D]
        let k_cont = gather_paged_kv(
            k_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let v_cont = gather_paged_kv(
            v_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        // Expand single KV head to num_heads via repeat: [B, 1, S_k, D] → [B, H, S_k, D]
        use numr::ops::ShapeOps;
        let k_expanded = self
            .repeat_interleave(&k_cont, num_heads, Some(1))
            .map_err(Error::Numr)?;
        let v_expanded = self
            .repeat_interleave(&v_cont, num_heads, Some(1))
            .map_err(Error::Numr)?;

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
        )
    }

    fn paged_attention_fwd_fp8(
        &self,
        _q: &Tensor<CpuRuntime>,
        _k_blocks: &Tensor<CpuRuntime>,
        _v_blocks: &Tensor<CpuRuntime>,
        _block_table: &Tensor<CpuRuntime>,
        _num_heads: usize,
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
        _seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let batch_size = q.shape()[0];

        // Gather paged blocks into contiguous tensors
        let k_cont = gather_paged_kv(
            k_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let v_cont = gather_paged_kv(
            v_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        // Expand KV to num_heads
        use numr::ops::ShapeOps;
        let k_expanded = self
            .repeat_interleave(&k_cont, num_heads, Some(1))
            .map_err(Error::Numr)?;
        let v_expanded = self
            .repeat_interleave(&v_cont, num_heads, Some(1))
            .map_err(Error::Numr)?;

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

        // Sum dk/dv back from [B, H, S_k, D] to [B, 1, S_k, D] then scatter to paged blocks
        use numr::ops::ReduceOps;
        let dk_summed = self.sum(&dk_exp, &[1], true).map_err(Error::Numr)?;
        let dv_summed = self.sum(&dv_exp, &[1], true).map_err(Error::Numr)?;

        // Scatter contiguous gradients back to paged block layout
        let dk_blocks = scatter_to_paged(
            &dk_summed,
            k_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;
        let dv_blocks = scatter_to_paged(
            &dv_summed,
            v_blocks,
            block_table,
            batch_size,
            seq_len_k,
            head_dim,
            block_size,
        )?;

        Ok((dq, dk_blocks, dv_blocks))
    }
}

/// Scatter contiguous gradients [B, 1, S_k, D] back to paged block layout.
fn scatter_to_paged(
    grad_cont: &Tensor<CpuRuntime>,
    kv_blocks_ref: &Tensor<CpuRuntime>,
    block_table: &Tensor<CpuRuntime>,
    batch_size: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor<CpuRuntime>> {
    let grad_data = grad_cont.to_vec::<f32>();
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];
    let block_shape = kv_blocks_ref.shape();

    let total_blocks = block_shape[0];
    let mut out = vec![0.0f32; total_blocks * block_size * head_dim];

    for b in 0..batch_size {
        for t in 0..seq_len_k {
            let logical_block = t / block_size;
            let block_offset = t % block_size;
            let physical_block = bt_data[b * max_num_blocks + logical_block] as usize;
            let dst_offset = physical_block * block_size * head_dim + block_offset * head_dim;
            let src_offset = b * seq_len_k * head_dim + t * head_dim;
            for d in 0..head_dim {
                out[dst_offset + d] += grad_data[src_offset + d];
            }
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        block_shape,
        kv_blocks_ref.device(),
    ))
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
        Tensor::<CpuRuntime>::from_slice(&data, shape, device)
    }

    #[test]
    fn test_paged_attention_fwd_shape() {
        let (client, device) = cpu_setup();
        let (b, h, s, d, bs): (usize, usize, usize, usize, usize) = (1, 4, 8, 16, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, d], &device);

        // Identity block table
        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table = Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device);

        let (out, lse) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
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
        let (b, h, s, d, bs): (usize, usize, usize, usize, usize) = (1, 2, 8, 8, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, d], &device);

        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table = Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device);

        let (out_causal, _) = client
            .paged_attention_fwd(&q, &k_blocks, &v_blocks, &block_table, h, s, s, d, bs, true)
            .unwrap();
        let (out_full, _) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
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
        let (b, h, s, d, bs): (usize, usize, usize, usize, usize) = (1, 2, 8, 8, 4);
        let num_blocks = s.div_ceil(bs);
        let total_blocks = b * num_blocks;

        let q = rand_tensor(&[b, h, s, d], &device);
        let k_blocks = rand_tensor(&[total_blocks, bs, d], &device);
        let v_blocks = rand_tensor(&[total_blocks, bs, d], &device);

        let bt_data: Vec<i32> = (0..b * num_blocks).map(|i| i as i32).collect();
        let block_table = Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, num_blocks], &device);

        let (out, lse) = client
            .paged_attention_fwd(
                &q,
                &k_blocks,
                &v_blocks,
                &block_table,
                h,
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
