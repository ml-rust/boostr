//! CPU implementation of AttentionOps and FlashAttentionOps
//!
//! AttentionOps delegates to impl_generic (Var-based autograd).
//! FlashAttentionOps uses standard O(N²) attention via numr Tensor ops
//! (no fused kernel — CPU fallback).

use super::flash_helpers::{standard_attention_bwd, standard_attention_fwd};
use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::{AttentionOps, FlashAttentionOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AttentionOps<CpuRuntime> for CpuClient {
    fn multi_head_attention(
        &self,
        q: &Var<CpuRuntime>,
        k: &Var<CpuRuntime>,
        v: &Var<CpuRuntime>,
        mask: Option<&Var<CpuRuntime>>,
        num_heads: usize,
    ) -> Result<Var<CpuRuntime>> {
        multi_head_attention_impl(self, q, k, v, mask, num_heads)
    }
}

impl FlashAttentionOps<CpuRuntime> for CpuClient {
    fn flash_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
        kv_seq_len: Option<usize>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        // If kv_seq_len override provided, narrow K/V to actual seq len first
        if let Some(seq_len) = kv_seq_len {
            let k_narrow = k.narrow(2, 0, seq_len)?;
            let v_narrow = v.narrow(2, 0, seq_len)?;
            let k_c = k_narrow.contiguous();
            let v_c = v_narrow.contiguous();
            return self.flash_attention_fwd(
                q,
                &k_c,
                &v_c,
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                window_size,
                None,
            );
        }

        // Fast path: fused decode attention for S_q=1 (single token generation)
        // Avoids all intermediate tensor allocations and GQA expansion
        let seq_len_q = q.shape()[2];
        if seq_len_q == 1
            && !causal
            && window_size == 0
            && q.dtype() == DType::F32
            && k.dtype() == DType::F32
            && v.dtype() == DType::F32
        {
            return super::decode_attention::fused_decode_attention(
                q,
                k,
                v,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        let _ = head_dim; // validated by shape
        standard_attention_fwd(self, q, k, v, causal, num_heads, num_kv_heads, window_size)
    }

    fn flash_attention_fwd_fp8(
        &self,
        _q: &Tensor<CpuRuntime>,
        _k: &Tensor<CpuRuntime>,
        _v: &Tensor<CpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _o_scale: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "FP8 Flash Attention is not supported on CPU".into(),
        })
    }

    fn flash_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        _head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        standard_attention_bwd(
            self,
            dout,
            q,
            k,
            v,
            output,
            lse,
            causal,
            num_heads,
            num_kv_heads,
            window_size,
        )
    }

    fn flash_attention_bwd_fp8(
        &self,
        _dout: &Tensor<CpuRuntime>,
        _q: &Tensor<CpuRuntime>,
        _k: &Tensor<CpuRuntime>,
        _v: &Tensor<CpuRuntime>,
        _output: &Tensor<CpuRuntime>,
        _lse: &Tensor<CpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _do_scale: f32,
        _o_scale: f32,
        _dq_scale: f32,
        _dk_scale: f32,
        _dv_scale: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "FP8 Flash Attention backward is not supported on CPU".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::{BinaryOps, ReduceOps, UnaryOps};

    fn rand_tensor(
        shape: &[usize],
        _client: &CpuClient,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        // Simple deterministic pseudo-random data
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device)
    }

    #[test]
    fn test_flash_fwd_output_shape() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (2, 4, 8, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_flash_fwd_causal() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 6, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_causal, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
            .unwrap();

        // Causal and full should differ (unless trivial inputs)
        let diff = client.sub(&out_causal, &out_full).unwrap();
        let abs_diff = client.abs(&diff).unwrap();
        let max_diff = client.max(&abs_diff, &[], false).unwrap();
        let max_val = max_diff.to_vec::<f32>()[0];
        assert!(
            max_val > 1e-6,
            "Causal and non-causal outputs should differ"
        );
    }

    #[test]
    fn test_flash_fwd_sliding_window() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 12, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_window, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 4, None)
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
            .unwrap();

        let ow = out_window.to_vec::<f32>();
        let of = out_full.to_vec::<f32>();
        let max_diff = ow
            .iter()
            .zip(of.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Sliding window should differ from full attention"
        );
    }

    #[test]
    fn test_flash_fwd_gqa() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, s, d) = (1, 8, 2, 4, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, nkv, s, d], &client, &device);
        let v = rand_tensor(&[b, nkv, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, nkv, d, false, 0, None)
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_flash_bwd_produces_gradients() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, h, s, d]);
        assert_eq!(dv.shape(), &[b, h, s, d]);

        // Gradients should be non-zero
        let dq_abs = client.abs(&dq).unwrap();
        let dq_sum = client.sum(&dq_abs, &[], false).unwrap();
        assert!(dq_sum.to_vec::<f32>()[0] > 1e-6);
    }

    #[test]
    fn test_flash_bwd_causal_gradients() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, true, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, h, s, d]);
        assert_eq!(dv.shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_flash_bwd_gqa_gradient_shapes() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, s, d) = (1, 8, 2, 4, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, nkv, s, d], &client, &device);
        let v = rand_tensor(&[b, nkv, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, nkv, d, false, 0, None)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, nkv, d, false, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, nkv, s, d]);
        assert_eq!(dv.shape(), &[b, nkv, s, d]);
    }

    #[test]
    fn test_var_flash_attention_autograd() {
        use crate::ops::autograd_attention::var_flash_attention;
        use numr::autograd::Var;

        let (_client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);

        let q_t = rand_tensor(&[b, h, s, d], &_client, &device);
        let k_t = rand_tensor(&[b, h, s, d], &_client, &device);
        let v_t = rand_tensor(&[b, h, s, d], &_client, &device);

        let q = Var::new(q_t, true);
        let k = Var::new(k_t, true);
        let v = Var::new(v_t, true);

        let out = var_flash_attention::<CpuRuntime>(&q, &k, &v, h, h, d, false, 0).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
        assert!(
            out.grad_fn().is_some(),
            "Output should have grad_fn when inputs require grad"
        );
    }
}
