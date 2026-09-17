//! CPU implementation of AttentionOps and FlashAttentionOps
//!
//! AttentionOps delegates to impl_generic (Var-based autograd).
//! FlashAttentionOps uses standard O(N²) attention via numr Tensor ops
//! (no fused kernel — CPU fallback). The forward lives in `flash_fwd.rs`;
//! the backward composes `standard_attention_bwd` here.

use super::flash_fwd;
use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::{
    StandardAttnConfig, multi_head_attention_impl, standard_attention_bwd,
};
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
use crate::ops::traits::{AttentionOps, AttnOutLayout, FlashAttentionOps};
use numr::autograd::Var;
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
        kv_start: Option<&Tensor<CpuRuntime>>,
        out_layout: AttnOutLayout,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        flash_fwd::flash_attention_fwd_cpu(
            self,
            q,
            k,
            v,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
            kv_seq_len,
            kv_start,
            out_layout,
        )
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

    fn flash_attention_fwd_fp8_kv(
        &self,
        q: &Tensor<CpuRuntime>,
        k_quant: &Tensor<CpuRuntime>,
        v_quant: &Tensor<CpuRuntime>,
        k_scales: &Tensor<CpuRuntime>,
        v_scales: &Tensor<CpuRuntime>,
        num_heads: usize,
        _head_dim: usize,
        causal: bool,
        per_token_scales: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        super::flash_fp8_kv::flash_attention_fwd_fp8_kv_impl(
            self,
            q,
            k_quant,
            v_quant,
            k_scales,
            v_scales,
            num_heads,
            causal,
            per_token_scales,
        )
    }

    fn flash_attention_fwd_int4_kv(
        &self,
        q: &Tensor<CpuRuntime>,
        k_quant: &Tensor<CpuRuntime>,
        v_quant: &Tensor<CpuRuntime>,
        k_scales: &Tensor<CpuRuntime>,
        k_zeros: &Tensor<CpuRuntime>,
        v_scales: &Tensor<CpuRuntime>,
        v_zeros: &Tensor<CpuRuntime>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        group_size: Int4GroupSize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        super::flash_int4_kv::flash_attention_fwd_int4_kv_impl(
            self, q, k_quant, v_quant, k_scales, k_zeros, v_scales, v_zeros, num_heads, head_dim,
            causal, group_size,
        )
    }

    fn flash_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        _lse: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        _head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let cfg = StandardAttnConfig {
            num_heads,
            num_kv_heads,
            causal,
            window_size,
        };
        standard_attention_bwd(self, dout, q, k, v, output, cfg)
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
    use numr::ops::{ReduceOps, UnaryOps};

    fn rand_tensor(
        shape: &[usize],
        _client: &CpuClient,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        // Simple deterministic pseudo-random data
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    #[test]
    fn test_flash_bwd_produces_gradients() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
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
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                true,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
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
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                nkv,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
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

        let out = var_flash_attention::<CpuRuntime>(
            &q,
            &k,
            &v,
            h,
            h,
            d,
            false,
            0,
            None,
            AttnOutLayout::HeadMajor,
        )
        .unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
        assert!(
            out.grad_fn().is_some(),
            "Output should have grad_fn when inputs require grad"
        );
    }
}
