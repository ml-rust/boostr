//! CPU implementation of AttentionOps and FlashAttentionOps
//!
//! AttentionOps delegates to impl_generic (Var-based autograd).
//! FlashAttentionOps uses standard O(N²) attention via numr Tensor ops
//! (no fused kernel — CPU fallback).

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::{
    StandardAttnConfig, multi_head_attention_impl, standard_attention_bwd, standard_attention_fwd,
};
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
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
            let k_c = k_narrow.contiguous()?;
            let v_c = v_narrow.contiguous()?;
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
        let cfg = StandardAttnConfig {
            num_heads,
            num_kv_heads,
            causal,
            window_size,
        };
        standard_attention_fwd(self, q, k, v, cfg)
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

// Split into `flash_tests.rs` to keep this file under the crate's
// `cpu/*.rs` line-count limit — still ordinary same-crate unit tests.
#[cfg(test)]
#[path = "flash_tests.rs"]
mod tests;
