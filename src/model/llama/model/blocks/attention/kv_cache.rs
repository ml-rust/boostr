//! Incremental forward against a contiguous `KvCache` (flash path, ALiBi path).

use super::super::helpers::{repeat_kv, var_contiguous};
use super::LlamaAttention;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V projections (batched: quantize activation once for all 3)
        let qkv = MaybeQuantLinear::forward_batch(
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            client,
            x,
        )?;
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        // Reshape to [B, S, H, D] then permute to [B, H, S, D]
        let q = var_reshape(q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // Contiguous Q/K needed because fused RoPE kernel assumes contiguous layout.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply RoPE or skip for ALiBi models
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let (q, k) = self.apply_rotary_if_needed(client, q, k, &cos_offset, &sin_offset)?;

        // V also needs to be contiguous for flash attention kernel
        let v = var_contiguous(&v)?;

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;

        let kv_seq_len = kv_cache.seq_len();
        let attn_out = if self.use_alibi {
            // ALiBi: use generic attention with bias (no flash attention)
            let k_full = Var::new(
                kv_cache
                    .k_cache_raw()
                    .narrow(2, 0, kv_seq_len)
                    .map_err(Error::Numr)?
                    .contiguous()?,
                false,
            );
            let v_full = Var::new(
                kv_cache
                    .v_cache_raw()
                    .narrow(2, 0, kv_seq_len)
                    .map_err(Error::Numr)?
                    .contiguous()?,
                false,
            );
            // Repeat KV heads for GQA
            let (k_full, v_full) = if self.num_kv_heads < self.num_heads {
                let repeat = self.num_heads / self.num_kv_heads;
                let k_rep = repeat_kv(&k_full, repeat).map_err(Error::Numr)?;
                let v_rep = repeat_kv(&v_full, repeat).map_err(Error::Numr)?;
                (k_rep, v_rep)
            } else {
                (k_full, v_full)
            };
            // Build ALiBi + causal mask (single backend-specific kernel call)
            let sq = seq_len;
            let sk = kv_seq_len;
            let mask = Tensor::<R>::zeros(
                &[batch, self.num_heads, sq, sk],
                DType::F32,
                q.tensor().device(),
            )?;
            client.alibi_add_bias_causal(&mask, batch, self.num_heads, sq, sk, position)?;
            let mask_var = Var::new(mask, false);
            multi_head_attention_impl(
                client,
                &q,
                &k_full,
                &v_full,
                Some(&mask_var),
                self.num_heads,
            )?
        } else {
            let is_prefill = seq_len > 1;
            let (out, _lse) = client.flash_attention_fwd(
                q.tensor(),
                kv_cache.k_cache_raw(),
                kv_cache.v_cache_raw(),
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                is_prefill,
                self.sliding_window,
                Some(kv_seq_len),
            )?;
            Var::new(out, false)
        };

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}
