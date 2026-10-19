//! [`Mla`] forward pass, with an optional document mask.

use super::model::Mla;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use crate::ops::RoPEOps;
use crate::ops::impl_generic::attention::document_mask::document_mask_bias;
use crate::ops::impl_generic::attention::mla::{
    scaled_dot_product_attention_impl, scaled_dot_product_attention_masked_impl,
};
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Mla<R> {
    /// Forward pass: [B, S, hidden] → [B, S, hidden]
    pub fn forward<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        self.forward_masked(client, hidden, None)
    }

    /// [`Mla::forward`] with an optional document mask over packed documents.
    ///
    /// `segment_ids` is `[B, S]` I32. Equal ids in one row mark one document.
    /// No token attends to a token of another document. Attention stays causal inside each document.
    /// `None` runs [`Mla::forward`] unchanged.
    ///
    /// RoPE needs no position reset per document. Rotated scores depend only on
    /// the offset `i - j`. The mask builds on the device, and scores must be F32.
    pub fn forward_with_segments<C>(
        &self,
        client: &C,
        hidden: &Var<R>,
        segment_ids: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let Some(segment_ids) = segment_ids else {
            return self.forward(client, hidden);
        };
        let shape = hidden.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "hidden",
                reason: format!("Mla: hidden must be [B, S, hidden], got {shape:?}"),
            });
        }
        let bias = document_mask_bias(client, segment_ids, shape[0], shape[1], true)?;
        self.forward_masked(client, hidden, Some(&bias))
    }

    /// Shared forward body. `bias` replaces the causal mask. `None` keeps the causal mask.
    fn forward_masked<C>(
        &self,
        client: &C,
        hidden: &Var<R>,
        bias: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let shape = hidden.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];
        let qk_dim = self.head_dim + self.rope_head_dim;

        // === Q path ===
        let q = if let Some(q_down) = &self.q_down {
            let q_latent = q_down.forward(client, hidden)?;
            let q_latent = if let Some(norm) = &self.q_norm {
                norm.forward(client, &q_latent)?
            } else {
                q_latent
            };
            self.q_up.forward(client, &q_latent)?
        } else {
            self.q_up.forward(client, hidden)?
        };

        // [B, S, num_heads * qk_dim] → [B, S, H, qk_dim] → [B, H, S, qk_dim]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, qk_dim]).map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;

        // Split Q into nope and pe
        let q_nope = var_narrow(&q, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let q_nope = var_contiguous(&q_nope)?;
        let q_pe = var_narrow(&q, 3, self.head_dim, self.rope_head_dim).map_err(Error::Numr)?;
        let q_pe = var_contiguous(&q_pe)?;

        // === KV path ===
        // Compress: [B, S, hidden] → [B, S, kv_lora_rank + rope_head_dim]
        let kv_compressed = self.kv_compress.forward(client, hidden)?;

        // Split: c_kv [B, S, kv_lora_rank], k_pe_raw [B, S, rope_head_dim]
        let c_kv = var_narrow(&kv_compressed, 2, 0, self.kv_lora_rank).map_err(Error::Numr)?;
        let c_kv = var_contiguous(&c_kv)?;
        let k_pe_raw = var_narrow(&kv_compressed, 2, self.kv_lora_rank, self.rope_head_dim)
            .map_err(Error::Numr)?;
        let k_pe_raw = var_contiguous(&k_pe_raw)?;

        // Normalize c_kv
        let c_kv = if let Some(norm) = &self.kv_norm {
            norm.forward(client, &c_kv)?
        } else {
            c_kv
        };

        // Decompress: [B, S, kv_lora_rank] → [B, S, num_heads * (head_dim + head_dim_v)]
        let kv = self.kv_decompress.forward(client, &c_kv)?;
        let kv = var_reshape(
            &kv,
            &[
                batch,
                seq_len,
                self.num_heads,
                self.head_dim + self.head_dim_v,
            ],
        )
        .map_err(Error::Numr)?;
        // → [B, H, S, head_dim + head_dim_v]
        let kv = var_permute(&kv, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let kv = var_contiguous(&kv)?;

        // Split K_nope and V
        let k_nope = var_narrow(&kv, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let k_nope = var_contiguous(&k_nope)?;
        let v = var_narrow(&kv, 3, self.head_dim, self.head_dim_v).map_err(Error::Numr)?;
        let v = var_contiguous(&v)?;

        // K_pe: [B, S, rope_head_dim] → [B, 1, S, rope_head_dim] → [B, H, S, rope_head_dim]
        let k_pe = var_reshape(&k_pe_raw, &[batch, 1, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_broadcast_to(&k_pe, &[batch, self.num_heads, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_contiguous(&k_pe)?;

        // Apply RoPE to q_pe and k_pe (decoupled)
        let q_pe = apply_rope_impl(client, &q_pe, self.rope.cos_cache(), self.rope.sin_cache())?;
        let k_pe = apply_rope_impl(client, &k_pe, self.rope.cos_cache(), self.rope.sin_cache())?;

        // Concatenate: Q = [q_nope, q_pe], K = [k_nope, k_pe]
        let q = var_cat(&[&q_nope, &q_pe], 3, client).map_err(Error::Numr)?;
        let k = var_cat(&[&k_nope, &k_pe], 3, client).map_err(Error::Numr)?;

        // Attention: Q,K [B, H, S, qk_dim], V [B, H, S, head_dim_v]
        let attn_out = match bias {
            None => scaled_dot_product_attention_impl(client, &q, &k, &v, self.scale, true)?,
            Some(bias) => scaled_dot_product_attention_masked_impl(
                client,
                &q,
                &k,
                &v,
                self.scale,
                Some(bias),
            )?,
        };

        // [B, H, S, head_dim_v] → [B, S, H, head_dim_v] → [B, S, H*head_dim_v]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(
            &attn_out,
            &[batch, seq_len, self.num_heads * self.head_dim_v],
        )
        .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}
