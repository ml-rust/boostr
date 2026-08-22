//! LLaMA GQA attention block.

use super::helpers::{repeat_kv, var_contiguous};
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// GQA attention with Q/K/V projections
pub struct LlamaAttention<R: Runtime> {
    pub(crate) q_proj: MaybeQuantLinear<R>,
    pub(crate) k_proj: MaybeQuantLinear<R>,
    pub(crate) v_proj: MaybeQuantLinear<R>,
    pub(crate) o_proj: MaybeQuantLinear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Optional Q/K layer norms (Command-R, Cohere)
    pub(crate) q_norm: Option<crate::nn::RmsNorm<R>>,
    pub(crate) k_norm: Option<crate::nn::RmsNorm<R>>,
    /// Use ALiBi instead of RoPE (Falcon v1, BLOOM, MPT)
    pub(crate) use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited context).
    ///
    /// The window is INCLUSIVE of the current token: query `i` may attend keys
    /// `j` with `i - sliding_window < j <= i`, i.e. exactly `sliding_window`
    /// keys. This matches the flash-attention kernel contract in
    /// `ops/impl_generic/attention/flash_standard.rs`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub(crate) sliding_window: usize,
}

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    /// Additive causal mask, shape `[1, 1, sq, sk]`.
    ///
    /// Thin wrapper over [`crate::model::attention_mask::causal_window_mask`],
    /// which owns the masking rule (causality, the inclusive sliding window,
    /// and the `f32::MIN` fill) for every architecture. Only called with
    /// `sq == sk` here — the prefill/training path builds Q and K from the same
    /// activation — so the shared builder's key offset is `0`.
    fn causal_mask<C>(
        client: &C,
        sq: usize,
        sk: usize,
        window_size: usize,
        device: &R::Device,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
    {
        crate::model::attention_mask::causal_window_mask(client, sq, sk, window_size, device)
    }

    /// Apply optional Q/K layer norms (Command-R, Cohere).
    /// Input shape: [B, H, S, D] — norm is applied over the last dimension (head_dim).
    fn apply_qk_norms<C>(&self, client: &C, q: &Var<R>, k: &Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let q = match &self.q_norm {
            Some(norm) => norm.forward(client, q)?,
            None => q.clone(),
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(client, k)?,
            None => k.clone(),
        };
        Ok((q, k))
    }

    /// Apply RoPE to Q/K or skip for ALiBi models.
    fn apply_rotary_if_needed<C>(
        &self,
        client: &C,
        q: Var<R>,
        k: Var<R>,
        cos: &Var<R>,
        sin: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
    {
        if self.use_alibi {
            Ok((q, k))
        } else {
            let q = client.apply_rope(&q, cos, sin)?;
            let k = client.apply_rope(&k, cos, sin)?;
            Ok((q, k))
        }
    }

    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
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
            + ConditionalOps<R>,
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
        // V skips contiguous — matmul handles strided inputs via copy_strided.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply RoPE or skip for ALiBi models
        let (q, k) =
            self.apply_rotary_if_needed(client, q, k, rope.cos_cache(), rope.sin_cache())?;

        // GQA: repeat K/V heads to match Q heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
            let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
            (k, v)
        } else {
            (k, v)
        };

        // Additive attention mask.
        //
        // ⚠ This branch MUST supply a causal mask. It is the prefill/training
        // path: `Model::forward` and every training step run through it with
        // the whole sequence at once, so without a mask each position attends
        // to FUTURE tokens. That is not a subtle numerical issue — it makes the
        // next-token objective trivially cheatable during training and corrupts
        // every prompt position during inference. It also stays invisible to
        // shape checks and still emits fluent text, which is how it survived
        // until `tests/qwen3_parity.rs` measured against HuggingFace.
        //
        // The decode paths do not need this: `forward_with_kv_cache` either
        // builds the causal mask via `alibi_add_bias_causal` or delegates to
        // `flash_attention_fwd`, whose kernel applies causality from
        // `is_prefill`.
        let sq = q.shape()[2];
        let sk = k.shape()[2];
        let mask = self.prefill_mask(client, batch, sq, sk, q.tensor().device())?;
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, Some(&mask), self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

    /// The prefill/training additive mask: ALiBi bias or causal(+window).
    ///
    /// Split out so the masking rule is directly testable — a non-causal mask
    /// here is invisible to shape checks and still produces fluent text.
    pub(crate) fn prefill_mask<C>(
        &self,
        client: &C,
        batch: usize,
        sq: usize,
        sk: usize,
        device: &R::Device,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
    {
        Ok(if self.use_alibi {
            // MUST be the `_causal` kernel. `alibi_add_bias` writes a SYMMETRIC
            // `-slope * |qi - ki|` over the whole rectangle and masks nothing,
            // so using it here let every prefill position attend to FUTURE
            // tokens — the same defect the non-ALiBi path had until parity
            // testing caught it. `alibi_add_bias_causal` adds the same distance
            // bias and sets `ki > qi + position` to -inf.
            //
            // `position = 0`: this is the prefill/training path, so query `qi`
            // is at absolute position `qi`.
            let mask = Tensor::<R>::zeros(&[batch, self.num_heads, sq, sk], DType::F32, device);
            client.alibi_add_bias_causal(&mask, batch, self.num_heads, sq, sk, 0)?;
            Var::new(mask, false)
        } else {
            Var::new(
                Self::causal_mask(client, sq, sk, self.sliding_window, device)?,
                false,
            )
        })
    }

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
            + ConditionalOps<R>,
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
            );
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

// ── Paged-attention forward, graph-mode forward (CUDA only) ──────────
//
// Split out to sibling files under `attention/` to keep this file readable.
mod graph_mode;
mod paged;

#[cfg(test)]
mod tests;
