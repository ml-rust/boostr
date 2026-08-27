//! AttentionBlock and SsmBlock sub-modules for the hybrid model.

use crate::error::{Error, Result};
use crate::inference::{KvCache, SsmState};
use crate::model::attention_mask::causal_window_mask;
use crate::model::mamba::mamba2::Mamba2;
use crate::model::traits::ModelClient;
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::nn::{Linear, RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use numr::autograd::{Var, var_add, var_mul, var_narrow, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Attention block: pre-norm → multi-head attention → residual + pre-norm → MLP → residual
pub(super) struct AttentionBlock<R: Runtime> {
    pub(super) input_layernorm: RmsNorm<R>,
    pub(super) q_proj: Linear<R>,
    pub(super) k_proj: Linear<R>,
    pub(super) v_proj: Linear<R>,
    pub(super) o_proj: Linear<R>,
    pub(super) post_attention_layernorm: RmsNorm<R>,
    pub(super) gate_proj: Linear<R>,
    pub(super) up_proj: Linear<R>,
    pub(super) down_proj: Linear<R>,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    /// Use ALiBi instead of RoPE (Falcon v1, BLOOM, MPT).
    pub(super) use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited context).
    ///
    /// The window is INCLUSIVE of the current token: query at absolute position
    /// `p` may attend keys `j` with `p - sliding_window < j <= p`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub(super) sliding_window: usize,
}

/// SSM block: pre-norm → Mamba2 → residual
pub(super) struct SsmBlock<R: Runtime> {
    pub(super) norm: RmsNorm<R>,
    pub(super) mamba: Mamba2<R>,
}

// ── AttentionBlock forward ──────────────────────────────────────────

impl<R: Runtime<DType = DType>> AttentionBlock<R> {
    pub(super) fn forward_with_kv_cache<C>(
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
        // Pre-norm attention + residual
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.attention_forward(client, &normed, rope, kv_cache, position)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        // Pre-norm MLP + residual
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp_forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    fn attention_forward<C>(
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

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_contiguous(&k)?;
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_contiguous(&v)?;

        // Apply RoPE with position offset, or skip it for ALiBi models: ALiBi
        // carries positional information in the attention bias instead.
        let (q, k) = if self.use_alibi {
            (q, k)
        } else {
            let cos_offset =
                var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            let sin_offset =
                var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            let q = apply_rope_impl(client, &q, &cos_offset, &sin_offset)?;
            let k = apply_rope_impl(client, &k, &cos_offset, &sin_offset)?;
            (q, k)
        };

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        kv_cache.update(k.tensor(), v.tensor())?;

        // Get full cached K/V for attention
        let (cached_k, cached_v) = kv_cache.get_kv()?;
        let cached_k = Var::new(cached_k.contiguous()?, false);
        let cached_v = Var::new(cached_v.contiguous()?, false);

        // GQA: repeat K/V heads to match Q heads if needed
        let (cached_k, cached_v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k_rep = repeat_kv(&cached_k, repeat).map_err(Error::Numr)?;
            let v_rep = repeat_kv(&cached_v, repeat).map_err(Error::Numr)?;
            (k_rep, v_rep)
        } else {
            (cached_k, cached_v)
        };

        let sq = q.shape()[2];
        let sk = cached_k.shape()[2];
        let mask = self.attention_mask(
            client,
            batch,
            sq,
            sk,
            position,
            q.tensor().dtype(),
            q.tensor().device(),
        )?;

        // Multi-head attention (Q attends to full cached K/V)
        let attn_out = multi_head_attention_impl(
            client,
            &q,
            &cached_k,
            &cached_v,
            mask.as_ref(),
            self.num_heads,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

    /// Additive attention mask for one attention step.
    ///
    /// Always `Some`: the ALiBi branch returns the bias, and every other
    /// configuration returns a causal mask (windowed when `sliding_window > 0`).
    /// The `Option` is the caller's argument type, not a signal that masking is
    /// optional — an unmasked prefill lets every position attend to FUTURE
    /// tokens, which stays invisible to shape checks and still emits fluent
    /// text.
    ///
    /// `dtype` is the dtype of the attention scores this mask is added to. The
    /// additive-mask sites do not reconcile dtypes, so a stack running in
    /// BF16/F16 must state its dtype here.
    ///
    /// Row `i` is absolute position `position + i` and the cache holds
    /// `sk = position + sq` keys, so [`causal_window_mask`] derives the key
    /// offset from `sk - sq` and needs no extra argument.
    // Seven independent scalars, none derivable from another: `position` is the
    // ALiBi branch's own key offset, and `dtype`/`device` describe the scores
    // this mask is added to, not each other. Bundling them into a struct would
    // add a type whose only job is to be destructured back at the one call site.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_mask<C>(
        &self,
        client: &C,
        batch: usize,
        sq: usize,
        sk: usize,
        position: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Option<Var<R>>>
    where
        C: ModelClient<R>,
        R::Client: numr::ops::TypeConversionOps<R>,
    {
        if self.use_alibi {
            // ALiBi's own kernel writes the causal structure along with the
            // distance bias, so the sliding window does not apply here.
            let bias = Tensor::<R>::zeros(&[batch, self.num_heads, sq, sk], DType::F32, device)?;
            client.alibi_add_bias_causal(&bias, batch, self.num_heads, sq, sk, position)?;
            // The ALiBi kernel writes F32 slopes; cast once so the bias
            // carries the dtype of the scores it is added to.
            let bias = bias.to_dtype(dtype)?;
            Ok(Some(Var::new(bias, false)))
        } else {
            // ALWAYS masked, even with no sliding window. This branch is the
            // prefill/training path: without a causal mask every position
            // attends to FUTURE tokens, which makes the next-token objective
            // trivially cheatable and corrupts every prompt position at
            // inference. It stays invisible to shape checks and still emits
            // fluent text — the same failure that survived in the LLaMA
            // decoder until parity testing caught it.
            //
            // `window_size == 0` yields a pure causal mask, so this covers both
            // the windowed and unwindowed cases. On the decode path
            // (`sq == 1`, `sk == position + 1`) the shared builder's key offset
            // makes every cached key visible, as it must be.
            //
            // The window predicate alone does not mask the future — the shared
            // builder always applies causality alongside it.
            let mask = causal_window_mask(client, sq, sk, self.sliding_window, dtype, device)?;
            Ok(Some(Var::new(mask, false)))
        }
    }

    fn mlp_forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }
}

// ── SsmBlock forward ────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> SsmBlock<R> {
    pub(super) fn forward_inference<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: &mut SsmState<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        let normed = self.norm.forward(client, x)?;
        let out_tensor = self
            .mamba
            .forward_inference(client, normed.tensor(), state)?;
        let out = Var::new(out_tensor, false);
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
    }
}
