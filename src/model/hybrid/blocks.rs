//! AttentionBlock and SsmBlock sub-modules for the hybrid model.

use crate::error::{Error, Result};
use crate::inference::{KvCache, SsmState};
use crate::model::mamba::mamba2::Mamba2;
use crate::model::traits::ModelClient;
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
        let q = var_contiguous(&q);
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_contiguous(&k);
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_contiguous(&v);

        // Apply RoPE with position offset
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        let q = apply_rope_impl(client, &q, &cos_offset, &sin_offset)?;
        let k = apply_rope_impl(client, &k, &cos_offset, &sin_offset)?;

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        kv_cache.update(k.tensor(), v.tensor())?;

        // Get full cached K/V for attention
        let (cached_k, cached_v) = kv_cache.get_kv()?;
        let cached_k = Var::new(cached_k.contiguous(), false);
        let cached_v = Var::new(cached_v.contiguous(), false);

        // GQA: repeat K/V heads to match Q heads if needed
        let (cached_k, cached_v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k_rep = repeat_kv(&cached_k, repeat).map_err(Error::Numr)?;
            let v_rep = repeat_kv(&cached_v, repeat).map_err(Error::Numr)?;
            (k_rep, v_rep)
        } else {
            (cached_k, cached_v)
        };

        // Multi-head attention (Q attends to full cached K/V)
        let attn_out =
            multi_head_attention_impl(client, &q, &cached_k, &cached_v, None, self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
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

// ── Helpers ─────────────────────────────────────────────────────────

pub(super) fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

pub(super) fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Reshape to [B, H_kv, 1, S, D] then broadcast to [B, H_kv, repeat, S, D]
    let expanded = x.tensor().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    // Reshape to [B, H_kv * repeat, S, D] — need contiguous for reshape after broadcast
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}
