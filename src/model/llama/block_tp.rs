//! Tensor-parallel LLaMA block, attention, and MLP sub-modules.

use crate::distributed::tensor_parallel::{ColumnParallelLinear, RowParallelLinear};
use crate::error::{Error, Result};
use crate::inference::kv_cache::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::impl_generic::attention::rope::apply_rope_interleaved_impl;
use numr::autograd::{Var, var_add, var_mul, var_narrow, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

pub(super) struct LlamaBlockTp<R: Runtime> {
    pub(super) input_layernorm: RmsNorm<R>,
    pub(super) self_attn: LlamaAttentionTp<R>,
    pub(super) post_attention_layernorm: RmsNorm<R>,
    pub(super) mlp: LlamaMlpTp<R>,
}

pub(super) struct LlamaAttentionTp<R: Runtime> {
    pub(super) q_proj: ColumnParallelLinear<R>,
    pub(super) k_proj: ColumnParallelLinear<R>,
    pub(super) v_proj: ColumnParallelLinear<R>,
    pub(super) o_proj: RowParallelLinear<R>,
    pub(super) num_heads: usize,    // LOCAL heads (total / world_size)
    pub(super) num_kv_heads: usize, // LOCAL kv heads (total / world_size)
    pub(super) head_dim: usize,
}

pub(super) struct LlamaMlpTp<R: Runtime> {
    pub(super) gate_proj: ColumnParallelLinear<R>,
    pub(super) up_proj: ColumnParallelLinear<R>,
    pub(super) down_proj: RowParallelLinear<R>,
}

// ── LlamaBlockTp ────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaBlockTp<R> {
    pub(super) fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
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
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Forward pass with KV cache for autoregressive inference.
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
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self
            .self_attn
            .forward_with_kv_cache(client, &normed, rope, kv_cache, position)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }
}

// ── LlamaAttentionTp ────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaAttentionTp<R> {
    pub(super) fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
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

        // Q/K/V: column-parallel, each rank gets local heads
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Reshape to [B, S, local_H, D] then permute to [B, local_H, S, D]
        let q = numr::autograd::var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k =
            numr::autograd::var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
                .map_err(Error::Numr)?;
        let v =
            numr::autograd::var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
                .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q);
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_contiguous(&k);
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_contiguous(&v);

        // RoPE on Q and K
        let q = apply_rope_interleaved_impl(client, &q, rope.cos_cache(), rope.sin_cache())?;
        let k = apply_rope_interleaved_impl(client, &k, rope.cos_cache(), rope.sin_cache())?;

        // GQA repeat KV if needed (local heads)
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
            let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
            (k, v)
        } else {
            (k, v)
        };

        // Local attention (no communication)
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = numr::autograd::var_reshape(
            &attn_out,
            &[batch, seq_len, self.num_heads * self.head_dim],
        )
        .map_err(Error::Numr)?;

        // O projection (row-parallel → all-reduce)
        self.o_proj.forward(client, &attn_out)
    }

    /// Forward with KV cache for autoregressive inference.
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
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V: column-parallel, each rank gets local heads
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Reshape to [B, S, local_H, D] then permute to [B, local_H, S, D]
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
        let q = apply_rope_interleaved_impl(client, &q, &cos_offset, &sin_offset)?;
        let k = apply_rope_interleaved_impl(client, &k, &cos_offset, &sin_offset)?;

        // Update KV cache
        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;

        // Attention using full KV cache
        let kv_seq_len = kv_cache.seq_len();
        let is_prefill = seq_len > 1;
        let (attn_out, _lse) = client.flash_attention_fwd(
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            is_prefill,
            0,
            Some(kv_seq_len),
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // O projection (row-parallel → all-reduce)
        self.o_proj.forward(client, &attn_out)
    }
}

fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];
    let expanded = x.tensor().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}

// ── LlamaMlpTp ──────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaMlpTp<R> {
    pub(super) fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
        // down_proj is row-parallel → all-reduce
        self.down_proj.forward(client, &hidden)
    }
}
