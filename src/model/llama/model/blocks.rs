//! LLaMA building blocks: attention, MLP, and transformer layer.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::config::ModelConfig;
use crate::model::traits::ModelClient;
use crate::nn::{Linear, MaybeQuantLinear, RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::{KvCacheOps, PagedAttentionOps};
use numr::autograd::{Var, var_add, var_narrow, var_reshape, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Single transformer block
pub(super) struct LlamaBlock<R: Runtime> {
    pub(super) input_layernorm: RmsNorm<R>,
    pub(super) self_attn: LlamaAttention<R>,
    pub(super) post_attention_layernorm: RmsNorm<R>,
    pub(super) mlp: LlamaMlp<R>,
}

/// GQA attention with Q/K/V projections
pub(super) struct LlamaAttention<R: Runtime> {
    pub(super) q_proj: MaybeQuantLinear<R>,
    pub(super) k_proj: MaybeQuantLinear<R>,
    pub(super) v_proj: MaybeQuantLinear<R>,
    pub(super) o_proj: MaybeQuantLinear<R>,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
}

/// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
pub(super) struct LlamaMlp<R: Runtime> {
    pub(super) gate_proj: MaybeQuantLinear<R>,
    pub(super) up_proj: MaybeQuantLinear<R>,
    pub(super) down_proj: MaybeQuantLinear<R>,
}

// ── LlamaBlock ──────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaBlock<R> {
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
        // Pre-norm attention + residual
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        // Pre-norm MLP + residual
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

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
        // Pre-norm attention + residual (fused: 1 kernel instead of 2)
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self
            .self_attn
            .forward_with_kv_cache(client, &normed, rope, kv_cache, position)?;

        // Fused add + rms_norm: computes h = x + attn_out, then normed = rms_norm(h)
        let (normed, h) = self
            .post_attention_layernorm
            .fused_add_forward(client, x, &attn_out)?;

        // MLP + residual
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    pub(super) fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        layer_idx: usize,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
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
            + KvCacheOps<R>
            + PagedAttentionOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward_with_paged_kv_cache(
            client,
            &normed,
            rope,
            paged_cache,
            layer_idx,
            slot_mapping,
            block_table,
            seq_len_k,
            position,
        )?;

        let (normed, h) = self
            .post_attention_layernorm
            .fused_add_forward(client, x, &attn_out)?;

        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }
}

// ── LlamaAttention ──────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
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
        let q = var_contiguous(&q);
        let k = var_contiguous(&k);

        // Apply fused RoPE to Q and K (single kernel per tensor on CUDA)
        let q = client.apply_rope(&q, rope.cos_cache(), rope.sin_cache())?;
        let k = client.apply_rope(&k, rope.cos_cache(), rope.sin_cache())?;

        // GQA: repeat K/V heads to match Q heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
            let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
            (k, v)
        } else {
            (k, v)
        };

        // Multi-head attention
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

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
        let q = var_contiguous(&q);
        let k = var_contiguous(&k);

        // Apply fused RoPE with position offset (single kernel per tensor on CUDA)
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        let q = client.apply_rope(&q, &cos_offset, &sin_offset)?;
        let k = client.apply_rope(&k, &cos_offset, &sin_offset)?;

        // V also needs to be contiguous for flash attention kernel
        let v = var_contiguous(&v);

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        kv_cache.update(k.tensor(), v.tensor())?;

        // Get full cached K/V for attention
        let (cached_k, cached_v) = kv_cache.get_kv()?;
        let cached_k = cached_k.contiguous();
        let cached_v = cached_v.contiguous();

        // Flash attention handles GQA natively (no repeat_kv needed).
        // Single fused kernel: Q@K^T, scale, causal mask, softmax, @V
        let (attn_out, _lse) = client.flash_attention_fwd(
            q.tensor(),
            &cached_k,
            &cached_v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            false, // not causal for decode (Q has 1 token, sees all of KV cache)
            0,     // no sliding window
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        layer_idx: usize,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
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
            + KvCacheOps<R>
            + PagedAttentionOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V projections
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

        let q = var_contiguous(&q);
        let k = var_contiguous(&k);
        let v = var_contiguous(&v);

        // Apply RoPE with position offset
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        let q = client.apply_rope(&q, &cos_offset, &sin_offset)?;
        let k = client.apply_rope(&k, &cos_offset, &sin_offset)?;

        // Reshape K/V from [B, H_kv, S, D] to [B*S, H_kv, D] for paged cache update
        let k_flat = k
            .tensor()
            .permute(&[0, 2, 1, 3])? // [B, S, H_kv, D]
            .contiguous()
            .reshape(&[batch * seq_len, self.num_kv_heads, self.head_dim])?;
        let v_flat = v
            .tensor()
            .permute(&[0, 2, 1, 3])? // [B, S, H_kv, D]
            .contiguous()
            .reshape(&[batch * seq_len, self.num_kv_heads, self.head_dim])?;

        // Write K/V into paged cache using slot_mapping
        let layer_cache = paged_cache.layer(layer_idx);
        let rc = R::default_client(x.tensor().device());
        layer_cache.update(&k_flat, &v_flat, slot_mapping, &rc)?;

        // Paged attention: Q against full cached K/V
        let block_size = paged_cache.block_size();
        let _max_num_blocks = block_table.shape()[block_table.shape().len() - 1];
        let (attn_out, _lse) = rc.paged_attention_fwd(
            q.tensor(),
            layer_cache.k_cache(),
            layer_cache.v_cache(),
            block_table,
            self.num_heads,
            self.num_kv_heads,
            seq_len,
            seq_len_k,
            self.head_dim,
            block_size,
            false, // not causal for decode (Q sees all of KV cache)
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

// ── LlamaMlp ────────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaMlp<R> {
    /// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
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
        // Try fused SwiGLU path: single kernel for silu(gate_proj(x)) * up_proj(x)
        if let (MaybeQuantLinear::Quantized(gate_ql), MaybeQuantLinear::Quantized(up_ql)) =
            (&self.gate_proj, &self.up_proj)
        {
            if gate_ql.bias().is_none() && up_ql.bias().is_none() {
                let hidden_t = client.quant_swiglu(x.tensor(), gate_ql.weight(), up_ql.weight())?;
                let hidden = Var::new(hidden_t, false);
                return self.down_proj.forward(client, &hidden);
            }
        }

        // Fallback: batched gate+up + separate silu_mul
        let gate_up =
            MaybeQuantLinear::forward_batch(&[&self.gate_proj, &self.up_proj], client, x)?;
        let (gate, up) = (&gate_up[0], &gate_up[1]);
        let hidden = var_silu_mul(gate, up, client).map_err(Error::Numr)?;

        self.down_proj.forward(client, &hidden)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Make a Var contiguous (copies data if non-contiguous layout).
pub(super) fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

/// Repeat KV heads for GQA: [B, H_kv, S, D] -> [B, H_kv * repeat, S, D]
pub(super) fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Contiguous required: reshape needs contiguous layout, and inputs
    // (e.g. V after permute) may be strided.
    let expanded = x.tensor().contiguous().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}

/// Build a LlamaBlock for a given layer from a VarBuilder.
pub(super) fn build_block_from_varbuilder<R: Runtime<DType = DType>>(
    layer_vb: &mut crate::nn::VarBuilder<R>,
    config: &ModelConfig,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<LlamaBlock<R>> {
    let mut attn_vb = layer_vb.pp("self_attn");
    let q_proj = attn_vb.take_maybe_quant_linear("q_proj.weight", None)?;
    let k_proj = attn_vb.take_maybe_quant_linear("k_proj.weight", None)?;
    let v_proj = attn_vb.take_maybe_quant_linear("v_proj.weight", None)?;
    let o_proj = attn_vb.take_maybe_quant_linear("o_proj.weight", None)?;

    let mut mlp_vb = layer_vb.pp("mlp");
    let gate_proj = mlp_vb.take_maybe_quant_linear("gate_proj.weight", None)?;
    let up_proj = mlp_vb.take_maybe_quant_linear("up_proj.weight", None)?;
    let down_proj = mlp_vb.take_maybe_quant_linear("down_proj.weight", None)?;

    Ok(LlamaBlock {
        input_layernorm: RmsNorm::new(
            layer_vb.take_tensor("input_layernorm.weight")?,
            config.rms_norm_eps as f32,
            false,
        ),
        self_attn: LlamaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        },
        post_attention_layernorm: RmsNorm::new(
            layer_vb.take_tensor("post_attention_layernorm.weight")?,
            config.rms_norm_eps as f32,
            false,
        ),
        mlp: LlamaMlp {
            gate_proj,
            up_proj,
            down_proj,
        },
    })
}

/// Build a LlamaBlock initialized with zeros/ones for a given device.
pub(super) fn build_block_from_config<R: Runtime<DType = DType>>(
    config: &ModelConfig,
    device: &R::Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    dt: numr::dtype::DType,
) -> LlamaBlock<R> {
    let hidden = config.hidden_size;
    LlamaBlock {
        input_layernorm: RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        ),
        self_attn: LlamaAttention {
            q_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            k_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            v_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            o_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[hidden, num_heads * head_dim], dt, device),
                None,
                true,
            )),
            num_heads,
            num_kv_heads,
            head_dim,
        },
        post_attention_layernorm: RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        ),
        mlp: LlamaMlp {
            gate_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                None,
                true,
            )),
            up_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                None,
                true,
            )),
            down_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[hidden, intermediate], dt, device),
                None,
                true,
            )),
        },
    }
}
