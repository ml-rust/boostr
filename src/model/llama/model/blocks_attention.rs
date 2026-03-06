//! LLaMA GQA attention block.

use super::helpers::{repeat_kv, var_contiguous};
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::{KvCacheOps, PagedAttentionOps};
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
}

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
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
        let q = var_contiguous(&q);
        let k = var_contiguous(&k);

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply fused RoPE to Q and K (single kernel per tensor on CUDA)
        let q = client.apply_rope_interleaved(&q, rope.cos_cache(), rope.sin_cache())?;
        let k = client.apply_rope_interleaved(&k, rope.cos_cache(), rope.sin_cache())?;

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
        let q = var_contiguous(&q);
        let k = var_contiguous(&k);

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply fused RoPE with position offset (single kernel per tensor on CUDA)
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        let q = client.apply_rope_interleaved(&q, &cos_offset, &sin_offset)?;
        let k = client.apply_rope_interleaved(&k, &cos_offset, &sin_offset)?;

        // V also needs to be contiguous for flash attention kernel
        let v = var_contiguous(&v);

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        // Uses fused in-place write (single kernel) instead of slice_assign
        // (which copies the entire cache buffer before writing the slice).
        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;

        // Pass raw full-capacity KV cache buffers with explicit seq_len.
        // Avoids narrow() + contiguous() which copied the entire cache every token.
        let kv_seq_len = kv_cache.seq_len();
        let is_prefill = seq_len > 1;
        let (attn_out, _lse) = client.flash_attention_fwd(
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            is_prefill, // causal during prefill (seq_len > 1), not needed for single-token decode
            0,          // no sliding window
            Some(kv_seq_len),
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
    pub fn forward_with_paged_kv_cache<C>(
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

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply RoPE with position offset
        let cos_offset = var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin_offset = var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        let q = client.apply_rope_interleaved(&q, &cos_offset, &sin_offset)?;
        let k = client.apply_rope_interleaved(&k, &cos_offset, &sin_offset)?;

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

// ── Graph-mode forward (CUDA only) ───────────────────────────────────

#[cfg(feature = "cuda")]
impl LlamaAttention<numr::runtime::cuda::CudaRuntime> {
    /// Graph-mode attention forward — all CUDA ops, stable addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        x: &Var<numr::runtime::cuda::CudaRuntime>,
        cos_slice: &Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &Var<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &KvCache<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
    ) -> Result<Var<numr::runtime::cuda::CudaRuntime>>
    where
        numr::runtime::cuda::CudaClient: crate::model::traits::ModelClient<numr::runtime::cuda::CudaRuntime>
            + numr::ops::TensorOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ScalarOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ReduceOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::IndexingOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ShapeOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ActivationOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::BinaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::UnaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::CompareOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ConditionalOps<numr::runtime::cuda::CudaRuntime>,
    {
        use crate::ops::cuda::attention::flash::decode_attention_graph_fwd;
        use crate::ops::cuda::attention::kv_insert::kv_insert;

        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = 1usize; // graph mode is always single-token decode

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

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply RoPE using the stable cos/sin slices (updated before each replay)
        let q = client.apply_rope_interleaved(&q, cos_slice, sin_slice)?;
        let k = client.apply_rope_interleaved(&k, cos_slice, sin_slice)?;

        // Insert K/V into the full-capacity cache at the device-side write_pos
        kv_insert(
            client,
            k.tensor(),
            v.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            device_scalars.write_pos_ptr(),
        )?;

        // Decode attention against the full-capacity cache with device-side seq_len_k
        let kv_capacity = kv_cache.capacity();
        let (attn_out, _lse) = decode_attention_graph_fwd(
            client,
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            device_scalars.seq_len_k_ptr(),
            kv_capacity,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}
