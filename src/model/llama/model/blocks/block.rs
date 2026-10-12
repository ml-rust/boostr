//! LLaMA single transformer block.

use super::attention::LlamaAttention;
use super::moe::LlamaFfn;
use super::norm::BlockNorm;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::config::ResidualLayout;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::ops::traits::{KvCacheOps, KvSeqLen, PagedAttentionOps, RoPEPackedOps};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Single transformer block.
///
/// Every forward path splits the block in two halves around attention:
/// [`pre_attention`](Self::pre_attention) and
/// [`post_attention`](Self::post_attention). The residual layout lives in
/// the second half, so the cache-free, KV-cache, paged and graph paths
/// cannot disagree about it.
pub struct LlamaBlock<R: Runtime> {
    pub(crate) input_layernorm: BlockNorm<R>,
    pub(crate) self_attn: LlamaAttention<R>,
    /// Sequential: the pre-MLP norm of `h`. Sandwich (Gemma 2): the norm of
    /// the attention output. Parallel: the MLP norm of `x`. `None` for a
    /// shared-norm parallel block.
    pub(crate) post_attention_layernorm: Option<BlockNorm<R>>,
    /// Gemma 2 pre-MLP norm.
    pub(crate) pre_feedforward_layernorm: Option<BlockNorm<R>>,
    /// Gemma 2 post-MLP norm.
    pub(crate) post_feedforward_layernorm: Option<BlockNorm<R>>,
    pub(crate) mlp: LlamaFfn<R>,
    pub(crate) residual: ResidualLayout,
}

impl<R: Runtime<DType = DType>> LlamaBlock<R> {
    /// `(input_layernorm(x + prev), x + prev)`. A `prev_mlp_out` from the
    /// previous layer's deferred residual fuses into this norm.
    pub(crate) fn pre_attention<C>(
        &self,
        client: &C,
        x: &Var<R>,
        prev_mlp_out: Option<&Var<R>>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match prev_mlp_out {
            Some(prev) => self.input_layernorm.fused_add_forward(client, x, prev),
            None => Ok((self.input_layernorm.forward(client, x)?, x.clone())),
        }
    }

    fn post_norm(&self) -> Result<&BlockNorm<R>> {
        self.post_attention_layernorm
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "this block layout needs post_attention_layernorm, and the \
                         checkpoint has none"
                    .into(),
            })
    }

    /// `(h, mlp_out)` after attention: the block output is `h + mlp_out`,
    /// and the caller defers that add to the next layer.
    ///
    /// `fuse` takes the fused add+norm kernel, which bypasses autograd.
    pub(crate) fn post_attention<C>(
        &self,
        client: &C,
        x: &Var<R>,
        normed: &Var<R>,
        attn_out: &Var<R>,
        fuse: bool,
    ) -> Result<(Var<R>, Var<R>)>
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
        match self.residual {
            ResidualLayout::Sequential => {
                if let Some(pre_ff) = &self.pre_feedforward_layernorm {
                    // Gemma 2 sandwich: norm the attention output, add, then
                    // norm into and out of the MLP.
                    let attn_out = self.post_norm()?.forward(client, attn_out)?;
                    let (mlp_in, h) = if fuse {
                        pre_ff.fused_add_forward(client, x, &attn_out)?
                    } else {
                        pre_ff.residual_norm(client, x, &attn_out)?
                    };
                    let mut mlp_out = self.mlp.forward(client, &mlp_in)?;
                    if let Some(post_ff) = &self.post_feedforward_layernorm {
                        mlp_out = post_ff.forward(client, &mlp_out)?;
                    }
                    return Ok((h, mlp_out));
                }
                let norm = self.post_norm()?;
                let (mlp_in, h) = if fuse {
                    norm.fused_add_forward(client, x, attn_out)?
                } else {
                    norm.residual_norm(client, x, attn_out)?
                };
                Ok((h, self.mlp.forward(client, &mlp_in)?))
            }
            ResidualLayout::Parallel => {
                let h = var_add(x, attn_out, client).map_err(Error::Numr)?;
                let mlp_in = self.post_norm()?.forward(client, x)?;
                Ok((h, self.mlp.forward(client, &mlp_in)?))
            }
            ResidualLayout::ParallelSharedNorm => {
                let h = var_add(x, attn_out, client).map_err(Error::Numr)?;
                Ok((h, self.mlp.forward(client, normed)?))
            }
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
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let (h, mlp_out) = self.post_attention(client, x, &normed, &attn_out, false)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Forward with KV cache, accepting the previous layer's deferred MLP output.
    ///
    /// If `prev_mlp_out` is provided, fuses the residual add with this layer's
    /// input layernorm (1 kernel instead of 2). Returns `(h, mlp_out)` so the
    /// caller can defer the final add to the next layer.
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        prev_mlp_out: Option<&Var<R>>,
        rope: &RoPE<R>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<(Var<R>, Var<R>)>
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
        let (normed, x) = self.pre_attention(client, x, prev_mlp_out)?;
        let attn_out = self
            .self_attn
            .forward_with_kv_cache(client, &normed, rope, kv_cache, position)?;
        self.post_attention(client, &x, &normed, &attn_out, true)
    }

    /// Forward with paged KV cache, accepting the previous layer's deferred
    /// MLP output. `seq_len_k` sizes the gathered KV buffer; `kv_seq_len`
    /// narrows which of those gathered keys each row attends over (`None`
    /// keeps every row at `seq_len_k`). `rope_positions` gives each row its
    /// own RoPE position instead of the shared `position` scalar. See
    /// [`LlamaAttention::forward_with_paged_kv_cache`].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        prev_mlp_out: Option<&Var<R>>,
        rope: &RoPE<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        layer_idx: usize,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        position: usize,
        kv_seq_len: Option<KvSeqLen<'_, R>>,
        rope_positions: Option<&Tensor<R>>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R> + RoPEPackedOps<R>,
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
            + PagedAttentionOps<R>
            + DequantOps<R>,
    {
        let (normed, x) = self.pre_attention(client, x, prev_mlp_out)?;
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
            kv_seq_len,
            rope_positions,
        )?;
        self.post_attention(client, &x, &normed, &attn_out, true)
    }

    /// Whether this block is the plain Llama block that the CUDA graph
    /// paths implement.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn is_plain_llama(&self) -> bool {
        let plain_mlp = match &self.mlp {
            LlamaFfn::Dense(m) => {
                m.gate_proj.is_some() && m.activation == crate::model::config::MlpActivation::Silu
            }
            LlamaFfn::Moe(_) => true,
        };
        matches!(self.input_layernorm, BlockNorm::Rms(_))
            && matches!(self.post_attention_layernorm, Some(BlockNorm::Rms(_)))
            && self.pre_feedforward_layernorm.is_none()
            && self.residual == ResidualLayout::Sequential
            && self.self_attn.softcap.is_none()
            && self.self_attn.clip_qkv.is_none()
            && !self.self_attn.use_alibi
            && plain_mlp
    }
}

// ── Graph-mode forward (CUDA only) ───────────────────────────────────

#[cfg(feature = "cuda")]
impl LlamaBlock<numr::runtime::cuda::CudaRuntime> {
    /// Forward pass for CUDA graph capture / replay.
    /// Returns `(h, mlp_out)` to defer the residual add to the next layer.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        x: &Var<numr::runtime::cuda::CudaRuntime>,
        prev_mlp_out: Option<&Var<numr::runtime::cuda::CudaRuntime>>,
        cos_slice: &Var<numr::runtime::cuda::CudaRuntime>, // [1, head_dim]
        sin_slice: &Var<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &KvCache<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
    ) -> Result<(
        Var<numr::runtime::cuda::CudaRuntime>,
        Var<numr::runtime::cuda::CudaRuntime>,
    )>
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
        let (normed, x) = self.pre_attention(client, x, prev_mlp_out)?;

        let attn_out = self.self_attn.forward_graph_mode(
            client,
            &normed,
            cos_slice,
            sin_slice,
            kv_cache,
            device_scalars,
        )?;

        self.post_attention(client, &x, &normed, &attn_out, true)
    }
}
