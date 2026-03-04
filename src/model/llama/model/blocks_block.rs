//! LLaMA single transformer block.

use super::attention::LlamaAttention;
use super::mlp::LlamaMlp;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use crate::ops::traits::{KvCacheOps, PagedAttentionOps};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Single transformer block
pub struct LlamaBlock<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: LlamaAttention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: LlamaMlp<R>,
}

impl<R: Runtime<DType = DType>> LlamaBlock<R> {
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
        // Pre-norm attention + residual
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        // Pre-norm MLP + residual
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
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
            + ConditionalOps<R>,
    {
        // Fuse previous layer's residual add with this layer's input norm
        let (normed, x) = if let Some(prev_mlp) = prev_mlp_out {
            self.input_layernorm
                .fused_add_forward(client, x, prev_mlp)?
        } else {
            let normed = self.input_layernorm.forward(client, x)?;
            (normed, x.clone())
        };

        let attn_out = self
            .self_attn
            .forward_with_kv_cache(client, &normed, rope, kv_cache, position)?;

        // Fused add + rms_norm: computes h = x + attn_out, then normed = rms_norm(h)
        let (normed, h) = self
            .post_attention_layernorm
            .fused_add_forward(client, &x, &attn_out)?;

        // MLP — defer the residual add to the next layer
        let mlp_out = self.mlp.forward(client, &normed)?;

        Ok((h, mlp_out))
    }

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
            + KvCacheOps<R>
            + PagedAttentionOps<R>,
    {
        let (normed, x) = if let Some(prev_mlp) = prev_mlp_out {
            self.input_layernorm
                .fused_add_forward(client, x, prev_mlp)?
        } else {
            let normed = self.input_layernorm.forward(client, x)?;
            (normed, x.clone())
        };

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
            .fused_add_forward(client, &x, &attn_out)?;

        let mlp_out = self.mlp.forward(client, &normed)?;
        Ok((h, mlp_out))
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
        let (normed, x) = if let Some(prev_mlp) = prev_mlp_out {
            self.input_layernorm
                .fused_add_forward(client, x, prev_mlp)?
        } else {
            let normed = self.input_layernorm.forward(client, x)?;
            (normed, x.clone())
        };

        let attn_out = self.self_attn.forward_graph_mode(
            client,
            &normed,
            cos_slice,
            sin_slice,
            kv_cache,
            device_scalars,
        )?;

        let (normed, h) = self
            .post_attention_layernorm
            .fused_add_forward(client, &x, &attn_out)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        Ok((h, mlp_out))
    }
}
