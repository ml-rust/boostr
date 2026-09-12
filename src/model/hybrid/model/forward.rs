//! [`HybridModel`] forward passes and accessors.

use super::build::{HybridBlock, HybridModel};
use crate::error::{Error, Result};
use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::Mamba2Config;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Forward pass for hybrid model with both KV cache and SSM state.
    pub fn forward_hybrid<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        ssm_state: &mut LayeredSsmState<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>,
    {
        // Embed tokens
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        let mut attn_idx = 0usize;
        let mut ssm_idx = 0usize;

        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                HybridBlock::Attention(attn_block) => {
                    let cache = kv_cache
                        .layer_mut(attn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "KV cache missing for attention layer {i} (attn_idx={attn_idx})"
                            ),
                        })?;
                    hidden = attn_block
                        .forward_with_kv_cache(client, &hidden, &self.rope, cache, position)?;
                    attn_idx += 1;
                }
                HybridBlock::Ssm(ssm_block) => {
                    let state = ssm_state
                        .layer_mut(ssm_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!("SSM state missing for layer {i} (ssm_idx={ssm_idx})"),
                        })?;
                    hidden = ssm_block.forward_inference(client, &hidden, state)?;
                    ssm_idx += 1;
                }
            }
        }

        // Final norm
        hidden = self.norm.forward(client, &hidden)?;

        // LM head
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
    }

    /// Contextualized hidden states for embedding extraction.
    ///
    /// Runs embed + all attention/SSM blocks + final norm (no `lm_head`) with
    /// fresh throwaway KV cache and SSM state. Returns shape `[B, S, hidden]`.
    pub fn forward_hidden<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
    ) -> Result<numr::autograd::Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>,
    {
        let shape = input_ids.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let device = input_ids.device();
        let dtype = self.embed_tokens.weight().tensor().dtype();

        let num_attn_layers = self
            .blocks
            .iter()
            .filter(|b| matches!(b, HybridBlock::Attention(_)))
            .count();
        let num_ssm_layers = self.blocks.len() - num_attn_layers;

        let attn_cfg = self
            .config
            .attention
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "Hybrid forward_hidden requires attention config".into(),
            })?;
        let head_dim = attn_cfg.head_dim(self.config.hidden_size);
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: attn_cfg.kv_heads(),
            initial_capacity: seq_len,
            max_seq_len: self.config.max_seq_len,
            head_dim,
            dtype,
        };
        let mut kv_cache = LayeredKvCache::<R>::new(num_attn_layers, &kv_config, device)?;
        let mut ssm_state =
            LayeredSsmState::<R>::new(num_ssm_layers, batch, &self.mamba_config, dtype, device)?;

        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        let mut attn_idx = 0usize;
        let mut ssm_idx = 0usize;
        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                HybridBlock::Attention(attn_block) => {
                    let cache = kv_cache
                        .layer_mut(attn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!("KV cache missing for attention layer {i}"),
                        })?;
                    hidden =
                        attn_block.forward_with_kv_cache(client, &hidden, &self.rope, cache, 0)?;
                    attn_idx += 1;
                }
                HybridBlock::Ssm(ssm_block) => {
                    let state = ssm_state
                        .layer_mut(ssm_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!("SSM state missing for layer {i}"),
                        })?;
                    hidden = ssm_block.forward_inference(client, &hidden, state)?;
                    ssm_idx += 1;
                }
            }
        }
        hidden = self.norm.forward(client, &hidden)?;
        Ok(hidden)
    }

    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    pub fn hybrid_config(&self) -> &HybridConfig {
        &self.hybrid_config
    }

    pub fn mamba_config(&self) -> &Mamba2Config {
        &self.mamba_config
    }

    /// Number of attention layers (for KV cache allocation).
    pub fn num_attention_layers(&self) -> usize {
        self.hybrid_config.attention_layers.len()
    }

    /// Number of SSM layers (for SSM state allocation).
    pub fn num_ssm_layers(&self) -> usize {
        self.hybrid_config.ssm_layers.len()
    }

    /// RoPE module (for cos/sin cache access in CUDA graph setup).
    pub fn rope(&self) -> &crate::nn::RoPE<R> {
        &self.rope
    }
}
