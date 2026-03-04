//! Hybrid model mixing Llama attention and Mamba2 SSM blocks.

use super::blocks::{AttentionBlock, SsmBlock};
use crate::error::{Error, Result};
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::model::traits::ModelClient;
use crate::nn::{Embedding, Linear, RmsNorm, RoPE, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Hybrid model mixing attention (LLaMA-style) and SSM (Mamba2) blocks.
pub struct HybridModel<R: Runtime> {
    config: UniversalConfig,
    hybrid_config: HybridConfig,
    mamba_config: Mamba2Config,
    embed_tokens: Embedding<R>,
    blocks: Vec<HybridBlock<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
    rope: RoPE<R>,
}

/// A hybrid block is either an attention block or an SSM block.
enum HybridBlock<R: Runtime> {
    Attention(Box<AttentionBlock<R>>),
    Ssm(Box<SsmBlock<R>>),
}

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Load from a VarBuilder and UniversalConfig.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        config.validate()?;

        let hybrid_config = config
            .hybrid_layers
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "Hybrid model requires hybrid_layers config".into(),
            })?;
        hybrid_config.validate(config.num_layers)?;

        let attn_config = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "Hybrid model requires attention config for attention layers".into(),
        })?;

        let mamba_config = Mamba2Config::from_universal(config)?;
        mamba_config.validate()?;

        let hidden = config.hidden_size;
        let num_heads = attn_config.num_heads;
        let num_kv_heads = attn_config.kv_heads();
        let head_dim = attn_config.head_dim(hidden);

        // RoPE cache
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_config.rope_theta,
            attn_config.rope_scaling.as_ref(),
            vb.device(),
        );

        let mut model_vb = vb.pp("model");

        // Embedding
        let embed_weight = model_vb.take_tensor("embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight, false);

        // Build blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());

            if hybrid_config.is_ssm_layer(i) {
                // SSM block
                let norm = RmsNorm::new(
                    layer_vb.take_tensor("input_layernorm.weight")?,
                    config.rms_norm_eps as f32,
                    false,
                );
                let mut mixer_vb = layer_vb.pp("mixer");
                let mamba = Mamba2::from_varbuilder(&mamba_config, &mut mixer_vb, false)?;
                blocks.push(HybridBlock::Ssm(Box::new(SsmBlock { norm, mamba })));
            } else {
                // Attention block
                let input_layernorm = RmsNorm::new(
                    layer_vb.take_tensor("input_layernorm.weight")?,
                    config.rms_norm_eps as f32,
                    false,
                );

                let mut attn_vb = layer_vb.pp("self_attn");
                let q_proj = Linear::new(attn_vb.take_tensor("q_proj.weight")?, None, false);
                let k_proj = Linear::new(attn_vb.take_tensor("k_proj.weight")?, None, false);
                let v_proj = Linear::new(attn_vb.take_tensor("v_proj.weight")?, None, false);
                let o_proj = Linear::new(attn_vb.take_tensor("o_proj.weight")?, None, false);

                let post_attention_layernorm = RmsNorm::new(
                    layer_vb.take_tensor("post_attention_layernorm.weight")?,
                    config.rms_norm_eps as f32,
                    false,
                );

                let mut mlp_vb = layer_vb.pp("mlp");
                let gate_proj = Linear::new(mlp_vb.take_tensor("gate_proj.weight")?, None, false);
                let up_proj = Linear::new(mlp_vb.take_tensor("up_proj.weight")?, None, false);
                let down_proj = Linear::new(mlp_vb.take_tensor("down_proj.weight")?, None, false);

                blocks.push(HybridBlock::Attention(Box::new(AttentionBlock {
                    input_layernorm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    post_attention_layernorm,
                    gate_proj,
                    up_proj,
                    down_proj,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                })));
            }
        }

        // Final norm
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );

        // LM head
        let lm_head = if config.tie_word_embeddings {
            let embed_w = embed_tokens.weight().tensor().clone();
            Linear::new(embed_w, None, false)
        } else {
            Linear::new(vb.take_tensor("lm_head.weight")?, None, false)
        };

        Ok(Self {
            config: config.clone(),
            hybrid_config: hybrid_config.clone(),
            mamba_config,
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
        })
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config_parse() {
        let config = UniversalConfig {
            model_type: "hybrid".into(),
            vocab_size: 1000,
            hidden_size: 256,
            num_layers: 4,
            max_seq_len: 512,
            intermediate_size: None,
            rms_norm_eps: 1e-5,
            attention: Some(crate::model::config::AttentionConfig {
                num_heads: 4,
                num_kv_heads: None,
                head_dim: None,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                rope_theta: 10000.0,
                rope_scaling: None,
                sliding_window: None,
            }),
            ssm: Some(crate::model::config::SsmConfig {
                variant: "mamba2".into(),
                state_size: 16,
                num_heads: 2,
                head_dim: 256,
                expand: 2,
                conv_kernel: 4,
                chunk_size: 64,
                n_groups: 1,
                complex_rope: None,
                mimo_rank: None,
                use_conv: None,
            }),
            moe: None,
            hybrid_layers: Some(HybridConfig {
                ssm_layers: vec![0, 1],
                attention_layers: vec![2, 3],
            }),
            tie_word_embeddings: false,
        };

        config.validate().unwrap();
        assert_eq!(config.hybrid_layers.as_ref().unwrap().ssm_layers.len(), 2);
        assert_eq!(
            config
                .hybrid_layers
                .as_ref()
                .unwrap()
                .attention_layers
                .len(),
            2
        );
    }
}
