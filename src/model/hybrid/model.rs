//! Hybrid model mixing Llama attention and Mamba2 SSM blocks.

use crate::error::{Error, Result};
use crate::inference::{KvCache, LayeredKvCache, LayeredSsmState, SsmState};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::model::traits::ModelClient;
use crate::nn::{Embedding, Linear, RmsNorm, RoPE, VarBuilder};
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

/// Attention block: pre-norm → multi-head attention → residual + pre-norm → MLP → residual
struct AttentionBlock<R: Runtime> {
    input_layernorm: RmsNorm<R>,
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    o_proj: Linear<R>,
    post_attention_layernorm: RmsNorm<R>,
    gate_proj: Linear<R>,
    up_proj: Linear<R>,
    down_proj: Linear<R>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// SSM block: pre-norm → Mamba2 → residual
struct SsmBlock<R: Runtime> {
    norm: RmsNorm<R>,
    mamba: Mamba2<R>,
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
}

// ── AttentionBlock forward ──────────────────────────────────────────

impl<R: Runtime<DType = DType>> AttentionBlock<R> {
    fn forward_with_kv_cache<C>(
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
    fn forward_inference<C>(
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

fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
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
