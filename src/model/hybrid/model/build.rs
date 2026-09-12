//! The [`HybridModel`] type and its builder: walks `hybrid_layers` and
//! constructs an attention or Mamba2 block for each layer index.

use super::super::blocks::{AttentionBlock, SsmBlock};
use crate::error::{Error, Result};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::nn::{Embedding, Linear, RmsNorm, RoPE, VarBuilder};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// Hybrid model mixing attention (LLaMA-style) and SSM (Mamba2) blocks.
pub struct HybridModel<R: Runtime> {
    pub(super) config: UniversalConfig,
    pub(super) hybrid_config: HybridConfig,
    pub(super) mamba_config: Mamba2Config,
    pub(super) embed_tokens: Embedding<R>,
    pub(super) blocks: Vec<HybridBlock<R>>,
    pub(super) norm: RmsNorm<R>,
    pub(super) lm_head: Linear<R>,
    pub(super) rope: RoPE<R>,
}

/// A hybrid block is either an attention block or an SSM block.
pub(super) enum HybridBlock<R: Runtime> {
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
        let use_alibi = attn_config.use_alibi;
        // `0` is the disabled sentinel; `Some(0)` is not a zero-width window.
        let sliding_window = attn_config.sliding_window();

        // RoPE cache
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_config.rope_theta,
            attn_config.rope_scaling.as_ref(),
            vb.device(),
        )?;

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
                    use_alibi,
                    sliding_window,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::{AttentionConfig, SsmConfig};
    use crate::nn::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
                use_alibi: false,
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
            grow_vocab: false,
            vision: None,
            audio: None,
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

    fn hybrid_config(sliding_window: Option<usize>, use_alibi: bool) -> UniversalConfig {
        UniversalConfig {
            model_type: "hybrid".into(),
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 2,
            max_seq_len: 16,
            intermediate_size: Some(16),
            rms_norm_eps: 1e-5,
            attention: Some(AttentionConfig {
                num_heads: 2,
                num_kv_heads: None,
                head_dim: None,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                rope_theta: 10000.0,
                rope_scaling: None,
                sliding_window,
                use_alibi,
            }),
            ssm: Some(SsmConfig {
                variant: "mamba2".into(),
                state_size: 4,
                num_heads: 2,
                head_dim: 8,
                expand: 2,
                conv_kernel: 4,
                chunk_size: 4,
                n_groups: 1,
                complex_rope: None,
                mimo_rank: None,
                use_conv: None,
            }),
            moe: None,
            hybrid_layers: Some(HybridConfig {
                ssm_layers: vec![0],
                attention_layers: vec![1],
            }),
            tie_word_embeddings: true,
            grow_vocab: false,
            vision: None,
            audio: None,
        }
    }

    /// Weight names and shapes matching [`hybrid_config`]: layer 0 is Mamba2,
    /// layer 1 is attention.
    fn weight_shapes() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight", vec![16, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.mixer.in_proj.weight", vec![42, 8]),
            ("model.layers.0.mixer.conv1d.weight", vec![24, 1, 4]),
            ("model.layers.0.mixer.out_proj.weight", vec![8, 16]),
            ("model.layers.0.mixer.A_log", vec![2]),
            ("model.layers.0.mixer.dt_bias", vec![2]),
            ("model.layers.0.mixer.D", vec![2]),
            ("model.layers.1.input_layernorm.weight", vec![8]),
            ("model.layers.1.self_attn.q_proj.weight", vec![8, 8]),
            ("model.layers.1.self_attn.k_proj.weight", vec![8, 8]),
            ("model.layers.1.self_attn.v_proj.weight", vec![8, 8]),
            ("model.layers.1.self_attn.o_proj.weight", vec![8, 8]),
            ("model.layers.1.post_attention_layernorm.weight", vec![8]),
            ("model.layers.1.mlp.gate_proj.weight", vec![16, 8]),
            ("model.layers.1.mlp.up_proj.weight", vec![16, 8]),
            ("model.layers.1.mlp.down_proj.weight", vec![8, 16]),
            ("model.norm.weight", vec![8]),
        ]
    }

    /// `(sliding_window, use_alibi)` as carried by the model's attention layers.
    fn attention_flags(sliding_window: Option<usize>, use_alibi: bool) -> (usize, bool) {
        let (_, device) = cpu_setup();
        let config = hybrid_config(sliding_window, use_alibi);
        let mut varmap = VarMap::<CpuRuntime>::new();
        for (name, shape) in weight_shapes() {
            varmap.insert(
                name.into(),
                Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device).unwrap(),
            );
        }
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let model = HybridModel::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        let attn = model
            .blocks
            .iter()
            .find_map(|b| match b {
                HybridBlock::Attention(a) => Some(a),
                HybridBlock::Ssm(_) => None,
            })
            .expect("the config places an attention layer at index 1");
        (attn.sliding_window, attn.use_alibi)
    }

    #[test]
    fn builder_reads_sliding_window_from_config() {
        assert_eq!(attention_flags(Some(64), false).0, 64);
        assert_eq!(attention_flags(Some(1), false).0, 1);
    }

    #[test]
    fn absent_sliding_window_is_disabled() {
        assert_eq!(attention_flags(None, false).0, 0);
    }

    #[test]
    fn explicit_zero_sliding_window_is_disabled() {
        // `Some(0)` is not a zero-width window — it maps to the disabled sentinel.
        assert_eq!(attention_flags(Some(0), false).0, 0);
    }

    #[test]
    fn builder_reads_use_alibi_from_config() {
        assert!(attention_flags(None, true).1);
        assert!(!attention_flags(None, false).1);
    }

    #[test]
    fn default_config_leaves_both_features_off() {
        assert_eq!(attention_flags(None, false), (0, false));
    }
}
