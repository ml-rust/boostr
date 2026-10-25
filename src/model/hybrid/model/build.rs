//! The [`HybridModel`] type and its builder: derives the layer split from
//! the config and builds an SSM, dense attention or MLA block per layer.

use super::super::blocks::{AttentionBlock, MlaBlock, SsmBlock};
use super::layers::LayerPlan;
use crate::error::{Error, Result};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::Mamba2Config;
use crate::nn::{Embedding, Linear, MlaConfig, RmsNorm, RoPE, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// Hybrid model mixing attention (LLaMA-style or MLA) and SSM (Mamba2) blocks.
///
/// Every attention layer is MLA when the attention config sets
/// `kv_latent_dim`, else dense attention. Each attention layer's
/// feed-forward block is a MoE when the config has a `moe` section, else an
/// MLP. A config with no SSM layer builds too: an MLA model is this model
/// with every layer attention.
pub struct HybridModel<R: Runtime> {
    pub(super) config: UniversalConfig,
    pub(super) hybrid_config: HybridConfig,
    pub(super) mamba_config: Option<Mamba2Config>,
    pub(super) embed_tokens: Embedding<R>,
    pub(super) blocks: Vec<HybridBlock<R>>,
    pub(super) norm: RmsNorm<R>,
    pub(super) lm_head: Linear<R>,
    /// The RoPE tables of the dense attention layers, or the decoupled RoPE
    /// tables of the MLA layers.
    pub(super) rope: RoPE<R>,
    /// `(kv_heads, head_dim)` of one attention layer's KV cache. `None`
    /// without attention layers.
    pub(super) kv_dims: Option<(usize, usize)>,
}

/// One layer of the hybrid model.
pub(super) enum HybridBlock<R: Runtime> {
    Attention(Box<AttentionBlock<R>>),
    Mla(Box<MlaBlock<R>>),
    Ssm(Box<SsmBlock<R>>),
}

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R> + DequantOps<R>,
{
    /// Load from a VarBuilder and UniversalConfig.
    ///
    /// The layer split is [`UniversalConfig::layer_split`]. SSM layers read
    /// the oxidizr layout (`norm`, `mamba2`) or the Hugging Face one
    /// (`input_layernorm`, `mixer`).
    ///
    /// # Errors
    ///
    /// An invalid config, an SSM layer without an `ssm` section, an
    /// attention layer without an `attention` section, an MLA config that
    /// [`MlaConfig::from_universal`] refuses, or a missing tensor.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        config.validate()?;
        let hybrid_config = config.layer_split();
        let mamba_config = if hybrid_config.ssm_layers.is_empty() {
            None
        } else {
            let mamba = Mamba2Config::from_universal(config)?;
            mamba.validate()?;
            Some(mamba)
        };
        let attn_config = config.attention.as_ref();
        if attn_config.is_none() && !hybrid_config.attention_layers.is_empty() {
            return Err(Error::ModelError {
                reason: "Hybrid model requires attention config for attention layers".into(),
            });
        }
        let mla_config = match attn_config {
            Some(attn) if attn.is_mla() => Some(MlaConfig::from_universal(config)?),
            _ => None,
        };
        let rope = match (attn_config, &mla_config) {
            (_, Some(mla)) => RoPE::<R>::precompute_freqs(
                config.max_seq_len,
                mla.rope_head_dim,
                mla.rope_theta,
                mla.rope_scaling.as_ref(),
                vb.device(),
            )?,
            (Some(attn), None) => RoPE::<R>::precompute_freqs(
                config.max_seq_len,
                attn.head_dim(config.hidden_size),
                attn.rope_theta,
                attn.rope_scaling.as_ref(),
                vb.device(),
            )?,
            // No attention layer reads RoPE. A two-wide table keeps the
            // accessor total.
            (None, None) => {
                RoPE::<R>::precompute_freqs(config.max_seq_len, 2, 10000.0, None, vb.device())?
            }
        };
        let kv_dims = if hybrid_config.attention_layers.is_empty() {
            None
        } else {
            match (attn_config, &mla_config) {
                (_, Some(mla)) => Some((1, mla.latent_cache_dim())),
                (Some(attn), None) => Some((attn.kv_heads(), attn.head_dim(config.hidden_size))),
                (None, None) => None,
            }
        };

        let mut model_vb = vb.pp("model");
        let embed_tokens = Embedding::new(model_vb.take_tensor("embed_tokens.weight")?, false);
        let plan = LayerPlan {
            config,
            mamba: mamba_config.as_ref(),
            mla: mla_config.as_ref(),
            dtype: embed_tokens.weight().tensor().dtype(),
        };
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            blocks.push(plan.build(&mut layers_vb.pp(&i.to_string()), i)?);
        }

        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );
        let lm_head = if config.tie_word_embeddings {
            let embed_w = embed_tokens.weight().tensor().clone();
            Linear::new(embed_w, None, false)
        } else {
            Linear::new(vb.take_tensor("lm_head.weight")?, None, false)
        };

        Ok(Self {
            config: config.clone(),
            hybrid_config,
            mamba_config,
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
            kv_dims,
        })
    }
}

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Compute dtype, read from the final norm weight (owned, never
    /// quantized).
    pub fn dtype(&self) -> DType {
        self.norm.weight().tensor().dtype()
    }
}

#[cfg(test)]
pub(crate) mod tests {
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
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            block: Default::default(),
            quant_formats: Vec::new(),
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
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            block: Default::default(),
            quant_formats: Vec::new(),
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
                HybridBlock::Mla(_) | HybridBlock::Ssm(_) => None,
            })
            .expect("the config places an attention layer at index 1");
        (attn.sliding_window, attn.use_alibi)
    }

    pub(crate) fn tiny_model(device: &numr::runtime::cpu::CpuDevice) -> HybridModel<CpuRuntime> {
        let config = hybrid_config(None, false);
        let mut varmap = VarMap::<CpuRuntime>::new();
        for (name, shape) in weight_shapes() {
            varmap.insert(
                name.into(),
                Tensor::<CpuRuntime>::zeros(&shape, DType::F32, device).unwrap(),
            );
        }
        let mut vb = VarBuilder::new(&mut varmap, device);
        HybridModel::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap()
    }

    #[test]
    fn hybrid_dtype_matches_the_varmap_dtype_it_is_built_with() {
        let (_, device) = cpu_setup();
        let model = tiny_model(&device);
        assert_eq!(model.dtype(), DType::F32);
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
