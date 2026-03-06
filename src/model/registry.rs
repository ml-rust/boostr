//! Model registry for loading models by name
//!
//! Provides `LoadedModel<R>`, an enum dispatching to concrete model types
//! with a unified inference API.

use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::model::mamba::mamba2::Mamba2Config;
use crate::model::traits::Model;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// Enum of all supported model architectures
///
/// Provides dynamic dispatch at the model level without
/// sacrificing type safety. The runtime type parameter `R` is preserved
/// across all variants.
pub enum LoadedModel<R: Runtime> {
    /// Standard GQA transformer model
    ///
    /// Covers all architectures that share the LLaMA structure:
    /// token embedding → transformer blocks (GQA + FFN) → RMSNorm → LM head.
    ///
    /// | HF `model_type`  | Example models                          |
    /// |------------------|-----------------------------------------|
    /// | `llama`          | Llama 2/3, CodeLlama, Yi, Solar         |
    /// | `mistral`        | Mistral 7B, Mixtral (dense path)        |
    /// | `qwen2`          | Qwen2-7B, Qwen2-72B                    |
    /// | `qwen2_moe`      | Qwen2-57B-A14B (MoE variant)           |
    /// | `phi3`           | Phi-3-mini, Phi-3-medium                |
    /// | `phi`            | Phi-2                                   |
    /// | `gemma`          | Gemma 7B                                |
    /// | `gemma2`         | Gemma 2 9B/27B                          |
    /// | `starcoder2`     | StarCoder2 3B/7B/15B                    |
    /// | `internlm2`      | InternLM2 7B/20B                        |
    Llama(Box<super::llama::Llama<R>>),
    /// Tensor-parallel LLaMA model (sharded across multiple GPUs via NCCL)
    LlamaTp(Box<super::llama::LlamaTp<R>>),
    /// Mamba2 SSM model (full model with embedding + layers + lm_head)
    Mamba2(Box<super::mamba::Mamba2Model<R>>),
    /// Hybrid model mixing attention and SSM layers
    Hybrid(Box<super::hybrid::HybridModel<R>>),
}

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R> + crate::quant::DequantOps<R> + numr::ops::TypeConversionOps<R>,
{
    /// Load a model from universal config and weights.
    ///
    /// Uses capability-based dispatch: any model with an attention config
    /// is loaded as a Llama (universal transformer). This means new HF
    /// model types work automatically without code changes as long as they
    /// share the standard transformer structure.
    pub fn load(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        match config.model_type.as_str() {
            "mamba2" | "mamba3" => {
                let model = super::mamba::Mamba2Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Mamba2(Box::new(model)))
            }
            "hybrid" => {
                let model = super::hybrid::HybridModel::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Hybrid(Box::new(model)))
            }
            // Everything else with attention config → Llama (the universal transformer)
            _ if config.attention.is_some() => {
                let model = super::llama::Llama::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Llama(Box::new(model)))
            }
            other => Err(Error::ModelError {
                reason: format!(
                    "Unknown model type '{other}' without attention config. \
                     Only pure SSM models (mamba2/mamba3) and hybrid models are \
                     supported without attention configuration."
                ),
            }),
        }
    }

    /// Load a tensor-parallel model. Requires a NCCL communicator.
    ///
    /// Tensor parallelism is supported for any model with an attention config
    /// (i.e., transformer architectures loaded via the Llama struct).
    pub fn load_tp(
        config: &UniversalConfig,
        vb: &mut VarBuilder<R>,
        comm: std::sync::Arc<dyn numr::runtime::Communicator>,
    ) -> Result<Self> {
        if config.attention.is_some() {
            let model = super::llama::LlamaTp::from_varbuilder(vb, config, comm)?;
            Ok(LoadedModel::LlamaTp(Box::new(model)))
        } else {
            Err(Error::ModelError {
                reason: format!(
                    "Tensor parallelism not supported for model type '{}' \
                     (requires attention config)",
                    config.model_type
                ),
            })
        }
    }

    /// Load a model from GGUF format
    pub fn load_gguf(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        Self::load(config, vb)
    }
}

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Whether this model uses KV cache (transformer) or SSM state.
    pub fn needs_kv_cache(&self) -> bool {
        matches!(
            self,
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) | LoadedModel::Hybrid(_)
        )
    }

    /// Whether this model uses SSM state.
    pub fn needs_ssm_state(&self) -> bool {
        matches!(self, LoadedModel::Mamba2(_) | LoadedModel::Hybrid(_))
    }

    /// Get model type name
    pub fn model_type(&self) -> &str {
        match self {
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => "llama",
            LoadedModel::Mamba2(_) => "mamba2",
            LoadedModel::Hybrid(_) => "hybrid",
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().vocab_size,
            LoadedModel::LlamaTp(m) => m.config().vocab_size,
            LoadedModel::Mamba2(m) => m.config().vocab_size,
            LoadedModel::Hybrid(m) => m.config().vocab_size,
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().num_layers,
            LoadedModel::LlamaTp(m) => m.config().num_layers,
            LoadedModel::Mamba2(m) => m.config().num_layers,
            LoadedModel::Hybrid(m) => m.config().num_layers,
        }
    }

    /// Get hidden size (embedding dimension)
    pub fn hidden_size(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().hidden_size,
            LoadedModel::LlamaTp(m) => m.config().hidden_size,
            LoadedModel::Mamba2(m) => m.config().hidden_size,
            LoadedModel::Hybrid(m) => m.config().hidden_size,
        }
    }

    /// Get number of KV heads (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    /// For LlamaTp, returns LOCAL kv heads (total / world_size).
    pub fn num_kv_heads(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::LlamaTp(m) => m
                .config()
                .attention
                .as_ref()
                .map(|a| a.kv_heads() / m.world_size()),
            LoadedModel::Mamba2(_) => None,
            LoadedModel::Hybrid(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
        }
    }

    /// Get head dimension (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    pub fn head_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => {
                let config = m.config();
                config
                    .attention
                    .as_ref()
                    .map(|a| a.head_dim(config.hidden_size))
            }
            LoadedModel::LlamaTp(m) => {
                let config = m.config();
                config
                    .attention
                    .as_ref()
                    .map(|a| a.head_dim(config.hidden_size))
            }
            LoadedModel::Mamba2(_) => None,
            LoadedModel::Hybrid(m) => {
                let config = m.config();
                config
                    .attention
                    .as_ref()
                    .map(|a| a.head_dim(config.hidden_size))
            }
        }
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().max_seq_len,
            LoadedModel::LlamaTp(m) => m.config().max_seq_len,
            LoadedModel::Mamba2(m) => m.config().max_seq_len,
            LoadedModel::Hybrid(m) => m.config().max_seq_len,
        }
    }

    /// Whether this model uses Mixture of Experts.
    pub fn is_moe(&self) -> bool {
        match self {
            LoadedModel::Llama(m) => m.config().moe.is_some(),
            LoadedModel::LlamaTp(m) => m.config().moe.is_some(),
            LoadedModel::Mamba2(m) => m.config().moe.is_some(),
            LoadedModel::Hybrid(m) => m.config().moe.is_some(),
        }
    }

    /// Get MoE configuration, if this is an MoE model.
    pub fn moe_config(&self) -> Option<&crate::model::config::MoeConfig> {
        match self {
            LoadedModel::Llama(m) => m.config().moe.as_ref(),
            LoadedModel::LlamaTp(m) => m.config().moe.as_ref(),
            LoadedModel::Mamba2(m) => m.config().moe.as_ref(),
            LoadedModel::Hybrid(m) => m.config().moe.as_ref(),
        }
    }

    /// Get the RoPE cos/sin caches for Llama models (for CUDA graph setup).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use RoPE.
    pub fn rope_caches(&self) -> Option<(&numr::autograd::Var<R>, &numr::autograd::Var<R>)> {
        match self {
            LoadedModel::Llama(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
            LoadedModel::LlamaTp(_) => None, // TP model manages RoPE internally
            LoadedModel::Mamba2(_) => None,
            LoadedModel::Hybrid(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
        }
    }

    /// Get the Mamba2 config (for SSM state allocation).
    pub fn mamba_config(&self) -> Option<&Mamba2Config> {
        match self {
            LoadedModel::Mamba2(m) => Some(m.mamba_config()),
            LoadedModel::Hybrid(m) => Some(m.mamba_config()),
            _ => None,
        }
    }
}

impl<R: Runtime> std::fmt::Debug for LoadedModel<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedModel::Llama(_) => f.debug_tuple("Llama").finish(),
            LoadedModel::LlamaTp(_) => f.debug_tuple("LlamaTp").finish(),
            LoadedModel::Mamba2(_) => f.debug_tuple("Mamba2").finish(),
            LoadedModel::Hybrid(_) => f.debug_tuple("Hybrid").finish(),
        }
    }
}
