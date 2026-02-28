//! Model registry for loading models by name
//!
//! Provides `LoadedModel<R>`, an enum dispatching to concrete model types
//! with a unified inference API.

use crate::error::{Error, Result};
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::config::UniversalConfig;
use crate::model::mamba::mamba2::Mamba2Config;
use crate::model::traits::{Model, ModelClient};
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, NormalizationOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Enum of all supported model architectures
///
/// Provides dynamic dispatch at the model level without
/// sacrificing type safety. The runtime type parameter `R` is preserved
/// across all variants.
pub enum LoadedModel<R: Runtime> {
    /// Llama model (Llama 2, Llama 3, etc.)
    Llama(Box<super::llama::Llama<R>>),
    /// Mamba2 SSM model (full model with embedding + layers + lm_head)
    Mamba2(Box<super::mamba::Mamba2Model<R>>),
    /// Hybrid model mixing attention and SSM layers
    Hybrid(Box<super::hybrid::HybridModel<R>>),
}

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R> + crate::quant::DequantOps<R>,
{
    /// Load a model from universal config and weights
    pub fn load(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        match config.model_type.as_str() {
            "llama" | "mistral" => {
                let model = super::llama::Llama::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Llama(Box::new(model)))
            }
            "mamba2" | "mamba3" => {
                let model = super::mamba::Mamba2Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Mamba2(Box::new(model)))
            }
            "hybrid" => {
                let model = super::hybrid::HybridModel::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Hybrid(Box::new(model)))
            }
            other => Err(Error::ModelError {
                reason: format!("Unknown model type: {other}"),
            }),
        }
    }

    /// Load a model from GGUF format
    pub fn load_gguf(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        // GGUF uses same load path - VarBuilder handles name mapping
        Self::load(config, vb)
    }
}

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Forward pass with pre-allocated KV cache for transformer inference
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape `[batch_size, seq_len]`
    /// * `kv_cache` - Mutable reference to pre-allocated KV cache
    /// * `position` - Starting position for positional encoding
    ///
    /// # Returns
    /// Logits tensor, shape `[batch_size, seq_len, vocab_size]`
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => {
                m.forward_with_kv_cache(&client, input_ids, kv_cache, position)
            }
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not use KV cache — use forward_with_ssm_state() instead"
                    .into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_with_kv_cache — use forward_hybrid() instead"
                    .into(),
            }),
        }
    }

    /// Forward pass with SSM state for Mamba2 inference
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape `[batch_size, seq_len]`
    /// * `ssm_state` - Mutable reference to pre-allocated SSM state
    ///
    /// # Returns
    /// Logits tensor, shape `[batch_size, seq_len, vocab_size]`
    pub fn forward_with_ssm_state(
        &self,
        input_ids: &Tensor<R>,
        ssm_state: &mut LayeredSsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R::Client:
            ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Mamba2(m) => m.forward_with_ssm_state(&client, input_ids, ssm_state),
            LoadedModel::Llama(_) => Err(Error::ModelError {
                reason: "Llama does not use SSM state — use forward_with_kv_cache() instead".into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_with_ssm_state — use forward_hybrid() instead"
                    .into(),
            }),
        }
    }

    /// Forward pass for hybrid model with both KV cache and SSM state
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape `[batch_size, seq_len]`
    /// * `kv_cache` - Mutable reference to pre-allocated KV cache for attention layers
    /// * `ssm_state` - Mutable reference to pre-allocated SSM state for Mamba2 layers
    /// * `position` - Starting position for positional encoding
    ///
    /// # Returns
    /// Logits tensor, shape `[batch_size, seq_len, vocab_size]`
    pub fn forward_hybrid(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        ssm_state: &mut LayeredSsmState<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ShapeOps<R>
            + TensorOps<R>
            + ScalarOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Hybrid(m) => {
                m.forward_hybrid(&client, input_ids, kv_cache, ssm_state, position)
            }
            LoadedModel::Llama(_) => Err(Error::ModelError {
                reason: "Llama model does not support forward_hybrid — use forward_with_kv_cache()"
                    .into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason:
                    "Mamba2 model does not support forward_hybrid — use forward_with_ssm_state()"
                        .into(),
            }),
        }
    }

    /// Whether this model uses KV cache (transformer) or SSM state.
    pub fn needs_kv_cache(&self) -> bool {
        matches!(self, LoadedModel::Llama(_) | LoadedModel::Hybrid(_))
    }

    /// Whether this model uses SSM state.
    pub fn needs_ssm_state(&self) -> bool {
        matches!(self, LoadedModel::Mamba2(_) | LoadedModel::Hybrid(_))
    }

    /// Get model type name
    pub fn model_type(&self) -> &str {
        match self {
            LoadedModel::Llama(_) => "llama",
            LoadedModel::Mamba2(_) => "mamba2",
            LoadedModel::Hybrid(_) => "hybrid",
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().vocab_size,
            LoadedModel::Mamba2(m) => m.config().vocab_size,
            LoadedModel::Hybrid(m) => m.config().vocab_size,
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().num_layers,
            LoadedModel::Mamba2(m) => m.config().num_layers,
            LoadedModel::Hybrid(m) => m.config().num_layers,
        }
    }

    /// Get number of KV heads (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    /// For Hybrid, returns KV heads from attention layers.
    pub fn num_kv_heads(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::Mamba2(_) => None,
            LoadedModel::Hybrid(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
        }
    }

    /// Get head dimension (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    /// For Hybrid, returns head dimension from attention layers.
    pub fn head_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => {
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
            LoadedModel::Mamba2(m) => m.config().max_seq_len,
            LoadedModel::Hybrid(m) => m.config().max_seq_len,
        }
    }

    /// Get the Mamba2 config (for SSM state allocation). Returns None for non-Mamba2/Hybrid models.
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
            LoadedModel::Mamba2(_) => f.debug_tuple("Mamba2").finish(),
            LoadedModel::Hybrid(_) => f.debug_tuple("Hybrid").finish(),
        }
    }
}
