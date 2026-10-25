//! Size and dtype accessors on [`LoadedModel`]: the numbers a cache,
//! a stage or a loader sizes itself by.

use super::model::LoadedModel;
use crate::model::llama::LayerSpan;
use crate::model::traits::Model;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Compute dtype: the dtype every forward pass returns logits in.
    ///
    /// Reads an owned, never-quantized weight (the final norm) so a
    /// quantized checkpoint still reports its real activation dtype rather
    /// than the quantized storage type of its projection weights.
    pub fn dtype(&self) -> DType {
        match self {
            LoadedModel::Llama(m) => m.dtype(),
            LoadedModel::LlamaStage(m) => m.dtype(),
            LoadedModel::LlamaTp(m) => m.dtype(),
            LoadedModel::Mamba1(m) => m.dtype(),
            LoadedModel::Mamba2(m) => m.dtype(),
            LoadedModel::Mamba3(m) => m.dtype(),
            LoadedModel::Hybrid(m) => m.dtype(),
            LoadedModel::Multimodal(m) => m.llm().dtype(),
            LoadedModel::Qwen35(m) => m.dtype(),
        }
    }

    /// Distinct quantized formats the loaded checkpoint's tensors use.
    ///
    /// Non-empty only for a GGUF checkpoint whose tensors carry a quantized
    /// `GgmlType` (`LoadedModel::load` reads them from the loading
    /// `VarBuilder`'s tensor infos). Empty for SafeTensors.
    pub fn quant_formats(&self) -> &[crate::quant::QuantFormat] {
        match self {
            LoadedModel::Llama(m) => &m.config().quant_formats,
            LoadedModel::LlamaStage(m) => &m.config().quant_formats,
            LoadedModel::LlamaTp(m) => &m.config().quant_formats,
            LoadedModel::Mamba1(m) => &m.config().quant_formats,
            LoadedModel::Mamba2(m) => &m.config().quant_formats,
            LoadedModel::Mamba3(m) => &m.config().quant_formats,
            LoadedModel::Hybrid(m) => &m.config().quant_formats,
            LoadedModel::Multimodal(m) => &m.config().quant_formats,
            LoadedModel::Qwen35(m) => &m.config().quant_formats,
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().vocab_size,
            LoadedModel::LlamaStage(m) => m.config().vocab_size,
            LoadedModel::LlamaTp(m) => m.config().vocab_size,
            LoadedModel::Mamba1(m) => m.config().vocab_size,
            LoadedModel::Mamba2(m) => m.config().vocab_size,
            LoadedModel::Mamba3(m) => m.config().vocab_size,
            LoadedModel::Hybrid(m) => m.config().vocab_size,
            LoadedModel::Multimodal(m) => m.config().vocab_size,
            LoadedModel::Qwen35(m) => m.config().vocab_size,
        }
    }

    /// Layers of the whole model. A `LlamaStage` holds the
    /// [`layer_span`](Self::layer_span) of them only.
    pub fn num_layers(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().num_layers,
            LoadedModel::LlamaStage(m) => m.config().num_layers,
            LoadedModel::LlamaTp(m) => m.config().num_layers,
            LoadedModel::Mamba1(m) => m.config().num_layers,
            LoadedModel::Mamba2(m) => m.config().num_layers,
            LoadedModel::Mamba3(m) => m.config().num_layers,
            LoadedModel::Hybrid(m) => m.config().num_layers,
            LoadedModel::Multimodal(m) => m.config().num_layers,
            LoadedModel::Qwen35(m) => m.config().num_layers,
        }
    }

    /// Get hidden size (embedding dimension)
    pub fn hidden_size(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().hidden_size,
            LoadedModel::LlamaStage(m) => m.config().hidden_size,
            LoadedModel::LlamaTp(m) => m.config().hidden_size,
            LoadedModel::Mamba1(m) => m.config().hidden_size,
            LoadedModel::Mamba2(m) => m.config().hidden_size,
            LoadedModel::Mamba3(m) => m.config().hidden_size,
            LoadedModel::Hybrid(m) => m.config().hidden_size,
            LoadedModel::Multimodal(m) => m.config().hidden_size,
            LoadedModel::Qwen35(m) => m.config().hidden_size,
        }
    }

    /// Get number of KV heads (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    /// An MLA model caches one latent head per token: `Some(1)`.
    /// For LlamaTp, returns LOCAL kv heads (total / world_size).
    pub fn num_kv_heads(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::LlamaStage(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::LlamaTp(m) => m
                .config()
                .attention
                .as_ref()
                .map(|a| a.kv_heads() / m.world_size()),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => m.kv_cache_dims().map(|(heads, _)| heads),
            LoadedModel::Multimodal(m) => m.llm().num_kv_heads(),
            LoadedModel::Qwen35(m) => Some(m.attention_config().num_kv_heads),
        }
    }

    /// Get head dimension (for KV cache allocation).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use a KV cache.
    /// An MLA model's cache head is `kv_lora_rank + rope_head_dim` wide, for
    /// K and V alike.
    pub fn head_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => {
                let config = m.config();
                config
                    .attention
                    .as_ref()
                    .map(|a| a.head_dim(config.hidden_size))
            }
            LoadedModel::LlamaStage(m) => {
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
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => m.kv_cache_dims().map(|(_, width)| width),
            LoadedModel::Multimodal(m) => m.llm().head_dim(),
            LoadedModel::Qwen35(m) => Some(m.attention_config().head_dim),
        }
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().max_seq_len,
            LoadedModel::LlamaStage(m) => m.config().max_seq_len,
            LoadedModel::LlamaTp(m) => m.config().max_seq_len,
            LoadedModel::Mamba1(m) => m.config().max_seq_len,
            LoadedModel::Mamba2(m) => m.config().max_seq_len,
            LoadedModel::Mamba3(m) => m.config().max_seq_len,
            LoadedModel::Hybrid(m) => m.config().max_seq_len,
            LoadedModel::Multimodal(m) => m.config().max_seq_len,
            LoadedModel::Qwen35(m) => m.config().max_seq_len,
        }
    }

    /// The layer range a [`LlamaStage`](crate::model::llama::LlamaStage)
    /// holds. `None` for a model that holds every layer.
    pub fn layer_span(&self) -> Option<&LayerSpan> {
        match self {
            LoadedModel::LlamaStage(m) => Some(m.span()),
            LoadedModel::Multimodal(m) => m.llm().layer_span(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_)
            | LoadedModel::Qwen35(_) => None,
        }
    }
}
