//! Accessor methods on [`LoadedModel`]: capability flags and per-variant
//! config lookups.

use super::model::LoadedModel;
use crate::model::config::GdnConfig;
use crate::model::mamba::mamba1::Mamba1Config;
use crate::model::mamba::mamba2::Mamba2Config;
use crate::model::mamba::mamba3::Mamba3Config;
use crate::model::traits::Model;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Whether this model uses KV cache (transformer) or SSM state.
    pub fn needs_kv_cache(&self) -> bool {
        match self {
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Hybrid(_)
            | LoadedModel::Qwen35(_) => true,
            LoadedModel::Multimodal(m) => m.llm().needs_kv_cache(),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => false,
        }
    }

    /// Whether this model uses SSM state.
    pub fn needs_ssm_state(&self) -> bool {
        match self {
            LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => true,
            LoadedModel::Multimodal(m) => m.llm().needs_ssm_state(),
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) | LoadedModel::Qwen35(_) => false,
        }
    }

    /// Whether this model carries Gated DeltaNet state (`LayeredGdnState`).
    /// Distinct from `needs_ssm_state`: that one is `LayeredSsmState`,
    /// Mamba2-shaped.
    pub fn needs_gdn_state(&self) -> bool {
        match self {
            LoadedModel::Qwen35(_) => true,
            LoadedModel::Multimodal(m) => m.llm().needs_gdn_state(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => false,
        }
    }

    /// Get model type name
    pub fn model_type(&self) -> &str {
        match self {
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => "llama",
            LoadedModel::Mamba1(_) => "mamba1",
            LoadedModel::Mamba2(_) => "mamba2",
            LoadedModel::Mamba3(_) => "mamba3",
            LoadedModel::Hybrid(_) => "hybrid",
            LoadedModel::Multimodal(m) => m.config().model_type.as_str(),
            LoadedModel::Qwen35(_) => "qwen35",
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
            LoadedModel::LlamaTp(m) => m.config().vocab_size,
            LoadedModel::Mamba1(m) => m.config().vocab_size,
            LoadedModel::Mamba2(m) => m.config().vocab_size,
            LoadedModel::Mamba3(m) => m.config().vocab_size,
            LoadedModel::Hybrid(m) => m.config().vocab_size,
            LoadedModel::Multimodal(m) => m.config().vocab_size,
            LoadedModel::Qwen35(m) => m.config().vocab_size,
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().num_layers,
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
    /// For LlamaTp, returns LOCAL kv heads (total / world_size).
    pub fn num_kv_heads(&self) -> Option<usize> {
        match self {
            LoadedModel::Llama(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::LlamaTp(m) => m
                .config()
                .attention
                .as_ref()
                .map(|a| a.kv_heads() / m.world_size()),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => m.config().attention.as_ref().map(|a| a.kv_heads()),
            LoadedModel::Multimodal(m) => m.llm().num_kv_heads(),
            LoadedModel::Qwen35(m) => Some(m.attention_config().num_kv_heads),
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
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => {
                let config = m.config();
                config
                    .attention
                    .as_ref()
                    .map(|a| a.head_dim(config.hidden_size))
            }
            LoadedModel::Multimodal(m) => m.llm().head_dim(),
            LoadedModel::Qwen35(m) => Some(m.attention_config().head_dim),
        }
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        match self {
            LoadedModel::Llama(m) => m.config().max_seq_len,
            LoadedModel::LlamaTp(m) => m.config().max_seq_len,
            LoadedModel::Mamba1(m) => m.config().max_seq_len,
            LoadedModel::Mamba2(m) => m.config().max_seq_len,
            LoadedModel::Mamba3(m) => m.config().max_seq_len,
            LoadedModel::Hybrid(m) => m.config().max_seq_len,
            LoadedModel::Multimodal(m) => m.config().max_seq_len,
            LoadedModel::Qwen35(m) => m.config().max_seq_len,
        }
    }

    /// Whether this model uses Mixture of Experts.
    pub fn is_moe(&self) -> bool {
        match self {
            LoadedModel::Llama(m) => m.config().moe.is_some(),
            LoadedModel::LlamaTp(m) => m.config().moe.is_some(),
            LoadedModel::Mamba1(m) => m.config().moe.is_some(),
            LoadedModel::Mamba2(m) => m.config().moe.is_some(),
            LoadedModel::Mamba3(m) => m.config().moe.is_some(),
            LoadedModel::Hybrid(m) => m.config().moe.is_some(),
            LoadedModel::Multimodal(m) => m.llm().is_moe(),
            LoadedModel::Qwen35(_) => false,
        }
    }

    /// Get MoE configuration, if this is an MoE model.
    pub fn moe_config(&self) -> Option<&crate::model::config::MoeConfig> {
        match self {
            LoadedModel::Llama(m) => m.config().moe.as_ref(),
            LoadedModel::LlamaTp(m) => m.config().moe.as_ref(),
            LoadedModel::Mamba1(m) => m.config().moe.as_ref(),
            LoadedModel::Mamba2(m) => m.config().moe.as_ref(),
            LoadedModel::Mamba3(m) => m.config().moe.as_ref(),
            LoadedModel::Hybrid(m) => m.config().moe.as_ref(),
            LoadedModel::Multimodal(m) => m.llm().moe_config(),
            LoadedModel::Qwen35(_) => None,
        }
    }

    /// Get the RoPE cos/sin caches for Llama models (for CUDA graph setup).
    ///
    /// Returns `None` for Mamba2 — SSM layers do not use RoPE.
    pub fn rope_caches(&self) -> Option<(&numr::autograd::Var<R>, &numr::autograd::Var<R>)> {
        match self {
            LoadedModel::Llama(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
            LoadedModel::LlamaTp(_) => None, // TP model manages RoPE internally
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
            LoadedModel::Multimodal(m) => m.llm().rope_caches(),
            LoadedModel::Qwen35(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
        }
    }

    /// Get the Mamba1 config, if this is a Mamba1 model.
    pub fn mamba1_config(&self) -> Option<&Mamba1Config> {
        match self {
            LoadedModel::Mamba1(m) => Some(m.mamba_config()),
            LoadedModel::Multimodal(m) => m.llm().mamba1_config(),
            LoadedModel::LlamaTp(_)
            | LoadedModel::Llama(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_)
            | LoadedModel::Qwen35(_) => None,
        }
    }

    /// Get the Mamba2 config (for existing SSM state allocation).
    pub fn mamba_config(&self) -> Option<&Mamba2Config> {
        match self {
            LoadedModel::Mamba2(m) => Some(m.mamba_config()),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba3(_) => None,
            LoadedModel::Hybrid(m) => Some(m.mamba_config()),
            LoadedModel::Multimodal(m) => m.llm().mamba_config(),
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) | LoadedModel::Qwen35(_) => None,
        }
    }

    /// Get the Mamba3 config, if this is a Mamba3 model.
    pub fn mamba3_config(&self) -> Option<&Mamba3Config> {
        match self {
            LoadedModel::Mamba3(m) => Some(m.mamba_config()),
            LoadedModel::Multimodal(m) => m.llm().mamba3_config(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Hybrid(_)
            | LoadedModel::Qwen35(_) => None,
        }
    }

    /// Get the Gated DeltaNet config (for `LayeredGdnState` allocation).
    pub fn gdn_config(&self) -> Option<&GdnConfig> {
        match self {
            LoadedModel::Qwen35(m) => Some(m.gdn_config()),
            LoadedModel::Multimodal(m) => m.llm().gdn_config(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => None,
        }
    }

    /// Layers that read the `LayeredKvCache`. `Some` for `qwen35`, whose
    /// cache is sized per attention layer rather than per model layer.
    pub fn num_attention_layers(&self) -> Option<usize> {
        match self {
            LoadedModel::Qwen35(m) => Some(m.num_attention_layers()),
            LoadedModel::Multimodal(m) => m.llm().num_attention_layers(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => None,
        }
    }

    /// Layers that read the `LayeredGdnState`. `Some` for `qwen35`.
    pub fn num_gdn_layers(&self) -> Option<usize> {
        match self {
            LoadedModel::Qwen35(m) => Some(m.num_gdn_layers()),
            LoadedModel::Multimodal(m) => m.llm().num_gdn_layers(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35::model::tiny_model;
    use crate::test_utils::cpu_setup;

    #[test]
    fn qwen35_variant_reports_its_state_needs() {
        let (_client, device) = cpu_setup();
        let model = LoadedModel::Qwen35(Box::new(tiny_model(&device, 0x3535_0100)));
        assert_eq!(model.model_type(), "qwen35");
        assert!(model.needs_kv_cache());
        assert!(model.needs_gdn_state());
        assert!(!model.needs_ssm_state());
        assert!(!model.is_moe());
        assert!(model.moe_config().is_none());
        assert!(model.mamba_config().is_none());
        assert!(model.mamba1_config().is_none());
        assert!(model.mamba3_config().is_none());
        assert_eq!(model.num_kv_heads(), Some(1));
        assert_eq!(model.head_dim(), Some(8));
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.hidden_size(), 8);
        assert_eq!(model.vocab_size(), 16);
        assert_eq!(model.max_seq_len(), 32);
        assert_eq!(model.gdn_config().map(|g| g.value_heads), Some(4));
        assert_eq!(model.num_attention_layers(), Some(1));
        assert_eq!(model.num_gdn_layers(), Some(1));
        assert!(model.rope_caches().is_some());
        assert_eq!(format!("{model:?}"), "Qwen35");
    }
}
