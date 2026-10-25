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
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Hybrid(_)
            | LoadedModel::Qwen35(_) => true,
            LoadedModel::Multimodal(m) => m.llm().needs_kv_cache(),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => false,
        }
    }

    /// The tensor-parallel communicator of a `LlamaTp` shard. Every other
    /// variant runs on one device and returns `None`.
    pub fn tp_communicator(&self) -> Option<&dyn numr::runtime::Communicator> {
        match self {
            LoadedModel::LlamaTp(m) => Some(m.comm()),
            _ => None,
        }
    }

    /// Whether this model uses SSM state.
    pub fn needs_ssm_state(&self) -> bool {
        match self {
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => true,
            // An MLA model is a hybrid with no SSM layer.
            LoadedModel::Hybrid(m) => m.num_ssm_layers() > 0,
            LoadedModel::Multimodal(m) => m.llm().needs_ssm_state(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Qwen35(_) => false,
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
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => false,
        }
    }

    /// Whether [`forward_hybrid_rows`](Self::forward_hybrid_rows) runs this
    /// model: `true` for a `hybrid` model, MLA models included.
    pub fn supports_hybrid_rows(&self) -> bool {
        match self {
            LoadedModel::Hybrid(_) => true,
            LoadedModel::Multimodal(m) => m.llm().supports_hybrid_rows(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Qwen35(_) => false,
        }
    }

    /// Get model type name
    pub fn model_type(&self) -> &str {
        match self {
            LoadedModel::Llama(_) | LoadedModel::LlamaStage(_) | LoadedModel::LlamaTp(_) => "llama",
            LoadedModel::Mamba1(_) => "mamba1",
            LoadedModel::Mamba2(_) => "mamba2",
            LoadedModel::Mamba3(_) => "mamba3",
            LoadedModel::Hybrid(_) => "hybrid",
            LoadedModel::Multimodal(m) => m.config().model_type.as_str(),
            LoadedModel::Qwen35(_) => "qwen35",
        }
    }

    /// Why the paged-attention path cannot serve this model, or `None`.
    ///
    /// The paged kernel adds no ALiBi bias and no score softcap. A sliding
    /// window is checked per forward instead, since it matters only past the
    /// window length. Gated DeltaNet and SSM layers carry recurrent state
    /// that the paged KV cache does not hold.
    pub fn paged_attention_refusal(&self) -> Option<String> {
        let config = match self {
            LoadedModel::Llama(m) => m.config(),
            LoadedModel::LlamaTp(m) => m.config(),
            LoadedModel::Multimodal(m) => return m.llm().paged_attention_refusal(),
            LoadedModel::LlamaStage(m) => {
                return Some(m.full_model_refusal("paged attention").to_string());
            }
            LoadedModel::Qwen35(_) => {
                return Some(
                    "model type 'qwen35' carries Gated DeltaNet state and has no paged KV cache; \
                     turn paged attention off (max_batch_size > 1 serves it with CUDA lanes \
                     without paged attention)"
                        .to_string(),
                );
            }
            LoadedModel::Hybrid(m) if m.num_ssm_layers() > 0 => {
                return Some(
                    "model type 'hybrid' uses SSM layers, which keep recurrent state instead of \
                     a paged KV cache; turn paged attention off"
                        .to_string(),
                );
            }
            LoadedModel::Hybrid(m) if m.is_mla() => {
                return Some(
                    "this model uses MLA attention over a latent KV cache, which the \
                     paged-attention path does not run; turn paged attention off"
                        .to_string(),
                );
            }
            LoadedModel::Hybrid(_) => {
                return Some(
                    "model type 'hybrid' has no paged-attention forward; turn paged attention off"
                        .to_string(),
                );
            }
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => {
                return Some(format!(
                    "model type '{}' keeps SSM state and has no KV cache to page; \
                     turn paged attention off",
                    self.model_type()
                ));
            }
        };
        let feature = if config.attention.as_ref().is_some_and(|a| a.use_alibi) {
            "ALiBi"
        } else if config.block.attn_logit_softcap.is_some() {
            "attention logit softcapping"
        } else {
            return None;
        };
        Some(format!(
            "model type '{}' uses {feature}, which the paged-attention kernel does not apply; \
             turn paged attention off",
            config.model_type
        ))
    }

    /// Whether this model uses Mixture of Experts.
    pub fn is_moe(&self) -> bool {
        match self {
            LoadedModel::Llama(m) => m.config().moe.is_some(),
            LoadedModel::LlamaStage(m) => m.config().moe.is_some(),
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
            LoadedModel::LlamaStage(m) => m.config().moe.as_ref(),
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
            LoadedModel::LlamaStage(m) => Some((m.rope().cos_cache(), m.rope().sin_cache())),
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
            | LoadedModel::LlamaStage(_)
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
            LoadedModel::Hybrid(m) => m.mamba_config(),
            LoadedModel::Multimodal(m) => m.llm().mamba_config(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Qwen35(_) => None,
        }
    }

    /// Get the Mamba3 config, if this is a Mamba3 model.
    pub fn mamba3_config(&self) -> Option<&Mamba3Config> {
        match self {
            LoadedModel::Mamba3(m) => Some(m.mamba_config()),
            LoadedModel::Multimodal(m) => m.llm().mamba3_config(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
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
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Hybrid(_) => None,
        }
    }

    /// Layers that read the `LayeredKvCache`. `Some` for `qwen35` and
    /// `hybrid`, whose cache is sized per attention layer rather than per
    /// model layer.
    pub fn num_attention_layers(&self) -> Option<usize> {
        match self {
            LoadedModel::Qwen35(m) => Some(m.num_attention_layers()),
            LoadedModel::Hybrid(m) => Some(m.num_attention_layers()),
            LoadedModel::Multimodal(m) => m.llm().num_attention_layers(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_) => None,
        }
    }

    /// Layers that read the `LayeredSsmState` of a `hybrid` model, whose
    /// state is sized per SSM layer rather than per model layer. `None` for
    /// every other model; a pure SSM model reads one state per layer.
    pub fn num_ssm_layers(&self) -> Option<usize> {
        match self {
            LoadedModel::Hybrid(m) => Some(m.num_ssm_layers()),
            LoadedModel::Multimodal(m) => m.llm().num_ssm_layers(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
            | LoadedModel::LlamaTp(_)
            | LoadedModel::Mamba1(_)
            | LoadedModel::Mamba2(_)
            | LoadedModel::Mamba3(_)
            | LoadedModel::Qwen35(_) => None,
        }
    }

    /// Layers that read the `LayeredGdnState`. `Some` for `qwen35`.
    pub fn num_gdn_layers(&self) -> Option<usize> {
        match self {
            LoadedModel::Qwen35(m) => Some(m.num_gdn_layers()),
            LoadedModel::Multimodal(m) => m.llm().num_gdn_layers(),
            LoadedModel::Llama(_)
            | LoadedModel::LlamaStage(_)
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
        assert_eq!(model.dtype(), numr::dtype::DType::F32);
        assert_eq!(format!("{model:?}"), "Qwen35");
    }

    #[test]
    fn paged_attention_is_refused_for_state_models() {
        let (_client, device) = cpu_setup();
        let qwen = LoadedModel::Qwen35(Box::new(tiny_model(&device, 0x3535_0101)));
        let reason = qwen.paged_attention_refusal().unwrap();
        assert!(reason.contains("'qwen35'"), "{reason}");
        assert!(reason.contains("turn paged attention off"), "{reason}");

        let hybrid =
            LoadedModel::Hybrid(Box::new(crate::model::hybrid::model::tiny_model(&device)));
        assert!(
            hybrid
                .paged_attention_refusal()
                .unwrap()
                .contains("'hybrid'")
        );

        let mamba2 =
            LoadedModel::Mamba2(Box::new(crate::model::mamba::model::tiny_mamba2(&device)));
        assert!(
            mamba2
                .paged_attention_refusal()
                .unwrap()
                .contains("'mamba2'")
        );

        let yaml = "model_type: llama\nvocab_size: 32\nhidden_size: 16\nnum_layers: 1\n\
                    max_seq_len: 16\nintermediate_size: 32\nattention:\n  num_heads: 2\n";
        let config: crate::model::config::UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        let llama = LoadedModel::Llama(Box::new(
            crate::model::llama::Llama::<numr::runtime::cpu::CpuRuntime>::from_config(
                &config, &device,
            )
            .unwrap(),
        ));
        assert!(llama.paged_attention_refusal().is_none());

        let alibi = yaml.replace("num_heads: 2\n", "num_heads: 2\n  use_alibi: true\n");
        let config: crate::model::config::UniversalConfig = serde_saphyr::from_str(&alibi).unwrap();
        let llama = LoadedModel::Llama(Box::new(
            crate::model::llama::Llama::<numr::runtime::cpu::CpuRuntime>::from_config(
                &config, &device,
            )
            .unwrap(),
        ));
        assert!(llama.paged_attention_refusal().unwrap().contains("ALiBi"));
    }

    #[test]
    fn multimodal_over_qwen35_refuses_paged_attention() {
        let (_client, device) = cpu_setup();
        let llm = tiny_model(&device, 0x3535_0102);
        let config = llm.config().clone();
        let inner = LoadedModel::Qwen35(Box::new(llm));
        let wrapped = LoadedModel::Multimodal(Box::new(
            crate::model::multimodal::MultimodalModel::text_only(inner, config),
        ));
        assert!(
            wrapped
                .paged_attention_refusal()
                .unwrap()
                .contains("'qwen35'")
        );
    }
}
