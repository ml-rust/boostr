//! `LoadedModel` enum and its `Debug` impl.

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
    Llama(Box<crate::model::llama::Llama<R>>),
    /// Tensor-parallel LLaMA model (sharded across multiple GPUs via NCCL)
    LlamaTp(Box<crate::model::llama::LlamaTp<R>>),
    /// Mamba1 SSM model (original selective SSM + depthwise convolution)
    Mamba1(Box<crate::model::mamba::Mamba1Model<R>>),
    /// Mamba2 SSM model (full model with embedding + layers + lm_head)
    Mamba2(Box<crate::model::mamba::Mamba2Model<R>>),
    /// Mamba3 SSM model (trapezoidal discretization + optional complex RoPE/MIMO)
    Mamba3(Box<crate::model::mamba::Mamba3Model<R>>),
    /// Hybrid model mixing attention and SSM layers
    Hybrid(Box<crate::model::hybrid::HybridModel<R>>),
    /// Multimodal model with vision/audio encoders + LLM backbone
    Multimodal(Box<crate::model::multimodal::MultimodalModel<R>>),
    /// `qwen35`: Gated DeltaNet + gated full-attention hybrid (Bonsai).
    /// KV cache for the attention layers, GDN state for the rest.
    Qwen35(Box<crate::model::qwen35::Qwen35Model<R>>),
}

impl<R: Runtime> std::fmt::Debug for LoadedModel<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedModel::Llama(_) => f.debug_tuple("Llama").finish(),
            LoadedModel::LlamaTp(_) => f.debug_tuple("LlamaTp").finish(),
            LoadedModel::Mamba1(_) => f.debug_tuple("Mamba1").finish(),
            LoadedModel::Mamba2(_) => f.debug_tuple("Mamba2").finish(),
            LoadedModel::Mamba3(_) => f.debug_tuple("Mamba3").finish(),
            LoadedModel::Hybrid(_) => f.debug_tuple("Hybrid").finish(),
            LoadedModel::Multimodal(_) => f.debug_tuple("Multimodal").finish(),
            LoadedModel::Qwen35(_) => f.debug_tuple("Qwen35").finish(),
        }
    }
}
