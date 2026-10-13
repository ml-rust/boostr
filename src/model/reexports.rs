//! The flat `crate::model::*` names, re-exported from each architecture module.

pub use super::attention_core::{
    AttentionCoreSpec, AttentionKernel, attention_core, attention_core_flash,
    attention_core_masked, prefill_attention_mask,
};
pub use super::config::{
    AttentionConfig, AudioConfig, GdnConfig, HuggingFaceConfig, HybridConfig, ModelConfig,
    MoeConfig, Qwen35AttentionConfig, RopeScalingConfig, SsmConfig, UniversalConfig, VisionConfig,
    load_config_auto, load_huggingface_config,
};
pub use super::encoder::{EmbeddingPipeline, Encoder, EncoderClient, EncoderConfig, Pooling};
pub use super::hybrid::HybridModel;
pub use super::llama::model::blocks::ExpertWeights;
pub use super::llama::{Llama, LlamaTp};
pub use super::mamba::{
    Mamba1, Mamba1Config, Mamba1Model, Mamba1Weights, Mamba2, Mamba2Config, Mamba2Model,
    Mamba2Weights, Mamba3, Mamba3Config, Mamba3Model, Mamba3Weights,
};
pub use super::multimodal::{ModelInput, MultimodalModel};
pub use super::qwen35::{Qwen35Model, qwen35_config_from_gguf, qwen35_config_from_hf};
pub use super::registry::LoadedModel;
pub use super::speech_lm::{
    ALL_SPECIAL_TOKENS, CodecVocab, DEFAULT_CONTROL_REGION, ExpressiveTtsLayout, OwnedSpeechRecord,
    SpecialToken, SpeechLayout, SpeechRecord, SpeechVocab, pack_record, pack_records,
    pack_records_padded, unpack_record, unpack_records,
};
pub use super::traits::{Model, ModelClient};
pub use super::vocab_growth::fit_vocab_rows;
