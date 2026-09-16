//! Shared HuggingFaceConfig fixtures for the `convert` module's test files.

use super::types::HuggingFaceConfig;

/// Helper: build a minimal HuggingFaceConfig with only architectures set
/// (model_type = None) to exercise the fallback inference path.
pub(super) fn config_with_arch(arch: &str) -> HuggingFaceConfig {
    HuggingFaceConfig {
        model_type: None,
        architectures: Some(vec![arch.to_string()]),
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        max_seq_len: 4096,
        num_attention_heads: Some(32),
        num_kv_heads: None,
        head_dim: None,
        intermediate_size: None,
        rope_theta: 10000.0,
        sliding_window: None,
        rms_norm_eps: 1e-5,
        rope_scaling: None,
        tie_word_embeddings: false,
        num_local_experts: None,
        num_experts_per_tok: None,
        attention_bias: None,
        alibi: None,
        multi_query: None,
        new_decoder_architecture: None,
        parallel_attn: None,
        vision_config: None,
        audio_config: None,
    }
}

/// Helper: build a minimal HuggingFaceConfig with model_type set directly
/// (the primary path — HF configs almost always have model_type).
pub(super) fn config_with_model_type(mt: &str) -> HuggingFaceConfig {
    let mut c = config_with_arch("Unused");
    c.model_type = Some(mt.to_string());
    c
}
