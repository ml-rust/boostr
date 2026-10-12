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
        rope_theta: Some(10000.0),
        rope_parameters: None,
        sliding_window: None,
        use_sliding_window: None,
        max_window_layers: None,
        layer_types: None,
        rms_norm_eps: 1e-5,
        layer_norm_eps: None,
        rope_scaling: None,
        original_max_position_embeddings: None,
        tie_word_embeddings: Some(false),
        num_local_experts: None,
        num_experts_per_tok: None,
        norm_topk_prob: None,
        moe_intermediate_size: None,
        attention_bias: None,
        alibi: None,
        multi_query: None,
        new_decoder_architecture: None,
        parallel_attn: None,
        use_parallel_residual: None,
        hidden_act: None,
        hidden_activation: None,
        partial_rotary_factor: None,
        rotary_pct: None,
        query_pre_attn_scalar: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        logit_scale: None,
        use_qk_norm: None,
        attn_config: None,
        ffn_config: None,
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
