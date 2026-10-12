//! Real Hugging Face Qwen3.5 `config.json` files, minus `vision_config` and
//! token ids, for the `config_from_hf` tests.

use serde_json::Value;

/// `layer_types` of every config below: full attention at `3, 7, .., 31`.
const LAYER_TYPES: &str = r#"[
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention"]"#;

/// `Qwen/Qwen3.5-4B-Base`. Tied in both blocks.
pub(super) const QWEN35_4B_BASE: &str = r#"{
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false, "attn_output_gate": true, "dtype": "bfloat16",
        "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
        "hidden_size": 2560, "intermediate_size": 9216, "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_value_head_dim": 128, "max_position_embeddings": 262144,
        "mlp_only_layers": [], "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
            "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}
    },
    "tie_word_embeddings": true
}"#;

/// `Qwen/Qwen3.5-9B`. `tie_word_embeddings` sits at the top level only.
pub(super) const QWEN35_9B: &str = r#"{
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false, "attn_output_gate": true, "dtype": "bfloat16",
        "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
        "hidden_size": 4096, "intermediate_size": 12288, "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_value_head_dim": 128, "max_position_embeddings": 262144,
        "mlp_only_layers": [], "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06, "vocab_size": 248320, "mamba_ssm_dtype": "float32",
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
            "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}
    },
    "tie_word_embeddings": false
}"#;

/// `Cloudflare/clef-flash`. `partial_rotary_factor` sits both in
/// `text_config` and in `rope_parameters`.
pub(super) const CLEF_FLASH: &str = r#"{
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "dtype": "bfloat16",
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false, "attn_output_gate": true, "bos_token_id": null,
        "dtype": "bfloat16", "full_attention_interval": 4, "head_dim": 256,
        "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 12288,
        "layer_types": LAYER_TYPES, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144,
        "mlp_only_layers": [], "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 0,
        "mtp_use_dedicated_embeddings": false, "num_attention_heads": 16,
        "num_hidden_layers": 32, "num_key_value_heads": 4, "pad_token_id": null,
        "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "tie_word_embeddings": false, "vocab_size": 248320
    },
    "tie_word_embeddings": false
}"#;

/// One of the configs above, parsed.
pub(super) fn parse(json: &str) -> Value {
    serde_json::from_str(&json.replace("LAYER_TYPES", LAYER_TYPES)).unwrap()
}
