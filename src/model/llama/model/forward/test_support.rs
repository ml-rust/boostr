//! Shared config fixtures for the `forward` module's test files.

use crate::model::config::ModelConfig;

pub(super) fn tiny_config() -> ModelConfig {
    let yaml = r#"
model_type: llama
vocab_size: 32
hidden_size: 16
num_layers: 2
max_seq_len: 32
intermediate_size: 32
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  rope_theta: 10000.0
"#;
    serde_saphyr::from_str(yaml).unwrap()
}

pub(super) fn tiny_alibi_config() -> ModelConfig {
    let yaml = r#"
model_type: falcon
vocab_size: 32
hidden_size: 16
num_layers: 2
max_seq_len: 32
intermediate_size: 32
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  use_alibi: true
"#;
    serde_saphyr::from_str(yaml).unwrap()
}
