//! Encoder model configuration (BERT-style transformer encoders).

pub mod alibi;
pub mod arch_family;
pub mod attention_spec;
pub mod encoder_config;
pub mod ffn_variant;
pub mod gguf;
pub mod norm_scheme;
pub mod qk_norm_scope;

pub use alibi::alibi_slopes;
pub use arch_family::ArchFamily;
pub use attention_spec::LayerAttention;
pub use encoder_config::{DEFAULT_MAX_TOKENS_PER_FORWARD, EncoderConfig};
pub use ffn_variant::{FfnVariant, HiddenAct};
pub use norm_scheme::NormScheme;
pub use qk_norm_scope::QkNormScope;

#[cfg(test)]
mod tests;
