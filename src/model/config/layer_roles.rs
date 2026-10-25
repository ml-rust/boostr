//! Layer-role queries on [`UniversalConfig`]: which layers are SSM and
//! which are attention.

use super::hybrid::HybridConfig;
use super::universal::UniversalConfig;

impl UniversalConfig {
    /// Whether layer `layer_idx` is an SSM (Mamba) layer.
    ///
    /// Consult this rather than reading `hybrid_layers` directly. A config
    /// describes layer roles in three different shapes, and only one of them
    /// populates `hybrid_layers`:
    ///
    /// - **Mixed**: `hybrid_layers` is `Some` and lists the split explicitly.
    /// - **All SSM**: `ssm` is `Some`, `attention` is `None`, and
    ///   `hybrid_layers` is `None` — there is no mix to describe, so every
    ///   layer is an SSM layer.
    /// - **No SSM**: `ssm` is `None`, so no layer is.
    ///
    /// The trap this exists to close is the second shape: reading
    /// `hybrid_layers` alone reports `false` for every layer of a pure-Mamba
    /// model, which builds an all-attention model that trains and looks
    /// plausible while being the wrong architecture entirely.
    pub fn is_ssm_layer(&self, layer_idx: usize) -> bool {
        match self.hybrid_layers.as_ref() {
            Some(hybrid) => hybrid.is_ssm_layer(layer_idx),
            None => self.ssm.is_some() && self.attention.is_none(),
        }
    }

    /// Whether layer `layer_idx` is an attention layer — the complement of
    /// [`is_ssm_layer`](Self::is_ssm_layer) over `0..num_layers`.
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        !self.is_ssm_layer(layer_idx)
    }

    /// The SSM and attention layer indices of `0..num_layers`, by
    /// [`is_ssm_layer`](Self::is_ssm_layer). A config without
    /// `hybrid_layers` gets its split from the same rule.
    pub fn layer_split(&self) -> HybridConfig {
        let (ssm_layers, attention_layers) =
            (0..self.num_layers).partition(|&layer| self.is_ssm_layer(layer));
        HybridConfig {
            ssm_layers,
            attention_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure-SSM config carries no `hybrid_layers` — there is no mix to
    /// describe — so reading that field alone would report every layer as
    /// attention and silently build the wrong architecture.
    #[test]
    fn every_layer_of_a_pure_ssm_config_is_an_ssm_layer() {
        let yaml = r#"
model_type: mamba2
vocab_size: 1000
hidden_size: 256
num_layers: 4
max_seq_len: 128
ssm:
  variant: mamba2
  num_heads: 8
  head_dim: 64
  state_size: 16
  chunk_size: 32
"#;
        let config: UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        config.validate().unwrap();
        assert!(config.hybrid_layers.is_none(), "precondition for this test");
        for layer in 0..config.num_layers {
            assert!(config.is_ssm_layer(layer), "layer {layer}");
            assert!(!config.is_attention_layer(layer), "layer {layer}");
        }
    }

    #[test]
    fn no_layer_of_an_attention_only_config_is_an_ssm_layer() {
        let yaml = r#"
model_type: llama
vocab_size: 1000
hidden_size: 256
num_layers: 4
max_seq_len: 128
attention:
  num_heads: 4
"#;
        let config: UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        for layer in 0..config.num_layers {
            assert!(!config.is_ssm_layer(layer), "layer {layer}");
        }
    }

    #[test]
    fn a_hybrid_config_splits_layers_exactly_as_listed() {
        let yaml = r#"
model_type: hybrid
vocab_size: 1000
hidden_size: 256
num_layers: 4
max_seq_len: 128
attention:
  num_heads: 4
ssm:
  variant: mamba2
  num_heads: 8
  head_dim: 64
  state_size: 16
  chunk_size: 32
hybrid_layers:
  ssm_layers: [0, 2]
  attention_layers: [1, 3]
"#;
        let config: UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        config.validate().unwrap();
        assert!(config.is_ssm_layer(0));
        assert!(!config.is_ssm_layer(1));
        assert!(config.is_ssm_layer(2));
        assert!(!config.is_ssm_layer(3));
        let split = config.layer_split();
        assert_eq!(split.ssm_layers, vec![0, 2]);
        assert_eq!(split.attention_layers, vec![1, 3]);
    }

    #[test]
    fn a_config_without_hybrid_layers_splits_by_its_sections() {
        let yaml = "model_type: llama\nvocab_size: 8\nhidden_size: 8\nnum_layers: 3\n\
                    max_seq_len: 8\nattention:\n  num_heads: 2\n";
        let config: UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        let split = config.layer_split();
        assert!(split.ssm_layers.is_empty());
        assert_eq!(split.attention_layers, vec![0, 1, 2]);
    }
}
