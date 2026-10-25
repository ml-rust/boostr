//! [`LayerSpan`]: the contiguous layer range a pipeline stage model holds,
//! and the checkpoint tensors a load of that range reads.

use std::fmt;
use std::ops::Range;

use crate::error::{Error, Result};
use crate::format::safetensors_name_map::normalize_hf_name;
use crate::model::config::UniversalConfig;

/// Layers `start..end` of a model of `num_layers` layers.
///
/// The span that starts at layer 0 holds the token embedding. The span
/// that ends at `num_layers` holds the final norm and the LM head. A tied
/// head multiplies by the embedding, so that span reads the embedding too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpan {
    start: usize,
    end: usize,
    num_layers: usize,
}

impl LayerSpan {
    /// Layers `layers` of a model of `num_layers` layers.
    ///
    /// # Errors
    ///
    /// An empty range, or a range past `num_layers`.
    pub fn new(layers: Range<usize>, num_layers: usize) -> Result<Self> {
        if layers.start >= layers.end || layers.end > num_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "layers {}..{} are not a non-empty range inside the model's 0..{num_layers}",
                    layers.start, layers.end
                ),
            });
        }
        Ok(Self {
            start: layers.start,
            end: layers.end,
            num_layers,
        })
    }

    /// Layers `layers` of the model `config` describes, checked before any
    /// weight is read.
    ///
    /// # Errors
    ///
    /// A model [`layer_range_refusal`] refuses, or a range [`Self::new`]
    /// refuses.
    pub fn for_model(config: &UniversalConfig, layers: Range<usize>) -> Result<Self> {
        if let Some(reason) = layer_range_refusal(config) {
            return Err(Error::ModelError { reason });
        }
        Self::new(layers, config.num_layers)
    }

    /// First layer held.
    pub fn start(&self) -> usize {
        self.start
    }

    /// One past the last layer held.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Layers of the whole model.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The layers held.
    pub fn layers(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Whether the span holds the token embedding: it starts at layer 0.
    pub fn has_embedding(&self) -> bool {
        self.start == 0
    }

    /// Whether the span holds the final norm and LM head: it ends at the
    /// last layer.
    pub fn has_head(&self) -> bool {
        self.end == self.num_layers
    }

    /// Whether `start..end` lies inside the span.
    pub fn contains(&self, start: usize, end: usize) -> bool {
        self.start <= start && end <= self.end
    }

    /// Whether a load of this span reads the tensor `name`, in the Llama
    /// layout `restructure_hf_checkpoint` and the GGUF name map produce.
    ///
    /// It reads `model.layers.{i}.*` for each held layer `i`, and nothing
    /// for any other layer. The embedding goes to the span that holds it,
    /// and to the head span of a tied model. `model.norm.*` and an untied
    /// `lm_head.*` go to the head span. A Llama reads no other tensor, so
    /// a vision tower, an audio encoder or an MTP head is never read.
    pub fn reads(&self, config: &UniversalConfig, name: &str) -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            return layer_of(rest).is_some_and(|i| self.layers().contains(&i));
        }
        let tied = config.tie_word_embeddings;
        if name.starts_with("model.embed_tokens.") {
            return self.has_embedding() || (self.has_head() && tied);
        }
        if name.starts_with("model.norm.") {
            return self.has_head();
        }
        if name.starts_with("lm_head.") {
            return self.has_head() && !tied;
        }
        false
    }

    /// [`Self::reads`] for the tensor a Hugging Face checkpoint stores as
    /// `file_name`, renamed first as `restructure_hf_checkpoint` renames it.
    pub fn reads_hf_name(&self, config: &UniversalConfig, file_name: &str) -> bool {
        self.reads(config, &normalize_hf_name(&config.model_type, file_name))
    }
}

impl fmt::Display for LayerSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "layers {}..{} of {}",
            self.start, self.end, self.num_layers
        )?;
        if self.has_embedding() {
            f.write_str(", with the embedding")?;
        }
        if self.has_head() {
            f.write_str(", with the LM head")?;
        }
        Ok(())
    }
}

/// The layer index that leads `rest`, the part of a name after
/// `model.layers.`.
fn layer_of(rest: &str) -> Option<usize> {
    rest.split_once('.')?.0.parse().ok()
}

/// Why the model `config` describes cannot load as a layer range, or
/// `None` when it can.
///
/// A range load builds a [`LlamaStage`](super::LlamaStage). A model with
/// per-layer recurrent state needs that state beside the layer, and a
/// stage carries a KV cache only, so it is refused before any weight is
/// read. A multimodal checkpoint loads its text model: a stage runs token
/// ids and activations only, so its vision and audio towers stay unread.
pub fn layer_range_refusal(config: &UniversalConfig) -> Option<String> {
    if config.attention.as_ref().is_some_and(|a| a.is_mla()) {
        return Some(
            "an MLA model loads as the hybrid model, and a layer range holds Llama layers \
             only. Load the whole model on one node"
                .into(),
        );
    }
    match config.model_type.as_str() {
        "mamba1" | "mamba2" | "mamba3" | "hybrid" => Some(format!(
            "a {} model carries SSM state in its layers, and a layer range carries only a \
             KV cache. Load the whole model on one node",
            config.model_type
        )),
        "qwen35" => Some(
            "a qwen35 model carries Gated DeltaNet state in its layers, and a layer range \
             carries only a KV cache. Load the whole model on one node"
                .into(),
        ),
        other if config.attention.is_none() => Some(format!(
            "model type '{other}' has no attention config, and only a Llama-family \
             transformer loads as a layer range"
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(tied: bool) -> UniversalConfig {
        let yaml = format!(
            "model_type: llama\nvocab_size: 16\nhidden_size: 8\nnum_layers: 4\n\
             max_seq_len: 16\nintermediate_size: 16\ntie_word_embeddings: {tied}\n\
             attention:\n  num_heads: 2\n"
        );
        serde_saphyr::from_str(&yaml).unwrap()
    }

    #[test]
    fn test_a_span_is_a_non_empty_range_inside_the_model() {
        let span = LayerSpan::new(1..3, 4).unwrap();
        assert_eq!((span.start(), span.end(), span.num_layers()), (1, 3, 4));
        assert!(!span.has_embedding() && !span.has_head());
        assert!(span.contains(1, 3) && span.contains(2, 3) && !span.contains(0, 2));
        assert!(LayerSpan::new(2..2, 4).is_err());
        assert!(LayerSpan::new(3..5, 4).is_err());
        let ends = LayerSpan::new(0..4, 4).unwrap();
        assert!(ends.has_embedding() && ends.has_head());
        assert_eq!(
            ends.to_string(),
            "layers 0..4 of 4, with the embedding, with the LM head"
        );
    }

    #[test]
    fn test_a_span_reads_its_own_layers_only() {
        let c = config(false);
        let mid = LayerSpan::new(1..3, 4).unwrap();
        for name in [
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.mlp.down_proj.weight",
        ] {
            assert!(mid.reads(&c, name), "{name}");
        }
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.3.input_layernorm.weight",
            "model.layers.10.input_layernorm.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "vision_model.encoder.layers.1.fc1.weight",
            "model.layers.x.weight",
        ] {
            assert!(!mid.reads(&c, name), "{name}");
        }
    }

    #[test]
    fn test_the_end_spans_read_the_embedding_and_the_head() {
        let untied = config(false);
        let first = LayerSpan::new(0..1, 4).unwrap();
        let last = LayerSpan::new(3..4, 4).unwrap();
        assert!(first.reads(&untied, "model.embed_tokens.weight"));
        assert!(!first.reads(&untied, "lm_head.weight"));
        assert!(!first.reads(&untied, "model.norm.weight"));
        assert!(!last.reads(&untied, "model.embed_tokens.weight"));
        assert!(last.reads(&untied, "lm_head.weight"));
        assert!(last.reads(&untied, "model.norm.bias"));
        let tied = config(true);
        assert!(last.reads(&tied, "model.embed_tokens.weight"));
        assert!(!last.reads(&tied, "lm_head.weight"));
    }

    #[test]
    fn test_hf_names_are_renamed_before_the_check() {
        let mut c = config(false);
        c.model_type = "falcon".into();
        let span = LayerSpan::new(2..3, 4).unwrap();
        let mapped = normalize_hf_name("falcon", "transformer.h.2.mlp.dense_4h_to_h.weight");
        assert!(mapped.starts_with("model.layers.2."), "{mapped}");
        assert!(span.reads_hf_name(&c, "transformer.h.2.mlp.dense_4h_to_h.weight"));
        assert!(!span.reads_hf_name(&c, "transformer.h.1.mlp.dense_4h_to_h.weight"));
    }

    #[test]
    fn test_state_models_are_refused_before_a_read() {
        let mut c = config(false);
        assert!(LayerSpan::for_model(&c, 0..2).is_ok());
        for model_type in ["mamba2", "hybrid", "qwen35"] {
            c.model_type = model_type.into();
            let err = LayerSpan::for_model(&c, 0..2).unwrap_err().to_string();
            assert!(err.contains("carries"), "{model_type}: {err}");
        }
        c.model_type = "llama".into();
        c.attention = None;
        assert!(
            layer_range_refusal(&c)
                .unwrap()
                .contains("no attention config")
        );
    }
}
