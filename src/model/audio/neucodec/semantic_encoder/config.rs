//! Geometry of NeuCodec's semantic branch — a 16-layer Wav2Vec2-BERT
//! conformer encoder.
//!
//! Defaults are the values carried by the released `neuphonic/neucodec`
//! checkpoint's `semantic_encoder.*` tensors. Every loader shape check is
//! derived from this struct, so a config that disagrees with a checkpoint
//! fails loudly at load time instead of silently building a wrong model.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::fbank::STACKED_DIM;

/// Dimensions and hyperparameters of the Wav2Vec2-BERT semantic encoder.
///
/// Deliberately absent: dropout, layerdrop, and attention-mask settings. This
/// port is inference-only over a single utterance, where all three are exact
/// no-ops; modelling them would add branches that can only ever be disabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticEncoderConfig {
    /// Residual-stream width of every conformer layer (checkpoint: 1024).
    pub hidden_size: usize,
    /// Attention heads per layer (checkpoint: 16).
    pub num_heads: usize,
    /// Per-head dimension (checkpoint: 64, so `num_heads * head_dim == hidden_size`).
    pub head_dim: usize,
    /// Hidden width of both feed-forward modules (checkpoint: 4096).
    pub intermediate_size: usize,
    /// Number of conformer layers (checkpoint: 16).
    pub num_layers: usize,
    /// Epsilon shared by every `LayerNorm` in the branch (checkpoint: 1e-5).
    pub layer_norm_eps: f32,
    /// Kernel size of the convolution module's depthwise conv (checkpoint: 31).
    pub conv_depthwise_kernel_size: usize,
    /// Width of the stacked filterbank features entering the feature
    /// projection (checkpoint: 160).
    pub feature_projection_input_dim: usize,
    /// Largest backward relative distance the position table represents
    /// (checkpoint: 64). Distances below `-left_max` are clamped to it.
    pub left_max_position_embeddings: usize,
    /// Largest forward relative distance the position table represents
    /// (checkpoint: 8). The window is ASYMMETRIC — see
    /// [`super::attention`] for why that is not a typo.
    pub right_max_position_embeddings: usize,
}

impl Default for SemanticEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 4096,
            num_layers: 16,
            layer_norm_eps: 1e-5,
            conv_depthwise_kernel_size: 31,
            feature_projection_input_dim: STACKED_DIM,
            left_max_position_embeddings: 64,
            right_max_position_embeddings: 8,
        }
    }
}

impl SemanticEncoderConfig {
    /// Number of rows in `self_attn.distance_embedding.weight`.
    ///
    /// One row per representable clamped distance in
    /// `-left_max ..= +right_max`, inclusive on both ends — hence the `+ 1`
    /// for distance zero. With the checkpoint's `64`/`8` this is `73`.
    pub fn distance_embedding_len(&self) -> usize {
        self.left_max_position_embeddings + self.right_max_position_embeddings + 1
    }

    /// Reject a config that cannot describe any real checkpoint.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_heads/head_dim",
                reason: "must both be > 0".into(),
            });
        }
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "hidden_size",
                reason: format!(
                    "num_heads*head_dim ({}) must equal hidden_size ({})",
                    self.num_heads * self.head_dim,
                    self.hidden_size
                ),
            });
        }
        if self.num_layers == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_layers",
                reason: "must be > 0".into(),
            });
        }
        if self.intermediate_size == 0 || self.feature_projection_input_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "intermediate_size/feature_projection_input_dim",
                reason: "must both be > 0".into(),
            });
        }
        if self.conv_depthwise_kernel_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "conv_depthwise_kernel_size",
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_checkpoint_geometry() {
        let cfg = SemanticEncoderConfig::default();
        cfg.validate().expect("default config must be valid");
        assert_eq!(cfg.distance_embedding_len(), 73);
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
    }

    #[test]
    fn rejects_head_geometry_mismatch() {
        let cfg = SemanticEncoderConfig {
            num_heads: 12,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }
}
