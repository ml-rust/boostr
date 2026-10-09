//! [`ClapConfig`]: a CLAP checkpoint's `config.json`, with transformers'
//! `ClapConfig`, `ClapAudioConfig` and `ClapTextConfig` defaults for every
//! field the file leaves out.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::nn::Activation;

/// The HTSAT audio tower: a Swin transformer over the log-mel image.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ClapAudioConfig {
    /// Attention window side, in patches.
    pub window_size: usize,
    /// Mel bins per frame.
    pub num_mel_bins: usize,
    /// Side of the square image the spectrogram is folded into.
    pub spec_size: usize,
    /// Activation of each block's MLP.
    pub hidden_act: String,
    /// Patch side.
    pub patch_size: usize,
    /// Patch stride, `[height, width]`.
    pub patch_stride: [usize; 2],
    /// Width of the pooled output, the projection's input.
    pub hidden_size: usize,
    /// Blocks per stage.
    pub depths: Vec<usize>,
    /// Attention heads per stage.
    pub num_attention_heads: Vec<usize>,
    /// Whether the checkpoint fuses local and global mel views.
    pub enable_fusion: bool,
    /// Channels of the patch embedding's input image.
    pub patch_embed_input_channels: usize,
    /// Width of the first stage.
    pub patch_embeds_hidden_size: usize,
    /// Whether the patch embedding ends in a LayerNorm.
    pub enable_patch_layer_norm: bool,
    /// Whether the query, key and value projections carry a bias.
    pub qkv_bias: bool,
    /// MLP width over block width.
    pub mlp_ratio: f64,
    /// LayerNorm epsilon of every block.
    pub layer_norm_eps: f64,
}

impl Default for ClapAudioConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            num_mel_bins: 64,
            spec_size: 256,
            hidden_act: "gelu".to_string(),
            patch_size: 4,
            patch_stride: [4, 4],
            hidden_size: 768,
            depths: vec![2, 2, 6, 2],
            num_attention_heads: vec![4, 8, 16, 32],
            enable_fusion: false,
            patch_embed_input_channels: 1,
            patch_embeds_hidden_size: 96,
            enable_patch_layer_norm: true,
            qkv_bias: true,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
        }
    }
}

/// The RoBERTa text tower.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ClapTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    /// The padding id, which RoBERTa's position numbering starts after.
    pub pad_token_id: i64,
}

impl Default for ClapTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 514,
            type_vocab_size: 1,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
        }
    }
}

/// `config.json` as read: the two towers and the shared projection.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ClapConfigFile {
    audio_config: ClapAudioConfig,
    text_config: ClapTextConfig,
    projection_dim: usize,
    projection_hidden_act: String,
}

impl Default for ClapConfigFile {
    fn default() -> Self {
        Self {
            audio_config: ClapAudioConfig::default(),
            text_config: ClapTextConfig::default(),
            projection_dim: 512,
            projection_hidden_act: "relu".to_string(),
        }
    }
}

/// A CLAP checkpoint's architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct ClapConfig {
    pub audio: ClapAudioConfig,
    pub text: ClapTextConfig,
    /// Width of both embeddings.
    pub projection_dim: usize,
    /// Activation between the two projection layers of each tower.
    pub projection_act: Activation,
    /// Activation of the audio blocks' MLPs.
    pub audio_act: Activation,
    /// Activation of the text layers' feed-forward.
    pub text_act: Activation,
}

impl ClapConfig {
    /// Parses `config.json` text and checks the architecture is one this
    /// port runs.
    ///
    /// Returns [`Error::ModelError`] for malformed JSON, an unknown
    /// activation, a fused checkpoint, overlapping patches, or tower sizes
    /// that disagree.
    pub fn from_json(text: &str) -> Result<Self> {
        let file: ClapConfigFile = serde_json::from_str(text).map_err(|e| Error::ModelError {
            reason: format!("CLAP config.json: {e}"),
        })?;
        let act = |name: &str| {
            Activation::from_hf_name(name).map_err(|_| Error::ModelError {
                reason: format!("CLAP config.json: unknown activation {name:?}"),
            })
        };
        let config = Self {
            projection_act: act(&file.projection_hidden_act)?,
            audio_act: act(&file.audio_config.hidden_act)?,
            text_act: act(&file.text_config.hidden_act)?,
            audio: file.audio_config,
            text: file.text_config,
            projection_dim: file.projection_dim,
        };
        config.check()?;
        Ok(config)
    }

    fn check(&self) -> Result<()> {
        let a = &self.audio;
        let refuse = |reason: String| Err(Error::ModelError { reason });
        if a.enable_fusion {
            return refuse(
                "CLAP config.json sets enable_fusion; this port runs unfused checkpoints only, \
                 such as laion/larger_clap_general"
                    .to_string(),
            );
        }
        if a.patch_stride != [a.patch_size, a.patch_size] {
            return refuse(format!(
                "CLAP patch stride {:?} differs from patch size {}; overlapping patches are \
                 not supported",
                a.patch_stride, a.patch_size
            ));
        }
        let stages = a.depths.len();
        if stages == 0 || a.num_attention_heads.len() != stages {
            return refuse(format!(
                "CLAP audio config has {} stages but {} head counts",
                stages,
                a.num_attention_heads.len()
            ));
        }
        let last = a.patch_embeds_hidden_size << (stages - 1);
        if last != a.hidden_size {
            return refuse(format!(
                "CLAP audio towers end at width {last} but hidden_size is {}",
                a.hidden_size
            ));
        }
        if a.patch_size == 0
            || a.window_size == 0
            || a.num_mel_bins == 0
            || !a.spec_size.is_multiple_of(a.patch_size)
            || !a.spec_size.is_multiple_of(a.num_mel_bins)
        {
            return refuse(format!(
                "CLAP spec_size {} must be a multiple of patch_size {} and num_mel_bins {}",
                a.spec_size, a.patch_size, a.num_mel_bins
            ));
        }
        Ok(())
    }

    /// Image rows per mel bin: `spec_size / num_mel_bins`.
    pub fn freq_ratio(&self) -> usize {
        self.audio.spec_size / self.audio.num_mel_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_clap_general_overrides_parse() {
        let text = r#"{"audio_config": {"depths": [2, 2, 12, 2], "hidden_size": 1024,
            "patch_embeds_hidden_size": 128}, "text_config": {}, "projection_dim": 512}"#;
        let c = ClapConfig::from_json(text).expect("config");
        assert_eq!(c.audio.depths, vec![2, 2, 12, 2]);
        assert_eq!(c.audio.window_size, 8);
        assert_eq!(c.text.max_position_embeddings, 514);
        assert_eq!(c.projection_act, Activation::Relu);
        assert_eq!(c.freq_ratio(), 4);
    }

    #[test]
    fn fused_and_inconsistent_checkpoints_are_refused() {
        assert!(ClapConfig::from_json(r#"{"audio_config": {"enable_fusion": true}}"#).is_err());
        // Default patch width 96 doubles to 768 over four stages; 1024 disagrees.
        assert!(ClapConfig::from_json(r#"{"audio_config": {"hidden_size": 1024}}"#).is_err());
        assert!(ClapConfig::from_json("not json").is_err());
    }
}
