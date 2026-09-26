//! `VitsConfig`: parses the HF `VitsConfig` (`facebook/mms-tts-*`, the MMS-TTS
//! family) `config.json`. Field names and defaults match
//! `transformers.models.vits.configuration_vits.VitsConfig` (checked against
//! installed `transformers` 4.57, 2026-09). Covers every field the full VITS
//! model needs (text encoder, duration predictor, flow, posterior encoder,
//! HiFi-GAN decoder) — later units read the fields their block needs.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level VITS (MMS-TTS) model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitsConfig {
    /// Text token vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Text-encoder / flow hidden size.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Text-encoder transformer layer count.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    /// Text-encoder attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// Relative-position attention window (each side). `0` disables it
    /// (HF's `if config.window_size:` is falsy at `0`).
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    /// Whether the text-encoder's attention/feed-forward projections carry a
    /// bias term.
    #[serde(default = "default_use_bias")]
    pub use_bias: bool,
    /// Text-encoder feed-forward inner width.
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    /// Text-encoder feed-forward conv1d kernel size.
    #[serde(default = "default_ffn_kernel_size")]
    pub ffn_kernel_size: usize,
    /// Flow / prior latent width (the text encoder's `project` conv emits
    /// `2 * flow_size` channels, split into means and log-variances).
    #[serde(default = "default_flow_size")]
    pub flow_size: usize,
    /// Target linear-spectrogram bin count (posterior encoder input width).
    #[serde(default = "default_spectrogram_bins")]
    pub spectrogram_bins: usize,
    /// Text-encoder feed-forward activation (`"relu"` on every published MMS
    /// checkpoint).
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Text-encoder `LayerNorm` epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// `true` selects the flow-based (stochastic) duration predictor over the
    /// deterministic one. Every published MMS-TTS checkpoint sets this `true`.
    #[serde(default = "default_use_stochastic_duration_prediction")]
    pub use_stochastic_duration_prediction: bool,
    /// Speaker count. `1` for every MMS-TTS single-speaker checkpoint.
    #[serde(default = "default_num_speakers")]
    pub num_speakers: usize,
    /// Speaker-embedding width. `0` means no `cond_layer` exists in the
    /// posterior encoder / flow / duration predictor (single-speaker model).
    #[serde(default = "default_speaker_embedding_size")]
    pub speaker_embedding_size: usize,
    /// HiFi-GAN decoder initial channel count before upsampling.
    #[serde(default = "default_upsample_initial_channel")]
    pub upsample_initial_channel: usize,
    /// HiFi-GAN decoder upsample stride per stage.
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<usize>,
    /// HiFi-GAN decoder transposed-conv kernel size per stage.
    #[serde(default = "default_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<usize>,
    /// HiFi-GAN resblock kernel sizes.
    #[serde(default = "default_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<usize>,
    /// HiFi-GAN resblock dilations, indexed `[resblock_idx][conv_idx]`.
    #[serde(default = "default_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// LeakyReLU negative slope used throughout the decoder.
    #[serde(default = "default_leaky_relu_slope")]
    pub leaky_relu_slope: f64,
    /// Duration-predictor / prior flow depth-separable conv channel count.
    #[serde(default = "default_depth_separable_channels")]
    pub depth_separable_channels: usize,
    /// Duration-predictor / prior flow depth-separable conv layer count.
    #[serde(default = "default_depth_separable_num_layers")]
    pub depth_separable_num_layers: usize,
    /// Stochastic duration predictor: rational-quadratic spline bin count.
    #[serde(default = "default_duration_predictor_flow_bins")]
    pub duration_predictor_flow_bins: usize,
    /// Stochastic duration predictor: spline tail bound.
    #[serde(default = "default_duration_predictor_tail_bound")]
    pub duration_predictor_tail_bound: f64,
    /// Stochastic duration predictor: conv kernel size.
    #[serde(default = "default_duration_predictor_kernel_size")]
    pub duration_predictor_kernel_size: usize,
    /// Stochastic duration predictor: number of stacked flows.
    #[serde(default = "default_duration_predictor_num_flows")]
    pub duration_predictor_num_flows: usize,
    /// Stochastic duration predictor: conv filter width.
    #[serde(default = "default_duration_predictor_filter_channels")]
    pub duration_predictor_filter_channels: usize,
    /// Prior encoder (normalizing flow over the text prior): number of flows.
    #[serde(default = "default_prior_encoder_num_flows")]
    pub prior_encoder_num_flows: usize,
    /// Prior encoder: WaveNet layers per flow.
    #[serde(default = "default_prior_encoder_num_wavenet_layers")]
    pub prior_encoder_num_wavenet_layers: usize,
    /// Posterior encoder: WaveNet layer count.
    #[serde(default = "default_posterior_encoder_num_wavenet_layers")]
    pub posterior_encoder_num_wavenet_layers: usize,
    /// WaveNet dilated-conv kernel size (shared by the prior flow and the
    /// posterior encoder).
    #[serde(default = "default_wavenet_kernel_size")]
    pub wavenet_kernel_size: usize,
    /// WaveNet dilation growth rate.
    #[serde(default = "default_wavenet_dilation_rate")]
    pub wavenet_dilation_rate: usize,
    /// Sample rate of the generated waveform, Hz.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
    /// Inference-time speaking-rate multiplier (`1.0 / speaking_rate` scales
    /// predicted durations).
    #[serde(default = "default_speaking_rate")]
    pub speaking_rate: f64,
    /// Inference-time Gaussian noise scale applied to the prior latents.
    #[serde(default = "default_noise_scale")]
    pub noise_scale: f64,
    /// Inference-time Gaussian noise scale applied inside the stochastic
    /// duration predictor's flow.
    #[serde(default = "default_noise_scale_duration")]
    pub noise_scale_duration: f64,
}

fn default_vocab_size() -> usize {
    38
}
fn default_hidden_size() -> usize {
    192
}
fn default_num_hidden_layers() -> usize {
    6
}
fn default_num_attention_heads() -> usize {
    2
}
fn default_window_size() -> usize {
    4
}
fn default_use_bias() -> bool {
    true
}
fn default_ffn_dim() -> usize {
    768
}
fn default_ffn_kernel_size() -> usize {
    3
}
fn default_flow_size() -> usize {
    192
}
fn default_spectrogram_bins() -> usize {
    513
}
fn default_hidden_act() -> String {
    "relu".to_string()
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_use_stochastic_duration_prediction() -> bool {
    true
}
fn default_num_speakers() -> usize {
    1
}
fn default_speaker_embedding_size() -> usize {
    0
}
fn default_upsample_initial_channel() -> usize {
    512
}
fn default_upsample_rates() -> Vec<usize> {
    vec![8, 8, 2, 2]
}
fn default_upsample_kernel_sizes() -> Vec<usize> {
    vec![16, 16, 4, 4]
}
fn default_resblock_kernel_sizes() -> Vec<usize> {
    vec![3, 7, 11]
}
fn default_resblock_dilation_sizes() -> Vec<Vec<usize>> {
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
}
fn default_leaky_relu_slope() -> f64 {
    0.1
}
fn default_depth_separable_channels() -> usize {
    2
}
fn default_depth_separable_num_layers() -> usize {
    3
}
fn default_duration_predictor_flow_bins() -> usize {
    10
}
fn default_duration_predictor_tail_bound() -> f64 {
    5.0
}
fn default_duration_predictor_kernel_size() -> usize {
    3
}
fn default_duration_predictor_num_flows() -> usize {
    4
}
fn default_duration_predictor_filter_channels() -> usize {
    256
}
fn default_prior_encoder_num_flows() -> usize {
    4
}
fn default_prior_encoder_num_wavenet_layers() -> usize {
    4
}
fn default_posterior_encoder_num_wavenet_layers() -> usize {
    16
}
fn default_wavenet_kernel_size() -> usize {
    5
}
fn default_wavenet_dilation_rate() -> usize {
    1
}
fn default_sampling_rate() -> usize {
    16_000
}
fn default_speaking_rate() -> f64 {
    1.0
}
fn default_noise_scale() -> f64 {
    0.667
}
fn default_noise_scale_duration() -> f64 {
    0.8
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            window_size: default_window_size(),
            use_bias: default_use_bias(),
            ffn_dim: default_ffn_dim(),
            ffn_kernel_size: default_ffn_kernel_size(),
            flow_size: default_flow_size(),
            spectrogram_bins: default_spectrogram_bins(),
            hidden_act: default_hidden_act(),
            layer_norm_eps: default_layer_norm_eps(),
            use_stochastic_duration_prediction: default_use_stochastic_duration_prediction(),
            num_speakers: default_num_speakers(),
            speaker_embedding_size: default_speaker_embedding_size(),
            upsample_initial_channel: default_upsample_initial_channel(),
            upsample_rates: default_upsample_rates(),
            upsample_kernel_sizes: default_upsample_kernel_sizes(),
            resblock_kernel_sizes: default_resblock_kernel_sizes(),
            resblock_dilation_sizes: default_resblock_dilation_sizes(),
            leaky_relu_slope: default_leaky_relu_slope(),
            depth_separable_channels: default_depth_separable_channels(),
            depth_separable_num_layers: default_depth_separable_num_layers(),
            duration_predictor_flow_bins: default_duration_predictor_flow_bins(),
            duration_predictor_tail_bound: default_duration_predictor_tail_bound(),
            duration_predictor_kernel_size: default_duration_predictor_kernel_size(),
            duration_predictor_num_flows: default_duration_predictor_num_flows(),
            duration_predictor_filter_channels: default_duration_predictor_filter_channels(),
            prior_encoder_num_flows: default_prior_encoder_num_flows(),
            prior_encoder_num_wavenet_layers: default_prior_encoder_num_wavenet_layers(),
            posterior_encoder_num_wavenet_layers: default_posterior_encoder_num_wavenet_layers(),
            wavenet_kernel_size: default_wavenet_kernel_size(),
            wavenet_dilation_rate: default_wavenet_dilation_rate(),
            sampling_rate: default_sampling_rate(),
            speaking_rate: default_speaking_rate(),
            noise_scale: default_noise_scale(),
            noise_scale_duration: default_noise_scale_duration(),
        }
    }
}

impl VitsConfig {
    /// Parse from an on-disk `config.json`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
            reason: format!("reading vits config {}: {e}", path.display()),
        })?;
        Self::from_json_bytes(&bytes)
    }

    /// Parse from in-memory JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| Error::ModelError {
            reason: format!("invalid vits config.json: {e}"),
        })
    }

    /// `window_size == 0` means HF's relative-position attention is off
    /// (`if config.window_size:` is falsy at `0`); otherwise the window size.
    pub fn attention_window_size(&self) -> Option<usize> {
        (self.window_size > 0).then_some(self.window_size)
    }

    /// Basic sanity checks applied after parsing.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::ModelError {
                reason: format!(
                    "hidden_size {} must be positive and divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        if self.vocab_size == 0 {
            return Err(Error::ModelError {
                reason: "vocab_size must be > 0".into(),
            });
        }
        if self.flow_size == 0 {
            return Err(Error::ModelError {
                reason: "flow_size must be > 0".into(),
            });
        }
        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "upsample_rates has {} entries but upsample_kernel_sizes has {}",
                    self.upsample_rates.len(),
                    self.upsample_kernel_sizes.len()
                ),
            });
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "resblock_kernel_sizes has {} entries but resblock_dilation_sizes has {}",
                    self.resblock_kernel_sizes.len(),
                    self.resblock_dilation_sizes.len()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_hf_vits_config() {
        let cfg = VitsConfig::default();
        assert_eq!(cfg.vocab_size, 38);
        assert_eq!(cfg.hidden_size, 192);
        assert_eq!(cfg.num_hidden_layers, 6);
        assert_eq!(cfg.num_attention_heads, 2);
        assert_eq!(cfg.window_size, 4);
        assert_eq!(cfg.ffn_dim, 768);
        assert_eq!(cfg.flow_size, 192);
        assert_eq!(cfg.hidden_act, "relu");
        assert_eq!(cfg.sampling_rate, 16_000);
        assert_eq!(cfg.noise_scale, 0.667);
        assert_eq!(cfg.noise_scale_duration, 0.8);
        cfg.validate().unwrap();
    }

    #[test]
    fn parses_minimal_json_with_all_defaults() {
        let cfg = VitsConfig::from_json_bytes(b"{}").unwrap();
        assert_eq!(cfg.vocab_size, default_vocab_size());
    }

    #[test]
    fn parses_real_mms_tts_zlm_config_shape() {
        // Mirrors facebook/mms-tts-zlm/config.json's overrides of the class
        // defaults (vocab_size and window_size differ per checkpoint).
        let json = br#"{"vocab_size": 34, "window_size": 4, "speaker_embedding_size": 0}"#;
        let cfg = VitsConfig::from_json_bytes(json).unwrap();
        assert_eq!(cfg.vocab_size, 34);
        assert_eq!(cfg.attention_window_size(), Some(4));
        assert_eq!(cfg.speaker_embedding_size, 0);
        cfg.validate().unwrap();
    }

    #[test]
    fn attention_window_size_is_none_when_zero() {
        let cfg = VitsConfig {
            window_size: 0,
            ..Default::default()
        };
        assert_eq!(cfg.attention_window_size(), None);
    }

    #[test]
    fn validate_rejects_indivisible_hidden_size() {
        let cfg = VitsConfig {
            hidden_size: 190,
            num_attention_heads: 3,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_mismatched_upsample_lists() {
        let cfg = VitsConfig {
            upsample_rates: vec![8, 8],
            upsample_kernel_sizes: vec![16, 16, 4],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_json_surfaces_model_error() {
        let err = VitsConfig::from_json_bytes(b"not json").unwrap_err();
        match err {
            Error::ModelError { reason } => assert!(reason.contains("invalid vits config.json")),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
