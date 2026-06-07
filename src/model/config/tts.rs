//! TTS model configs.
//!
//! Lives alongside `AudioConfig` (Whisper / ASR), `VisionConfig` (CLIP /
//! SigLIP), `AttentionConfig` (LLM), etc. — one module per model family in
//! `boostr::model::config`. TTS *architectures* live under
//! `boostr::model::audio::<model>/`; the loader for each uses the config type
//! defined here.
//!
//! Currently only Kokoro is supported; new TTS families (e.g. a future XTTS
//! or Dia port) add their own config struct in this file.
//!
//! Kokoro defaults match the `hexgrad/Kokoro-82M` checkpoint shipped on
//! HuggingFace (82M parameters, 24 kHz output, 178-symbol phoneme
//! vocabulary). When fields are missing from the JSON, we fall back to those
//! defaults rather than erroring, so ad-hoc configs (e.g. Kokoro-mini
//! variants) still load.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level Kokoro model configuration.
///
/// Defaults derived from the upstream `hexgrad/Kokoro-82M/config.json` and
/// the reference `kokoro/model.py` + `istftnet.py` sources (2026-04). These
/// are the values actually shipped with the public checkpoint, not guesses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    /// Phoneme vocabulary size (default 178).
    #[serde(default = "default_n_symbols")]
    pub n_symbols: usize,
    /// Text-encoder / decoder main hidden size (default 512).
    #[serde(default = "default_hidden_dim")]
    pub hidden_dim: usize,
    /// Per-head style dimensionality (default 128). Voice files contain a
    /// 256-d vector = concat of decoder style (128) + predictor style (128);
    /// each half feeds its respective module. See [`voice_style_dim`].
    #[serde(default = "default_style_dim")]
    pub style_dim: usize,
    /// Full voice-file style width (default 256 = 2 × `style_dim`).
    #[serde(default = "default_voice_style_dim")]
    pub voice_style_dim: usize,
    /// Text-encoder conv kernel size (default 5).
    #[serde(default = "default_text_conv_kernel")]
    pub text_conv_kernel: usize,
    /// Number of text-encoder conv blocks (default 3).
    #[serde(default = "default_text_conv_depth")]
    pub text_conv_depth: usize,
    /// Duration-predictor classification bins (default 50 = `max_dur`). The
    /// duration head emits a `[T_phon, 50]` logits tensor, not a scalar.
    #[serde(default = "default_max_dur")]
    pub max_dur: usize,
    /// LeakyReLU negative slope used throughout (default 0.2).
    #[serde(default = "default_leaky_slope")]
    pub leaky_slope: f64,
    /// Vocoder sample rate in Hz (default 24000).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Generator iSTFT n_fft (default 20 — Kokoro's small-FFT vocoder).
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    /// Generator iSTFT hop length in samples (default 5).
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    /// Generator upsample ratios (default `[10, 6]`). Product × `hop_length`
    /// × (n_fft/2+... actually simpler: total upsample = product of ratios).
    #[serde(default = "default_upsample_ratios")]
    pub upsample_ratios: Vec<usize>,
    /// Generator upsample kernel sizes, one per stage (default `[20, 12]`).
    #[serde(default = "default_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<usize>,
    /// Generator initial channel count before upsampling (default 512).
    #[serde(default = "default_upsample_initial_channel")]
    pub upsample_initial_channel: usize,
    /// Resblock kernel sizes applied per upsample stage (default `[3, 7, 11]`).
    /// Each upsample is followed by `len(resblock_kernel_sizes)` `AdaINResBlock1`
    /// blocks, each with `len(resblock_dilation_sizes[i])` conv pairs.
    #[serde(default = "default_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<usize>,
    /// Resblock dilation sizes, indexed `[resblock_idx][conv_idx]` (default
    /// `[[1,3,5], [1,3,5], [1,3,5]]`).
    #[serde(default = "default_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Harmonic count for source-filter excitation generator (default 8).
    #[serde(default = "default_harmonic_num")]
    pub harmonic_num: usize,
    /// ALBERT text-backbone hidden size (default 768). The `bert_encoder`
    /// linear projects `bert_hidden_size → hidden_dim`.
    #[serde(default = "default_bert_hidden_size")]
    pub bert_hidden_size: usize,
    /// ALBERT layer count (default 12).
    #[serde(default = "default_bert_num_layers")]
    pub bert_num_layers: usize,
    /// ALBERT attention heads (default 12).
    #[serde(default = "default_bert_num_heads")]
    pub bert_num_heads: usize,
    /// ALBERT embedding size pre-projection (default 128 — ALBERT factorizes
    /// the embedding into a low-dim table + linear projection).
    #[serde(default = "default_bert_embedding_size")]
    pub bert_embedding_size: usize,
    /// ALBERT feed-forward intermediate size (default 2048 — Kokoro's plBERT
    /// backbone uses `intermediate_size = 2048`, not the usual `4 × hidden`).
    #[serde(default = "default_bert_intermediate_size")]
    pub bert_intermediate_size: usize,
}

fn default_n_symbols() -> usize {
    178
}
fn default_hidden_dim() -> usize {
    512
}
fn default_style_dim() -> usize {
    128
}
fn default_voice_style_dim() -> usize {
    256
}
fn default_text_conv_kernel() -> usize {
    5
}
fn default_text_conv_depth() -> usize {
    3
}
fn default_max_dur() -> usize {
    50
}
fn default_leaky_slope() -> f64 {
    0.2
}
fn default_sample_rate() -> u32 {
    24_000
}
fn default_n_fft() -> usize {
    20
}
fn default_hop_length() -> usize {
    5
}
fn default_upsample_ratios() -> Vec<usize> {
    vec![10, 6]
}
fn default_upsample_kernel_sizes() -> Vec<usize> {
    vec![20, 12]
}
fn default_upsample_initial_channel() -> usize {
    512
}
fn default_resblock_kernel_sizes() -> Vec<usize> {
    vec![3, 7, 11]
}
fn default_resblock_dilation_sizes() -> Vec<Vec<usize>> {
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
}
fn default_harmonic_num() -> usize {
    8
}
fn default_bert_hidden_size() -> usize {
    768
}
fn default_bert_num_layers() -> usize {
    12
}
fn default_bert_num_heads() -> usize {
    12
}
fn default_bert_embedding_size() -> usize {
    128
}
fn default_bert_intermediate_size() -> usize {
    2048
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            n_symbols: default_n_symbols(),
            hidden_dim: default_hidden_dim(),
            style_dim: default_style_dim(),
            voice_style_dim: default_voice_style_dim(),
            text_conv_kernel: default_text_conv_kernel(),
            text_conv_depth: default_text_conv_depth(),
            max_dur: default_max_dur(),
            leaky_slope: default_leaky_slope(),
            sample_rate: default_sample_rate(),
            n_fft: default_n_fft(),
            hop_length: default_hop_length(),
            upsample_ratios: default_upsample_ratios(),
            upsample_kernel_sizes: default_upsample_kernel_sizes(),
            upsample_initial_channel: default_upsample_initial_channel(),
            resblock_kernel_sizes: default_resblock_kernel_sizes(),
            resblock_dilation_sizes: default_resblock_dilation_sizes(),
            harmonic_num: default_harmonic_num(),
            bert_hidden_size: default_bert_hidden_size(),
            bert_num_layers: default_bert_num_layers(),
            bert_num_heads: default_bert_num_heads(),
            bert_embedding_size: default_bert_embedding_size(),
            bert_intermediate_size: default_bert_intermediate_size(),
        }
    }
}

impl KokoroConfig {
    /// Parse from an on-disk `config.json`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
            reason: format!("reading kokoro config {}: {e}", path.display()),
        })?;
        Self::from_json_bytes(&bytes)
    }

    /// Parse from in-memory JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| Error::ModelError {
            reason: format!("invalid kokoro config.json: {e}"),
        })
    }

    /// Product of upsample ratios — how much the decoder lifts latent-rate
    /// frames toward waveform samples.
    pub fn total_upsample(&self) -> usize {
        self.upsample_ratios.iter().product()
    }

    /// Basic sanity checks applied after parsing. Separate from
    /// `from_json_*` so that callers can decide whether to enforce (e.g.
    /// during `load_kokoro`) or skip (e.g. during CLI `blazr list`).
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 || !self.hidden_dim.is_multiple_of(2) {
            return Err(Error::ModelError {
                reason: format!(
                    "hidden_dim must be positive and even (BiLSTM splits it), got {}",
                    self.hidden_dim
                ),
            });
        }
        if self.n_symbols == 0 {
            return Err(Error::ModelError {
                reason: "n_symbols must be > 0".into(),
            });
        }
        if self.sample_rate == 0 {
            return Err(Error::ModelError {
                reason: "sample_rate must be > 0".into(),
            });
        }
        if self.n_fft == 0 || self.hop_length == 0 {
            return Err(Error::ModelError {
                reason: "n_fft and hop_length must be > 0".into(),
            });
        }
        if self.upsample_ratios.is_empty() {
            return Err(Error::ModelError {
                reason: "upsample_ratios must have at least one entry".into(),
            });
        }
        if self.upsample_ratios.contains(&0) {
            return Err(Error::ModelError {
                reason: "upsample_ratios entries must be > 0".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_kokoro_82m() {
        let cfg = KokoroConfig::default();
        assert_eq!(cfg.n_symbols, 178);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.style_dim, 128);
        assert_eq!(cfg.voice_style_dim, 256);
        assert_eq!(cfg.voice_style_dim, 2 * cfg.style_dim);
        assert_eq!(cfg.max_dur, 50);
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.upsample_ratios, vec![10, 6]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![20, 12]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(cfg.harmonic_num, 8);
        assert_eq!(cfg.bert_hidden_size, 768);
        cfg.validate().unwrap();
    }

    #[test]
    fn parses_minimal_json_with_all_defaults() {
        let json = b"{}";
        let cfg = KokoroConfig::from_json_bytes(json).unwrap();
        assert_eq!(cfg.n_symbols, default_n_symbols());
    }

    #[test]
    fn partial_overrides_keep_remaining_defaults() {
        let json = br#"{"n_symbols": 200, "sample_rate": 22050}"#;
        let cfg = KokoroConfig::from_json_bytes(json).unwrap();
        assert_eq!(cfg.n_symbols, 200);
        assert_eq!(cfg.sample_rate, 22_050);
        assert_eq!(cfg.hidden_dim, default_hidden_dim());
    }

    #[test]
    fn total_upsample_multiplies_ratios() {
        let cfg = KokoroConfig {
            upsample_ratios: vec![2, 3, 5],
            ..Default::default()
        };
        assert_eq!(cfg.total_upsample(), 30);
    }

    #[test]
    fn validate_rejects_odd_hidden_dim() {
        let cfg = KokoroConfig {
            hidden_dim: 7,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_upsample() {
        let cfg = KokoroConfig {
            upsample_ratios: vec![],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_upsample_entry() {
        let cfg = KokoroConfig {
            upsample_ratios: vec![2, 0, 5],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_json_surfaces_model_error() {
        let err = KokoroConfig::from_json_bytes(b"not json").unwrap_err();
        match err {
            Error::ModelError { reason } => {
                assert!(reason.contains("invalid kokoro config.json"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
