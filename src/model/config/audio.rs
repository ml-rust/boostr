use serde::{Deserialize, Serialize};

/// Audio encoder configuration for multimodal models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Encoder type: "whisper"
    pub encoder_type: String,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of mel filterbank bins
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    /// Maximum audio length in frames
    #[serde(default = "default_max_audio_len")]
    pub max_audio_len: usize,
    /// Projector type: "linear", "mlp"
    #[serde(default = "default_audio_projector_type")]
    pub projector_type: String,
    /// Decoder vocabulary size (Whisper: 51865 for multilingual-v1, 51866 for v3)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Number of decoder transformer layers. Defaults to `num_layers` when `None`.
    #[serde(default)]
    pub decoder_layers: Option<usize>,
    /// Maximum decoder sequence length (Whisper: 448).
    #[serde(default = "default_max_target_positions")]
    pub max_target_positions: usize,
    /// FFN intermediate size (Whisper: 4 * hidden_size).
    #[serde(default)]
    pub intermediate_size: Option<usize>,
}

impl AudioConfig {
    pub fn decoder_layer_count(&self) -> usize {
        self.decoder_layers.unwrap_or(self.num_layers)
    }

    pub fn ffn_dim(&self) -> usize {
        self.intermediate_size.unwrap_or(self.hidden_size * 4)
    }
}

fn default_vocab_size() -> usize {
    51865
}

fn default_max_target_positions() -> usize {
    448
}

fn default_num_mel_bins() -> usize {
    128
}

fn default_max_audio_len() -> usize {
    3000
}

fn default_audio_projector_type() -> String {
    "linear".to_string()
}
