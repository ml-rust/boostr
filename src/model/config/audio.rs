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
