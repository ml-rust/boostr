use serde::{Deserialize, Serialize};

/// Vision encoder configuration for multimodal models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Encoder type: "clip", "siglip"
    pub encoder_type: String,
    /// Hidden dimension of the vision encoder
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Patch size for ViT
    pub patch_size: usize,
    /// Input image size (assumes square)
    pub image_size: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Projector type: "linear", "mlp"
    #[serde(default = "default_projector_type")]
    pub projector_type: String,
    /// Number of layers in MLP projector
    #[serde(default = "default_projector_depth")]
    pub projector_depth: usize,
    /// Which encoder layer to extract features from (negative = from end)
    #[serde(default)]
    pub select_layer: Option<i32>,
}

fn default_projector_type() -> String {
    "linear".to_string()
}

fn default_projector_depth() -> usize {
    2
}
