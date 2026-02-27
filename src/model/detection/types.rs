//! Architecture detection types.

/// Model format (oxidizr vs HuggingFace)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFormat {
    /// oxidizr format: `embed_tokens`, `layers.X.`, `norm`, `lm_head`
    #[default]
    Oxidizr,
    /// HuggingFace format: `model.embed_tokens`, `model.layers.X.`, `model.norm`, `lm_head`
    HuggingFace,
}

/// Detected layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Mamba2 state space model layer
    Mamba2,
    /// Mamba3 state space model layer (trapezoidal discretization + complex RoPE + MIMO)
    Mamba3,
    /// Multi-Head Latent Attention + Mixture of Experts
    MlaWithMoe,
    /// Multi-Head Latent Attention + standard MLP
    MlaWithMlp,
    /// Standard transformer (Llama-style GQA + MLP)
    StandardTransformer,
}

/// Auto-detected model configuration
#[derive(Debug, Clone)]
pub struct DetectedConfig {
    /// Number of layers
    pub num_layers: usize,
    /// Layer types for each layer
    pub layer_types: Vec<LayerType>,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model format (oxidizr vs HuggingFace)
    pub format: ModelFormat,
    /// Whether embeddings are tied (no separate lm_head)
    pub tie_word_embeddings: bool,

    // Mamba2 parameters (if detected)
    pub mamba2_num_heads: Option<usize>,
    pub mamba2_head_dim: Option<usize>,
    pub mamba2_state_size: Option<usize>,
    pub mamba2_conv_kernel: Option<usize>,
    pub mamba2_expand: Option<usize>,

    // Mamba3 parameters (if detected)
    pub mamba3_enabled: Option<bool>,
    pub mamba3_complex_rope: Option<bool>,
    pub mamba3_mimo_rank: Option<usize>,
    pub mamba3_use_conv: Option<bool>,

    // MLA parameters (if detected)
    pub num_attention_heads: Option<usize>,
    pub kv_latent_dim: Option<usize>,
    pub q_latent_dim: Option<usize>,
    pub d_rope: Option<usize>,

    // MoE parameters (if detected)
    pub num_experts: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub shared_expert_enabled: bool,

    // Standard transformer parameters
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
}

impl Default for DetectedConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            layer_types: Vec::new(),
            hidden_size: 0,
            vocab_size: 0,
            format: ModelFormat::Oxidizr,
            tie_word_embeddings: false,
            mamba2_num_heads: None,
            mamba2_head_dim: None,
            mamba2_state_size: None,
            mamba2_conv_kernel: None,
            mamba2_expand: None,
            mamba3_enabled: None,
            mamba3_complex_rope: None,
            mamba3_mimo_rank: None,
            mamba3_use_conv: None,
            num_attention_heads: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            num_experts: None,
            intermediate_size: None,
            shared_expert_enabled: false,
            num_kv_heads: None,
            head_dim: None,
        }
    }
}
