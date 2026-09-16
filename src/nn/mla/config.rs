//! [`MlaConfig`]: the shape and hyperparameters an [`super::Mla`] layer is
//! built from.

use crate::error::{Error, Result};

/// MLA configuration
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head for Q/K nope portion
    pub head_dim: usize,
    /// Dimension per head for values (can differ from head_dim)
    pub head_dim_v: usize,
    /// KV compression latent dimension
    pub kv_lora_rank: usize,
    /// Q compression latent dimension (0 = no compression)
    pub q_lora_rank: usize,
    /// Decoupled RoPE dimension
    pub rope_head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base theta
    pub rope_theta: f32,
    /// Whether to use RMSNorm on compressed representations
    pub use_norm: bool,
    /// RMSNorm epsilon
    pub norm_eps: f32,
}

impl MlaConfig {
    /// Create config with DeepSeek-V2 defaults
    pub fn deepseek_v2(
        hidden_size: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
        rope_head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            hidden_size,
            num_heads,
            head_dim,
            head_dim_v: head_dim,
            kv_lora_rank,
            q_lora_rank,
            rope_head_dim,
            max_seq_len,
            rope_theta: 10000.0,
            use_norm: true,
            norm_eps: 1e-6,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(Error::ModelError {
                reason: "hidden_size and num_heads must be > 0".into(),
            });
        }
        if self.kv_lora_rank == 0 {
            return Err(Error::ModelError {
                reason: "kv_lora_rank must be > 0 for MLA".into(),
            });
        }
        if self.rope_head_dim > self.head_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_head_dim ({}) > head_dim ({})",
                    self.rope_head_dim, self.head_dim
                ),
            });
        }
        Ok(())
    }

    /// Total Q/K dimension per head (nope + pe)
    pub fn qk_head_dim(&self) -> usize {
        self.head_dim + self.rope_head_dim
    }

    /// Whether Q uses low-rank compression
    pub fn q_uses_lora(&self) -> bool {
        self.q_lora_rank > 0
    }
}
