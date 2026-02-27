//! SSM (State Space Model) configuration.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// SSM (State Space Model) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmConfig {
    /// SSM variant: "mamba2", "mamba3"
    pub variant: String,

    /// Number of SSM heads
    pub num_heads: usize,

    /// Dimension per head
    pub head_dim: usize,

    /// SSM state dimension (N)
    pub state_size: usize,

    /// Chunk size for SSD algorithm
    pub chunk_size: usize,

    /// Number of groups
    #[serde(default = "default_n_groups")]
    pub n_groups: usize,

    /// Conv1D kernel size
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: usize,

    /// Expansion factor
    #[serde(default = "default_expand")]
    pub expand: usize,

    /// Enable complex-valued RoPE for state tracking (Mamba3)
    #[serde(default)]
    pub complex_rope: Option<bool>,

    /// MIMO rank (0 = SISO) (Mamba3)
    #[serde(default)]
    pub mimo_rank: Option<usize>,

    /// Use convolution (Mamba3: default false, uses trapezoidal)
    #[serde(default)]
    pub use_conv: Option<bool>,
}

fn default_n_groups() -> usize {
    1
}

fn default_conv_kernel() -> usize {
    4
}

fn default_expand() -> usize {
    2
}

impl SsmConfig {
    pub fn validate(&self, hidden_size: usize) -> Result<()> {
        if self.num_heads == 0 {
            return Err(Error::ModelError {
                reason: "ssm.num_heads must be > 0".into(),
            });
        }
        if self.head_dim == 0 {
            return Err(Error::ModelError {
                reason: "ssm.head_dim must be > 0".into(),
            });
        }
        if self.state_size == 0 {
            return Err(Error::ModelError {
                reason: "ssm.state_size must be > 0".into(),
            });
        }
        if self.chunk_size == 0 {
            return Err(Error::ModelError {
                reason: "ssm.chunk_size must be > 0".into(),
            });
        }
        // hidden_size * expand == num_heads * head_dim
        let expected = self.num_heads * self.head_dim;
        let actual = hidden_size * self.expand;
        if actual != expected {
            return Err(Error::ModelError {
                reason: format!(
                    "Mamba2 constraint violated: hidden_size * expand ({actual}) != num_heads * head_dim ({expected})"
                ),
            });
        }
        Ok(())
    }

    pub fn is_mamba3(&self) -> bool {
        self.variant == "mamba3"
    }

    pub fn intermediate_dim(&self, hidden_size: usize) -> usize {
        hidden_size * self.expand
    }
}
