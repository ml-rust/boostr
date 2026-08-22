//! Mixture of Experts configuration.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// MoE (Mixture of Experts) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts
    pub num_experts: usize,

    /// Number of experts per token (top-k)
    pub experts_per_tok: usize,

    /// Enable shared expert
    #[serde(default)]
    pub shared_expert: Option<bool>,

    /// Expert intermediate size
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Load balance loss coefficient
    #[serde(default = "default_load_balance_alpha")]
    pub load_balance_alpha: f64,

    /// Router z-loss coefficient
    #[serde(default = "default_z_loss_alpha")]
    pub z_loss_alpha: f64,
}

/// Auxiliary load-balancing loss coefficient used when a config omits it.
///
/// Public so consumers that build a `MoeConfig` in code (rather than through
/// serde) can use the same value instead of copying the literal and silently
/// drifting from it.
pub fn default_load_balance_alpha() -> f64 {
    0.01
}

/// Router z-loss coefficient used when a config omits it. Public for the same
/// reason as [`default_load_balance_alpha`].
pub fn default_z_loss_alpha() -> f64 {
    1e-3
}

impl MoeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(Error::ModelError {
                reason: "moe.num_experts must be > 0".into(),
            });
        }
        if self.experts_per_tok == 0 {
            return Err(Error::ModelError {
                reason: "moe.experts_per_tok must be > 0".into(),
            });
        }
        if self.experts_per_tok > self.num_experts {
            return Err(Error::ModelError {
                reason: format!(
                    "moe.experts_per_tok ({}) cannot exceed moe.num_experts ({})",
                    self.experts_per_tok, self.num_experts
                ),
            });
        }
        Ok(())
    }

    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert.unwrap_or(false)
    }
}
