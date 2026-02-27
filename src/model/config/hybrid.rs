//! Hybrid layer configuration for mixed SSM/attention architectures.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Hybrid layer configuration
///
/// Specifies which layers use SSM vs attention in hybrid architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Layer indices that use SSM (Mamba)
    pub ssm_layers: Vec<usize>,

    /// Layer indices that use attention (transformer)
    pub attention_layers: Vec<usize>,
}

impl HybridConfig {
    pub fn validate(&self, num_layers: usize) -> Result<()> {
        let mut assigned = vec![false; num_layers];

        for &layer in &self.ssm_layers {
            if layer >= num_layers {
                return Err(Error::ModelError {
                    reason: format!(
                        "ssm_layers contains invalid layer index {layer} (num_layers = {num_layers})"
                    ),
                });
            }
            if assigned[layer] {
                return Err(Error::ModelError {
                    reason: format!(
                        "Layer {layer} assigned to both ssm_layers and attention_layers"
                    ),
                });
            }
            assigned[layer] = true;
        }

        for &layer in &self.attention_layers {
            if layer >= num_layers {
                return Err(Error::ModelError {
                    reason: format!(
                        "attention_layers contains invalid layer index {layer} (num_layers = {num_layers})"
                    ),
                });
            }
            if assigned[layer] {
                return Err(Error::ModelError {
                    reason: format!(
                        "Layer {layer} assigned to both ssm_layers and attention_layers"
                    ),
                });
            }
            assigned[layer] = true;
        }

        for (i, &is_assigned) in assigned.iter().enumerate() {
            if !is_assigned {
                return Err(Error::ModelError {
                    reason: format!(
                        "Layer {i} is not assigned to either ssm_layers or attention_layers"
                    ),
                });
            }
        }

        Ok(())
    }

    pub fn is_ssm_layer(&self, layer_idx: usize) -> bool {
        self.ssm_layers.contains(&layer_idx)
    }

    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        self.attention_layers.contains(&layer_idx)
    }
}
