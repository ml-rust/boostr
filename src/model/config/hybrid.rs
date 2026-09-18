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
    /// Layer roles from the `qwen35` rule: layer `i` is attention when
    /// `(i + 1) % interval == 0` (`hparams.is_recurrent(il)` in the fork's
    /// `full_attention_interval` handling), else ssm.
    ///
    /// `interval == 0` makes every layer ssm.
    pub fn from_full_attention_interval(num_layers: usize, interval: usize) -> Self {
        let mut ssm_layers = Vec::new();
        let mut attention_layers = Vec::new();
        for i in 0..num_layers {
            if interval != 0 && (i + 1).is_multiple_of(interval) {
                attention_layers.push(i);
            } else {
                ssm_layers.push(i);
            }
        }
        Self {
            ssm_layers,
            attention_layers,
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_attention_interval_marks_every_nth_layer() {
        let cfg = HybridConfig::from_full_attention_interval(8, 4);
        assert_eq!(cfg.attention_layers, vec![3, 7]);
        assert_eq!(cfg.ssm_layers, vec![0, 1, 2, 4, 5, 6]);
        cfg.validate(8).unwrap();
    }

    #[test]
    fn full_attention_interval_zero_is_all_ssm() {
        let cfg = HybridConfig::from_full_attention_interval(3, 0);
        assert!(cfg.attention_layers.is_empty());
        assert_eq!(cfg.ssm_layers, vec![0, 1, 2]);
        cfg.validate(3).unwrap();
    }
}
