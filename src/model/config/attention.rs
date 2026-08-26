//! Attention configuration types.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,

    #[serde(default)]
    pub num_kv_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// KV latent dimension for MLA compression
    #[serde(default)]
    pub kv_latent_dim: Option<usize>,

    /// Query latent dimension for MLA compression
    #[serde(default)]
    pub q_latent_dim: Option<usize>,

    /// Decoupled RoPE dimension for MLA
    #[serde(default)]
    pub d_rope: Option<usize>,

    /// Sliding window size (for Mistral-style attention)
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Use ALiBi position embeddings instead of RoPE (Falcon v1, BLOOM, MPT)
    #[serde(default)]
    pub use_alibi: bool,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl AttentionConfig {
    pub fn validate(&self, hidden_size: usize) -> Result<()> {
        if self.num_heads == 0 {
            return Err(Error::ModelError {
                reason: "num_heads must be > 0".into(),
            });
        }
        // Divisibility only matters when `head_dim` is inferred. `head_dim()`
        // falls back to `hidden_size / num_heads`, so a non-divisible pair
        // would silently truncate — but an EXPLICIT `head_dim` never performs
        // that division, and rejecting it locks out legitimate configs whose
        // head_dim is not hidden_size/num_heads (Qwen3 and Llama-3.2 both ship
        // such shapes).
        match self.head_dim {
            Some(0) => {
                return Err(Error::ModelError {
                    reason: "head_dim must be > 0 when set explicitly".into(),
                });
            }
            Some(_) => {}
            None => {
                if !hidden_size.is_multiple_of(self.num_heads) {
                    return Err(Error::ModelError {
                        reason: format!(
                            "hidden_size ({hidden_size}) must be divisible by num_heads ({}), \
                             or head_dim must be set explicitly",
                            self.num_heads
                        ),
                    });
                }
            }
        }
        if let Some(kv) = self.num_kv_heads
            && !self.num_heads.is_multiple_of(kv)
        {
            return Err(Error::ModelError {
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({kv})",
                    self.num_heads
                ),
            });
        }
        Ok(())
    }

    pub fn head_dim(&self, hidden_size: usize) -> usize {
        self.head_dim.unwrap_or(hidden_size / self.num_heads)
    }

    pub fn kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// Sliding-window span, or `0` (unlimited) if unset. `Some(0)` is treated
    /// as disabled too: a zero-width window is undefined, and `0` is the
    /// flash-attention kernel's "unlimited" sentinel.
    pub fn sliding_window(&self) -> usize {
        self.sliding_window.unwrap_or(0)
    }

    /// Check if this is MLA (Multi-Head Latent Attention)
    pub fn is_mla(&self) -> bool {
        self.kv_latent_dim.is_some()
    }

    /// Check if this is GQA (Grouped Query Attention)
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads.is_some() && self.num_kv_heads != Some(self.num_heads)
    }
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub factor: f32,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_heads: usize, head_dim: Option<usize>) -> AttentionConfig {
        AttentionConfig {
            num_heads,
            num_kv_heads: None,
            head_dim,
            rope_theta: default_rope_theta(),
            rope_scaling: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            sliding_window: None,
            use_alibi: false,
        }
    }

    /// An explicit `head_dim` means `head_dim()` never divides, so a
    /// non-divisible `hidden_size` is legitimate. Rejecting it locked out real
    /// checkpoints whose head_dim is not `hidden_size / num_heads`.
    #[test]
    fn explicit_head_dim_allows_non_divisible_hidden_size() {
        let c = cfg(3, Some(4));
        assert!(c.validate(10).is_ok(), "explicit head_dim must be accepted");
        // And the accessor returns the explicit value, never the division.
        assert_eq!(c.head_dim(10), 4);
    }

    /// Without an explicit `head_dim` the division IS used, so a non-divisible
    /// pair would silently truncate (10 / 3 == 3, losing a dimension). That
    /// must still be refused, and the message must point at the way out.
    #[test]
    fn inferred_head_dim_still_requires_divisibility() {
        let c = cfg(3, None);
        let Err(err) = c.validate(10) else {
            panic!("a non-divisible hidden_size with inferred head_dim must be refused");
        };
        let msg = err.to_string();
        assert!(msg.contains("10"), "{msg}");
        assert!(msg.contains("head_dim must be set explicitly"), "{msg}");

        // The divisible case is unaffected.
        assert!(cfg(4, None).validate(12).is_ok());
        assert_eq!(cfg(4, None).head_dim(12), 3);
    }

    /// `head_dim: Some(0)` would make every head zero-width. The old code let
    /// it through whenever `hidden_size` happened to divide evenly.
    #[test]
    fn explicit_zero_head_dim_is_refused() {
        let Err(err) = cfg(4, Some(0)).validate(12) else {
            panic!("head_dim of 0 must be refused");
        };
        assert!(err.to_string().contains("head_dim must be > 0"), "{err}");
    }

    /// The num_heads/num_kv_heads rule is independent of head_dim and must
    /// still fire on either path.
    #[test]
    fn kv_head_divisibility_is_unchanged() {
        let mut c = cfg(4, Some(8));
        c.num_kv_heads = Some(3);
        assert!(c.validate(10).is_err(), "4 heads is not divisible by 3 kv");
        c.num_kv_heads = Some(2);
        assert!(c.validate(10).is_ok());
    }
}
