//! [`MlaConfig`]: the shape and hyperparameters an [`super::Mla`] layer is
//! built from.

use crate::error::{Error, Result};
use crate::model::config::{RopeScalingConfig, UniversalConfig};

/// Epsilon of the `q_a_layernorm` and `kv_a_layernorm` latent norms that
/// [`MlaConfig::from_universal`] sets.
pub const MLA_NORM_EPS: f32 = 1e-6;

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
    /// RoPE frequency scaling of the decoupled rotary part. `None` is plain RoPE.
    pub rope_scaling: Option<RopeScalingConfig>,
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
            rope_scaling: None,
            use_norm: true,
            norm_eps: MLA_NORM_EPS,
        }
    }

    /// The MLA layer an attention config with `kv_latent_dim` describes.
    ///
    /// - `head_dim` and `head_dim_v` are `hidden_size / num_heads`.
    /// - `kv_latent_dim`, `q_latent_dim` and `d_rope` give the KV rank, the
    ///   Q rank and the rotary width. Each must be set and non-zero.
    /// - Both latent norms are on, with eps [`MLA_NORM_EPS`].
    /// - `rope_theta` and `rope_scaling` come from the attention config.
    ///
    /// # Errors
    ///
    /// No attention config, `num_heads` of 0, a `hidden_size` that
    /// `num_heads` does not divide, an explicit `head_dim` other than
    /// `hidden_size / num_heads`, or a missing or zero MLA dimension.
    pub fn from_universal(config: &UniversalConfig) -> Result<Self> {
        let attn = config
            .attention
            .as_ref()
            .ok_or_else(|| model_error("attention config missing while building MLA config"))?;
        if attn.num_heads == 0 {
            return Err(model_error("MLA num_heads must be non-zero"));
        }
        if !config.hidden_size.is_multiple_of(attn.num_heads) {
            return Err(model_error(format!(
                "MLA hidden_size ({}) must be divisible by num_heads ({})",
                config.hidden_size, attn.num_heads
            )));
        }
        let head_dim = config.hidden_size / attn.num_heads;
        if let Some(explicit) = attn.head_dim
            && explicit != head_dim
        {
            return Err(model_error(format!(
                "MLA head_dim is hidden_size / num_heads ({head_dim}), but attention.head_dim \
                 is {explicit}; remove attention.head_dim"
            )));
        }
        let config = Self {
            hidden_size: config.hidden_size,
            num_heads: attn.num_heads,
            head_dim,
            head_dim_v: head_dim,
            kv_lora_rank: required_dim("kv_latent_dim", attn.kv_latent_dim)?,
            q_lora_rank: required_dim("q_latent_dim", attn.q_latent_dim)?,
            rope_head_dim: required_dim("d_rope", attn.d_rope)?,
            max_seq_len: config.max_seq_len,
            rope_theta: attn.rope_theta,
            rope_scaling: attn.rope_scaling.clone(),
            use_norm: true,
            norm_eps: MLA_NORM_EPS,
        };
        config.validate()?;
        Ok(config)
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

    /// Width of one token's entry in the absorbed KV cache: the latent and
    /// the rotated rotary part, `kv_lora_rank + rope_head_dim`.
    pub fn latent_cache_dim(&self) -> usize {
        self.kv_lora_rank + self.rope_head_dim
    }
}

fn model_error(reason: impl Into<String>) -> Error {
    Error::ModelError {
        reason: reason.into(),
    }
}

fn required_dim(name: &str, value: Option<usize>) -> Result<usize> {
    match value {
        None => Err(model_error(format!("MLA requires {name}"))),
        Some(0) => Err(model_error(format!("MLA {name} must be non-zero"))),
        Some(dim) => Ok(dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn universal(extra_attention: &str) -> UniversalConfig {
        let yaml = format!(
            "model_type: llama\nvocab_size: 16\nhidden_size: 16\nnum_layers: 1\n\
             max_seq_len: 8\nattention:\n  num_heads: 2\n  kv_latent_dim: 8\n  \
             q_latent_dim: 6\n  d_rope: 4\n{extra_attention}"
        );
        serde_saphyr::from_str(&yaml).unwrap()
    }

    #[test]
    fn from_universal_reads_the_mla_dims() {
        let config = MlaConfig::from_universal(&universal("")).unwrap();
        assert_eq!(config.head_dim, 8);
        assert_eq!(config.head_dim_v, 8);
        assert_eq!(config.kv_lora_rank, 8);
        assert_eq!(config.q_lora_rank, 6);
        assert_eq!(config.rope_head_dim, 4);
        assert_eq!(config.latent_cache_dim(), 12);
        assert_eq!(config.norm_eps, MLA_NORM_EPS);
        assert!(config.use_norm && config.rope_scaling.is_none());
    }

    #[test]
    fn from_universal_carries_rope_scaling() {
        let config = MlaConfig::from_universal(&universal(
            "  rope_scaling:\n    type: linear\n    factor: 2.0\n",
        ))
        .unwrap();
        let scaling = config.rope_scaling.expect("rope_scaling");
        assert_eq!(scaling.scaling_type, "linear");
        assert_eq!(scaling.factor, 2.0);
    }

    #[test]
    fn from_universal_refuses_a_missing_or_zero_dim_and_a_foreign_head_dim() {
        let mut config = universal("");
        config.attention.as_mut().unwrap().d_rope = None;
        let err = MlaConfig::from_universal(&config).unwrap_err().to_string();
        assert!(err.contains("MLA requires d_rope"), "{err}");

        config.attention.as_mut().unwrap().d_rope = Some(0);
        let err = MlaConfig::from_universal(&config).unwrap_err().to_string();
        assert!(err.contains("d_rope must be non-zero"), "{err}");

        let err = MlaConfig::from_universal(&universal("  head_dim: 4\n"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("attention.head_dim is 4"), "{err}");
    }
}
