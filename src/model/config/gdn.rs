//! Gated DeltaNet configuration: the linear-attention layer of `qwen35`.
//!
//! Field names follow the GGUF keys llama.cpp reads in
//! `build_layer_attn_linear` (`src/models/qwen35.cpp`):
//!
//! | Field         | llama.cpp hparam  | Bonsai |
//! | ------------- | ----------------- | ------ |
//! | `state_size`  | `ssm_d_state`     | 128    |
//! | `key_heads`   | `ssm_n_group`     | 16     |
//! | `value_heads` | `ssm_dt_rank`     | 48     |
//! | `inner_size`  | `ssm_d_inner`     | 6144   |
//! | `conv_kernel` | `ssm_conv1d.ne[0]`| 4      |
//! | `rms_eps`     | `f_norm_rms_eps`  | 1e-6   |
//!
//! One head width serves q, k and v: `head_v_dim = inner_size / value_heads`
//! must equal `state_size`, because the recurrence state is `[S, S]`.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Gated DeltaNet layer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdnConfig {
    /// Model width. Input to `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`,
    /// output of `ssm_out`.
    pub hidden_size: usize,
    /// Depthwise causal conv1d kernel width over the `qkv` projection.
    #[serde(default = "default_gdn_conv_kernel")]
    pub conv_kernel: usize,
    /// Head width `S` for q, k and v.
    pub state_size: usize,
    /// Key/query head count `H_k` (`ssm_n_group`).
    pub key_heads: usize,
    /// Value head count `H_v` (`ssm_dt_rank`). Multiple of `key_heads`.
    pub value_heads: usize,
    /// `value_heads * state_size`. Width of `attn_gate` and of `ssm_out`'s input.
    pub inner_size: usize,
    /// Epsilon for the q/k L2 norm and the gated output norm.
    #[serde(default = "default_gdn_rms_eps")]
    pub rms_eps: f32,
    /// Prefill chunk length for `gdn_chunk_prefill`.
    #[serde(default = "default_gdn_chunk_size")]
    pub chunk_size: usize,
    /// `ssm_out` columns are in grouped (training) head order rather than the
    /// tiled order the rest of the layer runs in. Set from the GGUF key
    /// `prism.hadamard.gdn_v_grouped`. See `GdnBlock::group_heads`.
    #[serde(default)]
    pub v_grouped: bool,
}

/// Conv kernel width used when a config omits it.
pub fn default_gdn_conv_kernel() -> usize {
    4
}

/// Norm epsilon used when a config omits it.
pub fn default_gdn_rms_eps() -> f32 {
    1e-6
}

/// Prefill chunk length used when a config omits it. The fork uses 64.
pub fn default_gdn_chunk_size() -> usize {
    64
}

impl GdnConfig {
    /// Check the head geometry.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when a count is zero, `inner_size` is not
    /// `value_heads * state_size`, or `value_heads` is not a multiple of
    /// `key_heads`.
    pub fn validate(&self) -> Result<()> {
        let nonzero = [
            ("hidden_size", self.hidden_size),
            ("conv_kernel", self.conv_kernel),
            ("state_size", self.state_size),
            ("key_heads", self.key_heads),
            ("value_heads", self.value_heads),
            ("chunk_size", self.chunk_size),
        ];
        for (name, value) in nonzero {
            if value == 0 {
                return Err(Error::ModelError {
                    reason: format!("gdn.{name} must be > 0"),
                });
            }
        }
        let want_inner = self.value_heads * self.state_size;
        if self.inner_size != want_inner {
            return Err(Error::ModelError {
                reason: format!(
                    "gdn.inner_size ({}) != value_heads * state_size ({want_inner})",
                    self.inner_size
                ),
            });
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(Error::ModelError {
                reason: format!(
                    "gdn.value_heads ({}) is not a multiple of key_heads ({})",
                    self.value_heads, self.key_heads
                ),
            });
        }
        Ok(())
    }

    /// `key_heads * state_size`: width of the q block and of the k block.
    pub fn key_dim(&self) -> usize {
        self.key_heads * self.state_size
    }

    /// `value_heads * state_size`: width of the v block, `attn_gate` and `ssm_out` input.
    pub fn value_dim(&self) -> usize {
        self.value_heads * self.state_size
    }

    /// `2 * key_dim + value_dim`: width of `attn_qkv` and channel count of the conv.
    pub fn qkv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Value heads per key head.
    pub fn head_repeat(&self) -> usize {
        self.value_heads / self.key_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai() -> GdnConfig {
        GdnConfig {
            hidden_size: 5120,
            conv_kernel: 4,
            state_size: 128,
            key_heads: 16,
            value_heads: 48,
            inner_size: 6144,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: true,
        }
    }

    #[test]
    fn bonsai_geometry() {
        let cfg = bonsai();
        cfg.validate().unwrap();
        assert_eq!(cfg.key_dim(), 2048);
        assert_eq!(cfg.value_dim(), 6144);
        assert_eq!(cfg.qkv_dim(), 2048 + 2048 + 6144);
        assert_eq!(cfg.head_repeat(), 3);
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut cfg = bonsai();
        cfg.inner_size = 6000;
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.value_heads = 50;
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.conv_kernel = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn serde_defaults() {
        let json =
            r#"{"hidden_size":8,"state_size":4,"key_heads":2,"value_heads":4,"inner_size":16}"#;
        let cfg: GdnConfig = serde_json::from_str(json).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.conv_kernel, 4);
        assert_eq!(cfg.chunk_size, 64);
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.v_grouped);
    }
}
