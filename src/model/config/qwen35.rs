//! Full-attention layer configuration for `qwen35`.
//!
//! Field names follow the GGUF keys the PrismML llama.cpp fork reads in
//! `build_layer_attn` (`src/models/qwen35.cpp`):
//!
//! | Field           | Fork hparam                 | Bonsai          |
//! | --------------- | --------------------------- | --------------- |
//! | `hidden_size`   | `n_embd`                    | 5120            |
//! | `num_heads`     | `n_head`                    | 24              |
//! | `num_kv_heads`  | `n_head_kv`                 | 4               |
//! | `head_dim`      | `n_embd_head_k/v`           | 256             |
//! | `rope_dim`      | `n_rot`                     | 64              |
//! | `rope_sections` | `rope.dimension_sections`   | `[11,11,10,0]`  |
//! | `rope_theta`    | `rope.freq_base`            | 1e7             |
//! | `rms_eps`       | `f_norm_rms_eps`            | 1e-6            |
//!
//! `attn_q` projects to `num_heads * 2 * head_dim`: per head, `head_dim`
//! query columns followed by `head_dim` gate columns.

use crate::error::{Error, Result};
use crate::nn::RoPE;
use numr::runtime::Runtime;
use serde::{Deserialize, Serialize};

/// `qwen35` full-attention layer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen35AttentionConfig {
    /// Model width.
    pub hidden_size: usize,
    /// Query heads.
    pub num_heads: usize,
    /// Key/value heads. Divides `num_heads`.
    pub num_kv_heads: usize,
    /// Width of one head, shared by q, k, v and the gate.
    pub head_dim: usize,
    /// Rotated width per head (`n_rot`). Dims past it pass through RoPE.
    pub rope_dim: usize,
    /// IMROPE pair counts per position stream `[t, h, w, e]`.
    pub rope_sections: [usize; 4],
    /// RoPE frequency base.
    pub rope_theta: f32,
    /// Epsilon for `attn_q_norm` / `attn_k_norm`.
    #[serde(default = "default_qwen35_rms_eps")]
    pub rms_eps: f32,
}

/// Norm epsilon used when a config omits it.
pub fn default_qwen35_rms_eps() -> f32 {
    1e-6
}

impl Qwen35AttentionConfig {
    /// Check the head and rope geometry.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when a count is zero, `num_heads` is not a
    /// multiple of `num_kv_heads`, `rope_dim` is odd or above `head_dim`,
    /// `rope_sections` sum to zero, or `rope_theta` is not positive.
    pub fn validate(&self) -> Result<()> {
        let nonzero = [
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
            ("rope_dim", self.rope_dim),
        ];
        for (name, value) in nonzero {
            if value == 0 {
                return Err(Error::ModelError {
                    reason: format!("qwen35_attention.{name} must be > 0"),
                });
            }
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention.num_heads ({}) is not a multiple of num_kv_heads ({})",
                    self.num_heads, self.num_kv_heads
                ),
            });
        }
        if !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention.rope_dim ({}) must be even and at most head_dim ({})",
                    self.rope_dim, self.head_dim
                ),
            });
        }
        if self.rope_sections.iter().sum::<usize>() == 0 {
            return Err(Error::ModelError {
                reason: "qwen35_attention.rope_sections must not all be zero".into(),
            });
        }
        if self.rope_theta.is_nan() || self.rope_theta <= 0.0 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention.rope_theta ({}) must be > 0",
                    self.rope_theta
                ),
            });
        }
        Ok(())
    }

    /// `num_heads * 2 * head_dim`: output width of `attn_q` (query + gate).
    pub fn q_gate_dim(&self) -> usize {
        self.num_heads * 2 * self.head_dim
    }

    /// `num_heads * head_dim`: width of the attention output and of
    /// `attn_output`'s input.
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// `num_kv_heads * head_dim`: output width of `attn_k` and `attn_v`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Query heads per key/value head.
    pub fn head_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// The cos/sin table this layer's IMROPE reads: `[max_positions, rope_dim / 2]`
    /// at `rope_theta`, no scaling.
    pub fn rope_table<R>(&self, max_positions: usize, device: &R::Device) -> Result<RoPE<R>>
    where
        R: Runtime<DType = numr::dtype::DType>,
    {
        RoPE::<R>::precompute_freqs(max_positions, self.rope_dim, self.rope_theta, None, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai() -> Qwen35AttentionConfig {
        Qwen35AttentionConfig {
            hidden_size: 5120,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_sections: [11, 11, 10, 0],
            rope_theta: 1e7,
            rms_eps: 1e-6,
        }
    }

    #[test]
    fn bonsai_geometry() {
        let cfg = bonsai();
        cfg.validate().unwrap();
        assert_eq!(cfg.q_gate_dim(), 12288);
        assert_eq!(cfg.q_dim(), 6144);
        assert_eq!(cfg.kv_dim(), 1024);
        assert_eq!(cfg.head_repeat(), 6);
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut cfg = bonsai();
        cfg.num_kv_heads = 5;
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.rope_dim = 65;
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.rope_dim = 512;
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.rope_sections = [0, 0, 0, 0];
        assert!(cfg.validate().is_err());

        let mut cfg = bonsai();
        cfg.rope_theta = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn serde_defaults() {
        let json = r#"{"hidden_size":8,"num_heads":2,"num_kv_heads":1,"head_dim":8,
            "rope_dim":4,"rope_sections":[1,1,0,0],"rope_theta":10000.0}"#;
        let cfg: Qwen35AttentionConfig = serde_json::from_str(json).unwrap();
        cfg.validate().unwrap();
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn rope_table_has_rope_dim_width() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let table = bonsai().rope_table::<CpuRuntime>(16, &device).unwrap();
        assert_eq!(table.cos_cache().shape(), &[16, 32]);
    }
}
