//! Per-architecture shape of a decoder block that runs on the Llama path.
//!
//! The defaults describe Llama: RMSNorm, a SwiGLU MLP, sequential residuals,
//! full split-half RoPE, and plain logits. A Hugging Face config sets the
//! fields its architecture needs. Tensor presence decides the rest:
//!
//! - a `gate_proj` tensor makes the MLP gated
//! - a norm `bias` tensor gives a LayerNorm its bias
//! - `pre_feedforward_layernorm` tensors select Gemma 2 sandwich norms

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The normalization every norm of the block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    /// RMSNorm: `x * rsqrt(mean(x^2) + eps) * w`.
    #[default]
    Rms,
    /// LayerNorm: `(x - mean) / sqrt(var + eps) * w + b`. A missing bias
    /// tensor is a zero bias (Cohere, DBRX).
    Layer,
}

/// The activation of the MLP, gated or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpActivation {
    /// SiLU, `x * sigmoid(x)`.
    #[default]
    Silu,
    /// Tanh-approximate GELU (`gelu_pytorch_tanh`, `gelu_new`).
    GeluTanh,
    /// Exact erf GELU (`gelu`).
    GeluErf,
}

impl MlpActivation {
    /// The activation a Hugging Face `hidden_act` string names.
    ///
    /// # Errors
    ///
    /// An activation boostr does not run, named in the error.
    pub fn from_hf(name: &str) -> Result<Self> {
        match name {
            "silu" | "swish" => Ok(Self::Silu),
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" | "gelu_tanh" => Ok(Self::GeluTanh),
            "gelu" => Ok(Self::GeluErf),
            other => Err(Error::ModelError {
                reason: format!(
                    "hidden_act '{other}' is not supported; supported: silu, swish, gelu, \
                     gelu_new, gelu_fast, gelu_pytorch_tanh"
                ),
            }),
        }
    }
}

/// How attention and MLP outputs join the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualLayout {
    /// `h = x + attn(n1(x))`, then `h + mlp(n2(h))` (Llama).
    #[default]
    Sequential,
    /// `x + attn(n1(x)) + mlp(n2(x))` with two norms (GPT-NeoX, Falcon-40B).
    Parallel,
    /// `x + attn(n1(x)) + mlp(n1(x))` with one norm (Cohere, Falcon-7B).
    ParallelSharedNorm,
}

/// Architecture details of a block beyond the attention and MoE configs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockConfig {
    /// The norm every norm of the block and the final norm use.
    #[serde(default)]
    pub norm: NormKind,
    /// Added to every norm weight at load. Gemma stores `w - 1`.
    #[serde(default)]
    pub norm_weight_offset: f32,
    /// The MLP activation.
    #[serde(default)]
    pub mlp_activation: MlpActivation,
    /// The residual layout.
    #[serde(default)]
    pub residual: ResidualLayout,
    /// Leading fraction of each head that RoPE rotates (GPT-NeoX
    /// `rotary_pct`, Phi `partial_rotary_factor`).
    #[serde(default = "one")]
    pub partial_rotary_factor: f32,
    /// Rotate adjacent pairs `(2d, 2d+1)` instead of split halves (Cohere).
    #[serde(default)]
    pub rope_interleaved: bool,
    /// Multiplies the token embeddings (Gemma: `sqrt(hidden_size)`).
    #[serde(default)]
    pub embedding_multiplier: Option<f32>,
    /// Softmax scale of the attention scores. `None` is `1/sqrt(head_dim)`.
    #[serde(default)]
    pub attention_scale: Option<f32>,
    /// `cap * tanh(score / cap)` on attention scores (Gemma 2).
    #[serde(default)]
    pub attn_logit_softcap: Option<f32>,
    /// `cap * tanh(logit / cap)` on output logits (Gemma 2).
    #[serde(default)]
    pub final_logit_softcap: Option<f32>,
    /// Multiplies the output logits (Cohere).
    #[serde(default)]
    pub logit_scale: Option<f32>,
    /// Clamps Q, K and V to `[-c, c]` after projection (DBRX).
    #[serde(default)]
    pub clip_qkv: Option<f32>,
    /// Multiplies the ALiBi bias. Falcon scales it by `1/sqrt(head_dim)`.
    #[serde(default)]
    pub alibi_bias_scale: Option<f32>,
    /// Per-layer flag: `true` applies `attention.sliding_window` to that
    /// layer. `None` applies it to every layer.
    #[serde(default)]
    pub sliding_window_layers: Option<Vec<bool>>,
}

fn one() -> f32 {
    1.0
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            norm: NormKind::Rms,
            norm_weight_offset: 0.0,
            mlp_activation: MlpActivation::Silu,
            residual: ResidualLayout::Sequential,
            partial_rotary_factor: 1.0,
            rope_interleaved: false,
            embedding_multiplier: None,
            attention_scale: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            logit_scale: None,
            clip_qkv: None,
            alibi_bias_scale: None,
            sliding_window_layers: None,
        }
    }
}

impl BlockConfig {
    /// Check the fields against the model shape.
    ///
    /// # Errors
    ///
    /// A field out of range, named in the error.
    pub fn validate(&self, head_dim: usize, num_layers: usize) -> Result<()> {
        let f = self.partial_rotary_factor;
        if !(f > 0.0 && f <= 1.0) {
            return Err(Error::ModelError {
                reason: format!("block.partial_rotary_factor must be in (0, 1], got {f}"),
            });
        }
        let rotary = self.rotary_dim(head_dim);
        if rotary == 0 || !rotary.is_multiple_of(2) {
            return Err(Error::ModelError {
                reason: format!(
                    "block.partial_rotary_factor {f} gives {rotary} rotary dims of head_dim \
                     {head_dim}; the count must be even and nonzero"
                ),
            });
        }
        for (name, cap) in [
            ("attn_logit_softcap", self.attn_logit_softcap),
            ("final_logit_softcap", self.final_logit_softcap),
            ("clip_qkv", self.clip_qkv),
        ] {
            if let Some(c) = cap
                && !(c > 0.0 && c.is_finite())
            {
                return Err(Error::ModelError {
                    reason: format!("block.{name} must be positive and finite, got {c}"),
                });
            }
        }
        if let Some(layers) = &self.sliding_window_layers
            && layers.len() != num_layers
        {
            return Err(Error::ModelError {
                reason: format!(
                    "block.sliding_window_layers has {} entries for {num_layers} layers",
                    layers.len()
                ),
            });
        }
        Ok(())
    }

    /// The leading head dimensions RoPE rotates.
    pub fn rotary_dim(&self, head_dim: usize) -> usize {
        if self.partial_rotary_factor >= 1.0 {
            head_dim
        } else {
            (head_dim as f32 * self.partial_rotary_factor) as usize
        }
    }

    /// Whether layer `layer` applies the sliding window.
    pub fn layer_slides(&self, layer: usize) -> bool {
        self.sliding_window_layers
            .as_ref()
            .is_none_or(|layers| layers.get(layer).copied().unwrap_or(true))
    }

    /// Whether every field is the Llama default. Code paths that run only
    /// the Llama block refuse anything else.
    pub fn is_llama(&self) -> bool {
        self == &Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_llama_and_valid() {
        let b = BlockConfig::default();
        assert!(b.is_llama());
        b.validate(64, 2).unwrap();
        assert_eq!(b.rotary_dim(64), 64);
    }

    #[test]
    fn partial_rotary_takes_the_leading_dims() {
        let b = BlockConfig {
            partial_rotary_factor: 0.25,
            ..BlockConfig::default()
        };
        assert_eq!(b.rotary_dim(64), 16);
        assert!(!b.is_llama());
        b.validate(64, 1).unwrap();
    }

    #[test]
    fn odd_rotary_dims_are_refused() {
        let b = BlockConfig {
            partial_rotary_factor: 0.3,
            ..BlockConfig::default()
        };
        assert!(b.validate(10, 1).is_err());
    }

    #[test]
    fn sliding_layers_must_cover_every_layer() {
        let b = BlockConfig {
            sliding_window_layers: Some(vec![true, false]),
            ..BlockConfig::default()
        };
        b.validate(8, 2).unwrap();
        assert!(b.layer_slides(0));
        assert!(!b.layer_slides(1));
        assert!(b.validate(8, 3).is_err());
    }

    #[test]
    fn hf_activation_names_map() {
        assert_eq!(MlpActivation::from_hf("silu").unwrap(), MlpActivation::Silu);
        assert_eq!(
            MlpActivation::from_hf("gelu_pytorch_tanh").unwrap(),
            MlpActivation::GeluTanh
        );
        assert_eq!(
            MlpActivation::from_hf("gelu").unwrap(),
            MlpActivation::GeluErf
        );
        assert!(MlpActivation::from_hf("relu2").is_err());
    }

    #[test]
    fn serde_defaults_fill_every_field() {
        let b: BlockConfig = serde_json::from_str("{}").unwrap();
        assert!(b.is_llama());
    }
}
