//! FFN variant and hidden activation.

use serde::{Deserialize, Serialize};

/// FFN variant: controls which feed-forward computation is used per layer.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FfnVariant {
    /// Standard BERT FFN: ffn_down(act(ffn_up(x))).
    #[default]
    Standard,
    /// SwiGLU (NomicBert, Qwen3): ffn_down(silu(ffn_gate(x)) * ffn_up(x)).
    GatedSilu,
    /// Gemma GeGLU: ffn_down(gelu(ffn_gate(x)) * ffn_up(x)).
    /// Gate activation is GELU (not SiLU/SwiGLU).
    GatedGelu,
}

/// Activation used by the `Standard` FFN variant.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    #[default]
    Gelu,
    Relu,
}
