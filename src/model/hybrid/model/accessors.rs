//! [`HybridModel`] accessors: the config, the layer split, and the shapes a
//! KV cache and an SSM state take.

use super::build::{HybridBlock, HybridModel};
use crate::model::config::{HybridConfig, UniversalConfig};
use crate::model::mamba::mamba2::Mamba2Config;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// The layer split the model was built with.
    pub fn hybrid_config(&self) -> &HybridConfig {
        &self.hybrid_config
    }

    /// The Mamba2 config of the SSM layers. `None` without SSM layers.
    pub fn mamba_config(&self) -> Option<&Mamba2Config> {
        self.mamba_config.as_ref()
    }

    /// Number of attention layers (for KV cache allocation).
    pub fn num_attention_layers(&self) -> usize {
        self.hybrid_config.attention_layers.len()
    }

    /// Number of SSM layers (for SSM state allocation).
    pub fn num_ssm_layers(&self) -> usize {
        self.hybrid_config.ssm_layers.len()
    }

    /// `(kv_heads, head_dim)` of one attention layer's KV cache: `(1,
    /// kv_lora_rank + rope_head_dim)` for MLA layers, the attention config's
    /// heads and head size for dense ones. `None` without attention layers.
    pub fn kv_cache_dims(&self) -> Option<(usize, usize)> {
        self.kv_dims
    }

    /// Whether the attention layers are MLA.
    pub fn is_mla(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block, HybridBlock::Mla(_)))
    }

    /// RoPE module (for cos/sin cache access in CUDA graph setup). For an
    /// MLA model these are the decoupled RoPE tables.
    pub fn rope(&self) -> &crate::nn::RoPE<R> {
        &self.rope
    }
}
