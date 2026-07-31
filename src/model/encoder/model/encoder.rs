//! [`Encoder`] — the model struct and the client trait its forwards require.

use super::layer::EncoderLayer;
use super::pooling::Pooling;
use crate::model::encoder::config::EncoderConfig;
use crate::nn::{Embedding, LayerNorm, RmsNorm};
use crate::ops::{RoPEOps, RoPEPackedOps, VarLenAttentionOps};
use crate::quant::QuantMatmulOps;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Transformer encoder for producing embeddings.
///
/// Takes token IDs and returns either per-token hidden states
/// (`encode_inference`) or pooled embeddings (`embed_inference`).
pub struct Encoder<R: Runtime> {
    pub(in crate::model::encoder) config: EncoderConfig,
    pub(in crate::model::encoder) token_embed: Embedding<R>,
    /// Absolute position embedding table (BERT/XLM-R only).
    /// RoPE architectures keep this as a sentinel zero embedding; never called.
    pub(in crate::model::encoder) position_embed: Embedding<R>,
    /// Post-embedding LayerNorm, applied after the token + position embedding
    /// sum. `None` for architectures that have no such tensor (Gemma, Qwen3).
    ///
    /// This is deliberately an `Option` rather than a unit-weight `LayerNorm`
    /// standing in as an identity: a LayerNorm with weight 1 and bias 0 still
    /// mean-centres and rescales its input, so substituting one changes the
    /// residual stream that every later block reads.
    pub(in crate::model::encoder) embed_norm: Option<LayerNorm<R>>,
    pub(in crate::model::encoder) layers: Vec<EncoderLayer<R>>,
    pub(in crate::model::encoder) pooling: Pooling,
    /// Row 0 of `token_types.weight` as `[1, hidden_size]` (NomicBert only).
    pub(in crate::model::encoder) token_type_embed: Option<Tensor<R>>,
    /// Final RMSNorm applied to all token hidden states before pooling
    /// (Gemma/Qwen3). `None` for BERT/XLM-R/NomicBert.
    pub(in crate::model::encoder) output_norm: Option<RmsNorm<R>>,
    /// CUDA graph capture cache. Compiled only when the `cuda` feature is active.
    #[cfg(feature = "cuda")]
    pub(in crate::model::encoder) forward_cache:
        std::sync::Arc<super::graph_cache::EncoderForwardCache>,
}

/// Client trait bounds needed by encoder forward passes.
pub trait EncoderClient<R: Runtime>:
    RuntimeClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + BinaryOps<R>
    + ReduceOps<R>
    + ShapeOps<R>
    + IndexingOps<R>
    + ActivationOps<R>
    + UnaryOps<R>
    + NormalizationOps<R>
    + QuantMatmulOps<R>
    + TypeConversionOps<R>
    + RoPEOps<R>
    + RoPEPackedOps<R>
    + VarLenAttentionOps<R>
{
}

impl<R, C> EncoderClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + ActivationOps<R>
        + UnaryOps<R>
        + NormalizationOps<R>
        + QuantMatmulOps<R>
        + TypeConversionOps<R>
        + RoPEOps<R>
        + RoPEPackedOps<R>
        + VarLenAttentionOps<R>,
{
}

impl<R: Runtime> Encoder<R> {
    /// Returns the encoder's configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Returns the pooling strategy used by this encoder.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Returns the number of distinct `(batch, seq_len)` shapes captured into
    /// CUDA graphs since this encoder was constructed.
    #[cfg(feature = "cuda")]
    pub fn graph_capture_count(&self) -> usize {
        self.forward_cache.capture_count()
    }
}
