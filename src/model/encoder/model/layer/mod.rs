//! Single transformer encoder layer: self-attention + FFN.

mod attention_mask;
mod attention_padded;
mod attention_varlen;
mod encoder_layer;
mod ffn;
mod norm;
mod qk_norm;

pub use attention_mask::SpanMasks;
pub(in crate::model::encoder) use attention_mask::ensure_varlen_span_is_unconstrained;
pub(in crate::model::encoder) use encoder_layer::{EncoderLayer, VarlenCtx};
pub(in crate::model::encoder) use norm::NormLayer;
