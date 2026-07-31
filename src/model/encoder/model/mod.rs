//! Transformer encoder for embedding generation.
//!
//! Supports BERT, XLM-RoBERTa, NomicBert, Gemma-embedding and Qwen3-embedding
//! backbones. Used by sentence embedding models (all-MiniLM, BGE, nomic-embed,
//! EmbeddingGemma, Qwen3-Embedding) and cross-encoder rerankers.

mod build;
mod encoder;
pub(in crate::model::encoder) mod layer;
pub(in crate::model::encoder) mod pooling;

mod forward;
mod inference_varlen;
mod train_forward;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_graph;
#[cfg(feature = "cuda")]
pub(crate) mod graph_cache;

pub use encoder::{Encoder, EncoderClient};
pub use layer::SpanMasks;
pub use pooling::Pooling;

#[cfg(test)]
mod tests;
