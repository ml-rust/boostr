//! ALBERT backbone for Kokoro's phoneme encoder.
//!
//! Kokoro wraps HuggingFace's `AlbertModel` (Lan et al. 2019) with a single
//! `Linear(768 → 512)` projection called `bert_encoder`. Distinctive ALBERT
//! features preserved here:
//!
//! * **Factorized embeddings** — a narrow `embedding_size=128` table projected
//!   up to `hidden_size=768` via a single `embedding_hidden_mapping_in` linear
//!   at the encoder boundary.
//! * **Cross-layer weight sharing** — all `num_hidden_layers=12` transformer
//!   layers are a single `albert_layer_groups[0].albert_layers[0]` block
//!   applied 12 times in a loop. State-dict contains exactly one layer's
//!   worth of parameters under that path.
//!
//! Inference-only (no masking yet — Kokoro runs one utterance at a time with
//! no padding). When we need to serve batched heterogeneous lengths, add
//! `attention_mask` support via an additive `-inf` pre-softmax mask.

mod embeddings;
mod layer;
mod model;
#[cfg(test)]
mod test_support;

pub use embeddings::{AlbertConfig, AlbertEmbeddings};
pub use layer::AlbertLayer;
pub use model::{AlbertModel, BertEncoder};
