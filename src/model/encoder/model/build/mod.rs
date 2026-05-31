//! Constructors for `Encoder`.
//!
//! - `build_bert`: `from_weights` (f32) and `from_weights_quant` (quantized GGUF)
//!   for BERT / XLM-RoBERTa models.
//! - `build_nomic`: `from_weights_nomic` for nomic-bert GGUF models.
//! - `build_gemma`: `from_weights_gemma` for gemma-embedding GGUF models.

mod build_bert;
mod build_gemma;
mod build_nomic;
