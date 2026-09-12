//! High-level text → embedding pipeline.
//!
//! Combines tokenization with encoder forward pass for one-call embedding.
//!
//! ```ignore
//! let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
//! let embeddings = pipeline.embed_text(&client, "Hello world")?; // Vec<f32>
//! let batch = pipeline.embed_texts(&client, &["Hello", "World"])?; // Vec<Vec<f32>>
//! ```

mod batch;
mod embed;
mod load;
mod varlen;

pub use embed::EmbeddingPipeline;
