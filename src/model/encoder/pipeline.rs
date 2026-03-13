//! High-level text → embedding pipeline.
//!
//! Combines tokenization with encoder forward pass for one-call embedding.
//!
//! ```ignore
//! let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
//! let embeddings = pipeline.embed_text(&client, "Hello world")?; // Vec<f32>
//! let batch = pipeline.embed_texts(&client, &["Hello", "World"])?; // Vec<Vec<f32>>
//! ```

use super::config::EncoderConfig;
use super::model::{Encoder, EncoderClient, Pooling};
use crate::error::Result;
use crate::format::Gguf;
use crate::format::gguf_tokenizer::GgufTokenizer;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::{Tokenize, Tokenizer};

/// End-to-end text → embedding pipeline.
///
/// Owns an `Encoder` and a tokenizer, providing a simple `embed_text()` API
/// that handles tokenization, forward pass, and pooling in one call.
///
/// The default tokenizer type is `splintr::Tokenizer` (BPE) for backward
/// compatibility. Use `EmbeddingPipeline<R, GgufTokenizer>` for GGUF models.
pub struct EmbeddingPipeline<R: Runtime, T: Tokenize = Tokenizer> {
    encoder: Encoder<R>,
    tokenizer: T,
    device: R::Device,
}

impl<R: Runtime<DType = DType>, T: Tokenize> EmbeddingPipeline<R, T> {
    /// Create a new embedding pipeline from an encoder, tokenizer, and device.
    pub fn new(encoder: Encoder<R>, tokenizer: T, device: R::Device) -> Self {
        Self {
            encoder,
            tokenizer,
            device,
        }
    }

    /// Embed a single text string → `[hidden_size]` f32 vector.
    pub fn embed_text<C>(&self, client: &C, text: &str) -> Result<Vec<f32>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let token_ids = self.tokenizer.encode(text);
        let seq_len = token_ids.len();
        let input: Vec<i64> = token_ids.into_iter().map(|t| t as i64).collect();
        let input_tensor = Tensor::<R>::from_slice(&input, &[1, seq_len], &self.device);

        let embedding = self.encoder.embed(client, &input_tensor)?;
        Ok(embedding.tensor().to_vec())
    }

    /// Embed multiple texts → one `[hidden_size]` f32 vector per text.
    ///
    /// Texts are padded to the length of the longest text in the batch and
    /// processed in a single forward pass.
    pub fn embed_texts<C>(&self, client: &C, texts: &[&str]) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all texts
        let all_ids: Vec<Vec<u32>> = texts.iter().map(|t| self.tokenizer.encode(t)).collect();
        let max_len = all_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

        if max_len == 0 {
            return Ok(vec![vec![]; texts.len()]);
        }

        // Pad to max_len (pad token = 0)
        let batch_size = texts.len();
        let mut flat: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        for ids in &all_ids {
            flat.extend(ids.iter().map(|&t| t as i64));
            flat.extend(std::iter::repeat_n(0i64, max_len - ids.len()));
        }

        let input_tensor = Tensor::<R>::from_slice(&flat, &[batch_size, max_len], &self.device);
        let embeddings = self.encoder.embed(client, &input_tensor)?;

        // Split [B, hidden] → Vec<Vec<f32>>
        let data: Vec<f32> = embeddings.tensor().to_vec();
        let hidden = self.encoder.config().hidden_size;
        let result = data.chunks_exact(hidden).map(|c| c.to_vec()).collect();
        Ok(result)
    }

    /// Returns a reference to the underlying encoder model.
    pub fn encoder(&self) -> &Encoder<R> {
        &self.encoder
    }

    /// Returns a reference to the tokenizer.
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Returns the encoder's configuration.
    pub fn config(&self) -> &EncoderConfig {
        self.encoder.config()
    }
}

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R, GgufTokenizer> {
    /// Load a complete sentence embedding model from a GGUF file.
    ///
    /// Extracts config, weights, and tokenizer from the single file.
    /// Handles the standard GGUF tensor naming convention (`blk.N.*`,
    /// `token_embd.*`, etc.) used by llama.cpp for all BERT conversions.
    pub fn from_gguf(gguf: &mut Gguf, device: R::Device) -> Result<Self> {
        let tokenizer = GgufTokenizer::from_gguf(gguf)?;
        let config = EncoderConfig::from_gguf_metadata(gguf.metadata())?;
        let d = &device;
        let encoder = Encoder::from_weights(config, Pooling::Mean, |hf_name| {
            let gguf_name = hf_name_to_gguf(hf_name);
            gguf.load_tensor_f32::<R>(&gguf_name, d)
        })?;
        Ok(Self::new(encoder, tokenizer, device))
    }
}

/// Map HuggingFace BERT weight names to GGUF standard names.
///
/// GGUF (llama.cpp) uses a flat naming scheme for all converted models:
/// - `token_embd.weight` / `position_embd.weight`
/// - `blk.{i}.attn_q.weight` / `.bias`, `attn_k`, `attn_v`, `attn_output`
/// - `blk.{i}.attn_output_norm.weight` / `.bias`
/// - `blk.{i}.ffn_up.weight` / `.bias` (intermediate.dense)
/// - `blk.{i}.ffn_down.weight` / `.bias` (output.dense)
/// - `blk.{i}.layer_output_norm.weight` / `.bias`
fn hf_name_to_gguf(hf: &str) -> String {
    // Embeddings
    if hf == "embeddings.word_embeddings.weight" {
        return "token_embd.weight".into();
    }
    if hf == "embeddings.position_embeddings.weight" {
        return "position_embd.weight".into();
    }

    // Encoder layers: encoder.layer.{i}.{rest}
    if let Some(rest) = hf.strip_prefix("encoder.layer.") {
        if let Some(dot) = rest.find('.') {
            let layer = &rest[..dot];
            let suffix = &rest[dot + 1..];
            let mapped = match suffix {
                "attention.self.query.weight" => "attn_q.weight",
                "attention.self.query.bias" => "attn_q.bias",
                "attention.self.key.weight" => "attn_k.weight",
                "attention.self.key.bias" => "attn_k.bias",
                "attention.self.value.weight" => "attn_v.weight",
                "attention.self.value.bias" => "attn_v.bias",
                "attention.output.dense.weight" => "attn_output.weight",
                "attention.output.dense.bias" => "attn_output.bias",
                "attention.output.LayerNorm.weight" => "attn_output_norm.weight",
                "attention.output.LayerNorm.bias" => "attn_output_norm.bias",
                "intermediate.dense.weight" => "ffn_up.weight",
                "intermediate.dense.bias" => "ffn_up.bias",
                "output.dense.weight" => "ffn_down.weight",
                "output.dense.bias" => "ffn_down.bias",
                "output.LayerNorm.weight" => "layer_output_norm.weight",
                "output.LayerNorm.bias" => "layer_output_norm.bias",
                _ => return hf.to_string(),
            };
            return format!("blk.{layer}.{mapped}");
        }
    }

    hf.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::config::HiddenAct;
    use super::*;
    use crate::error::Error;
    use crate::model::Pooling;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn make_test_pipeline() -> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
        let (client, device) = cpu_setup();

        // Use cl100k_base vocab size for a realistic tokenizer
        let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();
        let vocab_size = tokenizer.vocab_size();

        let config = EncoderConfig {
            vocab_size,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 64,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
        };

        let d = &device;
        let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
            "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
                &vec![0.1f32; vocab_size * 8],
                &[vocab_size, 8],
                d,
            )),
            "embeddings.position_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.01f32; 64 * 8], &[64, 8], d))
            }
            n if n.ends_with("query.weight")
                || n.ends_with("key.weight")
                || n.ends_with("value.weight")
                || n.ends_with("attention.output.dense.weight") =>
            {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], d))
            }
            n if n.ends_with("query.bias")
                || n.ends_with("key.bias")
                || n.ends_with("value.bias")
                || n.ends_with("attention.output.dense.bias")
                || n.ends_with("output.dense.bias") =>
            {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d))
            }
            n if n.ends_with("LayerNorm.weight") => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d)),
            n if n.ends_with("LayerNorm.bias") => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
            n if n.ends_with("intermediate.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], d))
            }
            n if n.ends_with("intermediate.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d))
            }
            n if n.ends_with("output.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], d))
            }
            _ => Err(Error::ModelError {
                reason: format!("unknown weight: {name}"),
            }),
        })
        .unwrap();

        let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
        (pipeline, client)
    }

    #[test]
    fn test_embed_text_returns_hidden_size() {
        let (pipeline, client) = make_test_pipeline();
        let emb = pipeline.embed_text(&client, "hello").unwrap();
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn test_embed_texts_batch() {
        let (pipeline, client) = make_test_pipeline();
        let embs = pipeline.embed_texts(&client, &["hello", "world"]).unwrap();
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].len(), 8);
        assert_eq!(embs[1].len(), 8);
    }

    #[test]
    fn test_embed_texts_empty() {
        let (pipeline, client) = make_test_pipeline();
        let embs = pipeline.embed_texts(&client, &[]).unwrap();
        assert!(embs.is_empty());
    }
}
