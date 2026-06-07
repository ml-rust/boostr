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
use crate::nn::Weight;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::{Tokenize, Tokenizer};

/// Preferred forward compute dtype for the embedding model.
///
/// F16 on CUDA (when the `f16` feature is enabled) so matmul uses WMMA tensor
/// cores; the F32 matmul kernel has no tensor cores and is ~50–100× slower for
/// these shapes. CPU and WGPU keep F32 (no host-side F16 WMMA; WGPU is 32-bit).
fn preferred_compute_dtype<R: Runtime>() -> DType {
    // `R::name()` is the dtype selector; bind it unconditionally so the type
    // parameter is always used (the F16 result is gated behind the feature).
    let is_cuda = R::name() == "cuda";
    #[cfg(feature = "f16")]
    if is_cuda {
        return DType::F16;
    }
    #[cfg(not(feature = "f16"))]
    let _ = is_cuda;
    DType::F32
}

/// Pad-length buckets — all multiples of 16 so numr's WMMA tensor-core GEMM
/// still fires. Bucketing the batch's padded sequence length (rather than
/// rounding each batch to its own next-multiple-of-16) bounds the number of
/// distinct `(batch, seq_len)` shapes seen across a long ingest, which keeps
/// the CUDA allocator pool and graph-capture buffers from growing unbounded.
const SEQ_LEN_BUCKETS: [usize; 6] = [64, 128, 192, 256, 384, 512];

/// Smallest bucket `>= raw` (capped at `max_seq`); for lengths beyond the last
/// bucket, the next multiple of 16.
fn seq_len_bucket(raw: usize, max_seq: usize) -> usize {
    SEQ_LEN_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= raw)
        .unwrap_or_else(|| raw.next_multiple_of(16))
        .min(max_seq)
}

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
        let max_seq = self.encoder.config().max_position_embeddings;
        let mut token_ids = self.tokenizer.encode(text);
        if token_ids.len() > max_seq {
            token_ids.truncate(max_seq);
        }
        let seq_len = token_ids.len();
        let input: Vec<i64> = token_ids.into_iter().map(|t| t as i64).collect();
        let input_tensor = Tensor::<R>::from_slice(&input, &[1, seq_len], &self.device);

        // Single input: no padding, no mask needed.
        let embedding = self.encoder.embed_inference(client, &input_tensor, None)?;
        Ok(embedding.to_vec())
    }

    /// Embed multiple texts → one `[hidden_size]` f32 vector per text.
    ///
    /// For NomicBert, builds a packed (varlen, unpadded) batch and routes to
    /// `embed_inference_varlen`, avoiding all padding waste.
    ///
    /// For all other architectures, pads to a sequence-length bucket and uses
    /// the standard `embed_inference` path with an attention mask.
    pub fn embed_texts<C>(&self, client: &C, texts: &[&str]) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let max_seq = self.encoder.config().max_position_embeddings;
        let all_ids: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| {
                let mut ids = self.tokenizer.encode(t);
                if ids.len() > max_seq {
                    ids.truncate(max_seq);
                }
                ids
            })
            .collect();

        use super::config::ArchFamily;
        // Route to the varlen (packed, unpadded) path when:
        //   - NomicBert (always uses varlen)
        //   - BERT or XLM-RoBERTa AND head_dim ∈ {64, 128}
        //     (varlen CUDA kernel supports only those two head dims; nonstandard
        //      head dims fall through to the padded path which always works).
        let arch = self.encoder.config().arch_family;
        let use_varlen = match arch {
            ArchFamily::NomicBert => true,
            ArchFamily::Bert | ArchFamily::XlmRoberta => {
                let hd = self.encoder.config().head_dim();
                hd == 64 || hd == 128
            }
            // Route Gemma to varlen when head_dim ∈ {64, 128, 256}.
            // The CPU varlen path supports head_dim=256 + GQA.
            // Nonstandard head dims fall through to padded (always correct).
            ArchFamily::GemmaEmbedding => {
                let hd = self.encoder.config().resolved_head_dim();
                hd == 64 || hd == 128 || hd == 256
            }
        };
        if use_varlen {
            return self.embed_texts_varlen(client, &all_ids);
        }

        // Padded path for BERT / XLM-R / Gemma with non-standard head_dim.
        // Pad to one of a small fixed set of sequence-length buckets (all
        // multiples of 16, so numr's WMMA tensor-core GEMM still fires on the
        // attention M/N dims). Bucketing — rather than rounding each batch to
        // its own next-multiple-of-16 — bounds the number of distinct (batch,
        // seq_len) shapes the encoder sees across a long ingest. Without it,
        // variable-length batches produce dozens of distinct shapes, each
        // allocating fresh CUDA-graph + workspace buffers that the pool caches
        // but rarely reuses, growing GPU memory monotonically until OOM
        // (observed on 12-layer models over a full-corpus embed). Padding
        // positions are masked to 0.0, so they cannot contaminate the
        // mean-pooled embedding (normalised by real token count).
        let max_len = {
            let raw = all_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
            if raw == 0 {
                0
            } else {
                seq_len_bucket(raw, max_seq)
            }
        };

        if max_len == 0 {
            return Ok(vec![vec![]; texts.len()]);
        }

        // Pad to max_len (pad token = 0) and build attention mask.
        //
        // The attention mask is a [B, S] float32 tensor: 1.0 for real tokens,
        // 0.0 for padding positions.  Passing it to `encode` prevents padded
        // positions from contaminating the attention scores of real tokens.
        let batch_size = texts.len();
        let mut flat: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut mask_flat: Vec<f32> = Vec::with_capacity(batch_size * max_len);
        for ids in &all_ids {
            let real_len = ids.len();
            flat.extend(ids.iter().map(|&t| t as i64));
            flat.extend(std::iter::repeat_n(0i64, max_len - real_len));
            mask_flat.extend(std::iter::repeat_n(1.0f32, real_len));
            mask_flat.extend(std::iter::repeat_n(0.0f32, max_len - real_len));
        }

        let input_tensor = Tensor::<R>::from_slice(&flat, &[batch_size, max_len], &self.device);
        let mask_tensor = Tensor::<R>::from_slice(&mask_flat, &[batch_size, max_len], &self.device);
        let embeddings = self
            .encoder
            .embed_inference(client, &input_tensor, Some(&mask_tensor))?;

        // Split [B, hidden] → Vec<Vec<f32>>
        let data: Vec<f32> = embeddings.to_vec();
        let hidden = self.encoder.config().hidden_size;
        let result = data.chunks_exact(hidden).map(|c| c.to_vec()).collect();
        Ok(result)
    }

    /// Build a packed (varlen) batch from pre-tokenized id lists and call
    /// `embed_inference_varlen`.  NomicBert only.
    ///
    /// Documents are processed in contiguous sub-batches whose total token
    /// count does not exceed `config.max_tokens_per_forward` (resolved via
    /// [`super::config::DEFAULT_MAX_TOKENS_PER_FORWARD`]).  A single document
    /// that exceeds the budget is always processed alone — documents are never
    /// split.  Output order matches input order exactly.
    fn embed_texts_varlen<C>(&self, client: &C, all_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        use super::config::DEFAULT_MAX_TOKENS_PER_FORWARD;

        let batch = all_ids.len();
        if batch == 0 {
            return Ok(vec![]);
        }

        let budget = self
            .encoder
            .config()
            .max_tokens_per_forward
            .unwrap_or(DEFAULT_MAX_TOKENS_PER_FORWARD);

        // Greedily partition all_ids into contiguous sub-batches, each with
        // total token count ≤ budget.  A single document that exceeds the
        // budget is still placed alone in its own sub-batch.
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(batch);
        let mut start = 0usize;

        while start < batch {
            let mut end = start;
            let mut tokens_in_sub = 0usize;

            // Always include at least one document even if it exceeds the budget.
            while end < batch {
                let doc_len = all_ids[end].len();
                if end == start || tokens_in_sub + doc_len <= budget {
                    tokens_in_sub += doc_len;
                    end += 1;
                } else {
                    break;
                }
            }

            let chunk = &all_ids[start..end];
            let mut sub_result = self.embed_one_varlen_batch(client, chunk)?;
            result.append(&mut sub_result);
            start = end;
        }

        Ok(result)
    }

    /// Pack `ids_chunk` into a single varlen forward pass and return one
    /// `[hidden_size]` embedding per document.
    ///
    /// Builds all host metadata (flat_ids / cu_seqlens / pos_ids / seg_ids)
    /// without any GPU↔CPU tensor-data transfers, uploads them, calls
    /// `embed_inference_varlen`, and splits the `[sub_batch, hidden]` result
    /// into per-document `Vec<f32>`.
    fn embed_one_varlen_batch<C>(&self, client: &C, ids_chunk: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let sub_batch = ids_chunk.len();
        if sub_batch == 0 {
            return Ok(vec![]);
        }

        // Build host metadata — no GPU↔CPU transfers of tensor data.
        let mut flat_ids: Vec<i64> = Vec::new();
        let mut cu: Vec<i32> = Vec::with_capacity(sub_batch + 1);
        let mut pos_ids: Vec<i64> = Vec::new();
        let mut seg_ids: Vec<i32> = Vec::new();
        let mut max_seqlen: usize = 0;

        // XLM-RoBERTa position-id offset: real token at within-sequence index `p`
        // maps to `pad_id + 1 + p`.  For BERT and NomicBert the offset is 0.
        let arch = self.encoder.config().arch_family;
        let xlmr_pad_id: i64 = self.encoder.config().padding_token_id;

        cu.push(0i32);
        for (b, ids) in ids_chunk.iter().enumerate() {
            let n = ids.len();
            if n > max_seqlen {
                max_seqlen = n;
            }
            flat_ids.extend(ids.iter().map(|&t| t as i64));
            for p in 0..n as i64 {
                let pid = match arch {
                    super::config::ArchFamily::XlmRoberta => xlmr_pad_id + 1 + p,
                    _ => p,
                };
                pos_ids.push(pid);
            }
            seg_ids.extend(std::iter::repeat_n(b as i32, n));
            let last = *cu.last().unwrap_or(&0);
            cu.push(last + n as i32);
        }

        if flat_ids.is_empty() {
            return Ok(vec![vec![]; sub_batch]);
        }

        let total_tokens = flat_ids.len();
        let d = &self.device;

        let input_t = Tensor::<R>::from_slice(&flat_ids, &[total_tokens], d);
        let cu_t = Tensor::<R>::from_slice(&cu, &[sub_batch + 1], d);
        let pos_t = Tensor::<R>::from_slice(&pos_ids, &[total_tokens], d);
        let seg_t = Tensor::<R>::from_slice(&seg_ids, &[total_tokens], d);

        // Bypass CUDA graph capture for varlen (graph capture requires fixed shapes).
        let embeddings = self.encoder.embed_inference_varlen(
            client, &input_t, &cu_t, &pos_t, &seg_t, sub_batch, max_seqlen,
        )?;

        // Split [sub_batch, hidden] → Vec<Vec<f32>>
        let data: Vec<f32> = embeddings.to_vec();
        let hidden = self.encoder.config().hidden_size;
        let chunk_result = data.chunks_exact(hidden).map(|c| c.to_vec()).collect();
        Ok(chunk_result)
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

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R, GgufTokenizer>
where
    R::Client: Clone + TypeConversionOps<R> + DequantOps<R>,
{
    /// Load a complete sentence embedding model from a GGUF file.
    ///
    /// Extracts config, weights, and tokenizer from the single file.
    /// Dispatches on `general.architecture`:
    /// - `"nomic-bert"` → `Encoder::from_weights_nomic` with direct GGUF tensor names.
    /// - All others → standard BERT/XLM-RoBERTa path via `hf_name_to_gguf`.
    ///
    /// Compute dtype defaults to F16 on CUDA (when built with the `f16` feature)
    /// so the forward uses WMMA tensor-core matmul — the F32 matmul kernel has no
    /// tensor cores and runs ~50–100× slower for these shapes (profiled: 0.5 vs
    /// ~29 docs/s for nomic-768 on a 3060). CPU/WGPU keep F32. GGUF weights are
    /// loaded dequantized to F32 then cast to the compute dtype by the builders.
    pub fn from_gguf(gguf: &mut Gguf, device: R::Device) -> Result<Self> {
        let tokenizer = GgufTokenizer::from_gguf(gguf)?;
        let mut config = EncoderConfig::from_gguf_metadata(gguf.metadata())?;
        config.compute_dtype = preferred_compute_dtype::<R>();
        let d = &device;

        let encoder = match config.arch_family {
            super::config::ArchFamily::NomicBert => {
                // Obtain a default client to satisfy from_weights_nomic's C bound.
                // compute_dtype remains F32 on this path so no casts are issued.
                let client = R::default_client(d);
                Encoder::from_weights_nomic(config, Pooling::Mean, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            super::config::ArchFamily::GemmaEmbedding => {
                // Obtain a default client to satisfy from_weights_gemma's C bound.
                // compute_dtype remains F32 on this path so no casts are issued.
                let client = R::default_client(d);
                Encoder::from_weights_gemma(config, Pooling::Mean, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            _ => Encoder::from_weights(config, Pooling::Mean, |hf_name| {
                let gguf_name = hf_name_to_gguf(hf_name);
                gguf.load_tensor_f32::<R>(&gguf_name, d)
            })?,
        };

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
    if hf == "embeddings.layer_norm.weight" {
        return "token_embd_norm.weight".into();
    }
    if hf == "embeddings.layer_norm.bias" {
        return "token_embd_norm.bias".into();
    }

    // Encoder layers: encoder.layer.{i}.{rest}
    if let Some(rest) = hf.strip_prefix("encoder.layer.")
        && let Some(dot) = rest.find('.')
    {
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

    hf.to_string()
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
