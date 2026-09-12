//! Multi-text embedding: varlen-vs-padded routing and the bucketed padded path.

use super::super::config::ArchFamily;
use super::super::model::EncoderClient;
use super::embed::EmbeddingPipeline;
use crate::error::Result;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R> {
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
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let max_seq = self.encoder.config().max_position_embeddings;
        let all_ids: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| self.wrap_special_tokens(self.tokenizer.encode_raw(t), max_seq))
            .collect();

        // Route to the varlen (packed, unpadded) path when:
        //   - NomicBert (always uses varlen)
        //   - BERT or XLM-RoBERTa AND head_dim ∈ {64, 128}
        //     (varlen CUDA kernel supports only those two head dims; nonstandard
        //      head dims fall through to the padded path which always works).
        let cfg = self.encoder.config();
        let arch = cfg.arch_family;
        // The varlen kernel cannot apply a bounded attention span, so a model
        // whose sliding window would actually bind at this batch's longest
        // sequence must take the padded path — which does mask — rather than
        // silently returning unwindowed results.
        let longest = all_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let span_allows_varlen = cfg.varlen_span_is_unconstrained(longest);

        let use_varlen = match arch {
            ArchFamily::NomicBert => true,
            ArchFamily::Bert | ArchFamily::XlmRoberta => {
                let hd = cfg.head_dim();
                hd == 64 || hd == 128
            }
            // Route Gemma to varlen when head_dim ∈ {64, 128, 256}.
            // The CPU varlen path supports head_dim=256 + GQA.
            // Nonstandard head dims fall through to padded (always correct).
            ArchFamily::GemmaEmbedding => {
                let hd = cfg.resolved_head_dim();
                span_allows_varlen && (hd == 64 || hd == 128 || hd == 256)
            }
            // Causal: the varlen kernel is invoked non-causally here, so the
            // padded path is the only correct one.
            ArchFamily::Qwen3 => false,
            // jina-bert-v2 encodes position as an ALiBi score bias the varlen
            // kernel has no slot for; jina-bert-v3's biased fused QKV is built
            // for the padded path. Both are correct there.
            ArchFamily::JinaBertV2 | ArchFamily::JinaBertV3 => false,
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

        let input_tensor = Tensor::<R>::from_slice(&flat, &[batch_size, max_len], &self.device)?;
        let mask_tensor =
            Tensor::<R>::from_slice(&mask_flat, &[batch_size, max_len], &self.device)?;
        let embeddings = self
            .encoder
            .embed_inference(client, &input_tensor, Some(&mask_tensor))?;

        // Split [B, hidden] → Vec<Vec<f32>>
        let data: Vec<f32> = embeddings.to_vec();
        let hidden = self.encoder.config().hidden_size;
        let result = data.chunks_exact(hidden).map(|c| c.to_vec()).collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::embed::tests::{make_pipeline_with_distinct_positions, make_test_pipeline};

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

    /// Core correctness test: embedding a short sequence alone (no padding)
    /// must produce the same vector as embedding it in a batch alongside a
    /// longer sequence (where it is padded on the right).
    ///
    /// Without an attention mask the pad tokens contribute to the mean-pool
    /// output, causing V1 != V1'.  With the mask they are excluded and
    /// V1 == V1' (within float epsilon).
    #[test]
    fn embed_texts_with_padding_excludes_pad_contamination() {
        let (pipeline, client) = make_pipeline_with_distinct_positions();

        // Embed "hello" alone — no padding, no mask needed.
        let solo = pipeline.embed_texts(&client, &["hello"]).unwrap();
        let v1 = &solo[0];

        // Embed "hello" together with a longer text — "hello" gets padded.
        let batch = pipeline
            .embed_texts(&client, &["hello", "this is a longer input sequence"])
            .unwrap();
        let v1_prime = &batch[0];

        assert_eq!(v1.len(), v1_prime.len());

        // Both should agree to within a small epsilon; if masking is broken
        // they will differ by the contribution of pad-token hidden states.
        let max_diff = v1
            .iter()
            .zip(v1_prime.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "pad contamination detected: max element-wise diff = {max_diff:.3e}"
        );
    }
}
