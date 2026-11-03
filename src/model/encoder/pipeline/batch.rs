//! Multi-text embedding: varlen-vs-padded routing and the bucketed padded path.

use super::super::config::{ArchFamily, DEFAULT_MAX_TOKENS_PER_FORWARD};
use super::super::model::EncoderClient;
use super::embed::EmbeddingPipeline;
use crate::error::Result;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::ops::Range;

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
    /// the standard `embed_inference` path with an attention mask. Rows run in
    /// contiguous sub-batches whose padded token count fits
    /// `config.max_tokens_per_forward`, so peak memory does not grow with
    /// `texts.len()`.
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

        let budget = cfg
            .max_tokens_per_forward
            .unwrap_or(DEFAULT_MAX_TOKENS_PER_FORWARD);
        let lens: Vec<usize> = all_ids.iter().map(Vec::len).collect();
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(all_ids.len());
        for range in padded_sub_batches(&lens, max_seq, budget) {
            result.append(&mut self.embed_padded_batch(client, &all_ids[range])?);
        }
        Ok(result)
    }

    /// Embed one padded sub-batch → one `[hidden_size]` vector per row.
    ///
    /// Padded path for BERT / XLM-R / Gemma with non-standard head_dim. Rows
    /// pad to one of a small fixed set of sequence-length buckets (all
    /// multiples of 16, so numr's WMMA tensor-core GEMM still fires on the
    /// attention M/N dims). Bucketing bounds the number of distinct seq_len
    /// shapes across a long ingest; each distinct shape allocates CUDA-graph
    /// and workspace buffers that the pool caches. Padding positions are
    /// masked to 0.0, so they cannot contaminate the mean-pooled embedding
    /// (normalised by real token count).
    fn embed_padded_batch<C>(&self, client: &C, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        let cfg = self.encoder.config();
        let raw = rows.iter().map(Vec::len).max().unwrap_or(0);
        if raw == 0 {
            return Ok(vec![vec![]; rows.len()]);
        }
        let max_len = seq_len_bucket(raw, cfg.max_position_embeddings);

        let batch_size = rows.len();
        let (flat, mask_flat) = pad_rows(rows, max_len, cfg.padding_token_id);

        let input_tensor = Tensor::<R>::from_slice(&flat, &[batch_size, max_len], &self.device)?;
        let mask_tensor =
            Tensor::<R>::from_slice(&mask_flat, &[batch_size, max_len], &self.device)?;
        let embeddings = self
            .encoder
            .embed_inference(client, &input_tensor, Some(&mask_tensor))?;

        // Split [B, hidden] → Vec<Vec<f32>>
        let data: Vec<f32> = embeddings.to_vec();
        let hidden = cfg.hidden_size;
        Ok(data.chunks_exact(hidden).map(|c| c.to_vec()).collect())
    }
}

/// Contiguous row ranges whose padded size fits `budget` tokens.
///
/// A sub-batch's padded size is its row count times the bucket of its longest
/// row. One forward over every row holds the `[B, H, S, S]` attention scores
/// of all rows at once, so peak memory grows with the input count. Each range
/// holds at least one row, even a row whose bucket alone exceeds `budget`.
/// Ranges cover `lens` in order.
fn padded_sub_batches(lens: &[usize], max_seq: usize, budget: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < lens.len() {
        let mut end = start + 1;
        let mut longest = lens[start];
        while end < lens.len() {
            let next_longest = longest.max(lens[end]);
            let padded = (end + 1 - start) * seq_len_bucket(next_longest, max_seq);
            if padded > budget {
                break;
            }
            longest = next_longest;
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Each row's ids right-padded to `max_len` with `pad_id`, flattened, and
/// the matching `[B, S]` attention mask: 1.0 for real tokens, 0.0 for pads.
///
/// The pad must be the model's own pad token. XLM-RoBERTa numbers positions
/// by token value, so a pad written as any other id would take a real
/// position. The mask keeps pads out of the attention scores of real tokens.
pub(crate) fn pad_rows(rows: &[Vec<u32>], max_len: usize, pad_id: i64) -> (Vec<i64>, Vec<f32>) {
    let mut flat: Vec<i64> = Vec::with_capacity(rows.len() * max_len);
    let mut mask: Vec<f32> = Vec::with_capacity(rows.len() * max_len);
    for ids in rows {
        let real_len = ids.len();
        flat.extend(ids.iter().map(|&t| i64::from(t)));
        flat.extend(std::iter::repeat_n(pad_id, max_len - real_len));
        mask.extend(std::iter::repeat_n(1.0f32, real_len));
        mask.extend(std::iter::repeat_n(0.0f32, max_len - real_len));
    }
    (flat, mask)
}

#[cfg(test)]
mod tests {
    use super::super::embed::tests::{make_pipeline_with_distinct_positions, make_test_pipeline};
    use super::{pad_rows, padded_sub_batches};

    /// Pads carry the model's pad id, so XLM-RoBERTa's position ids treat
    /// them as padding.
    #[test]
    fn rows_are_padded_with_the_model_pad_id() {
        let (flat, mask) = pad_rows(&[vec![0, 7, 2], vec![0, 2]], 4, 1);
        assert_eq!(flat, [0, 7, 2, 1, 0, 2, 1, 1]);
        assert_eq!(mask, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    /// Sub-batches stay within the padded-token budget, keep input order and
    /// cover every row; an over-budget row still runs alone.
    #[test]
    fn padded_sub_batches_respect_budget() {
        // Every length buckets to 64: two rows fit 128 tokens, three do not.
        assert_eq!(
            padded_sub_batches(&[10, 20, 30, 40, 50], 512, 128),
            [0..2, 2..4, 4..5]
        );
        // A 300-token row buckets to 384, over the 256 budget: it runs alone.
        assert_eq!(
            padded_sub_batches(&[10, 300, 10], 512, 256),
            [0..1, 1..2, 2..3]
        );
        // Budget large enough for everything: one forward.
        let whole = padded_sub_batches(&[5, 6, 7], 512, 16384);
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0], 0..3);
        assert!(padded_sub_batches(&[], 512, 16384).is_empty());
    }

    /// Splitting a batch across forwards must not change any row's vector.
    #[test]
    fn embed_texts_split_across_forwards_matches_one_forward() {
        let (mut pipeline, client) = make_pipeline_with_distinct_positions();
        let texts = ["hello", "this is a longer input sequence", "world"];
        let whole = pipeline.embed_texts(&client, &texts).unwrap();
        // Every row buckets to the 64-position table, so a 64-token budget
        // runs each row in its own forward.
        pipeline.encoder.config.max_tokens_per_forward = Some(64);
        let split = pipeline.embed_texts(&client, &texts).unwrap();
        assert_eq!(split.len(), texts.len());
        for (i, (a, b)) in whole.iter().zip(&split).enumerate() {
            let max_diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-5, "row {i} differs by {max_diff:.3e}");
        }
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
