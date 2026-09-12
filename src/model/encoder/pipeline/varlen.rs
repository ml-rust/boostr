//! Packed (varlen, unpadded) batch embedding under a per-forward token budget.

use super::super::config::DEFAULT_MAX_TOKENS_PER_FORWARD;
use super::super::model::EncoderClient;
use super::embed::EmbeddingPipeline;
use crate::error::Result;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R> {
    /// Build a packed (varlen) batch from pre-tokenized id lists and call
    /// `embed_inference_varlen`.  NomicBert only.
    ///
    /// Documents are processed in contiguous sub-batches whose total token
    /// count does not exceed `config.max_tokens_per_forward` (resolved via
    /// [`super::super::config::DEFAULT_MAX_TOKENS_PER_FORWARD`]).  A single document
    /// that exceeds the budget is always processed alone — documents are never
    /// split.  Output order matches input order exactly.
    pub(super) fn embed_texts_varlen<C>(
        &self,
        client: &C,
        all_ids: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
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
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
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

        // Position numbering is a config property: XLM-RoBERTa offsets real
        // tokens past its reserved padding slot, then `position_row` re-bases
        // onto the table the weights actually carry. Everything else is 0-based.
        let cfg = self.encoder.config();

        cu.push(0i32);
        for (b, ids) in ids_chunk.iter().enumerate() {
            let n = ids.len();
            if n > max_seqlen {
                max_seqlen = n;
            }
            flat_ids.extend(ids.iter().map(|&t| t as i64));
            for p in 0..n as i64 {
                pos_ids.push(cfg.position_row(p));
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

        let input_t = Tensor::<R>::from_slice(&flat_ids, &[total_tokens], d)?;
        let cu_t = Tensor::<R>::from_slice(&cu, &[sub_batch + 1], d)?;
        let pos_t = Tensor::<R>::from_slice(&pos_ids, &[total_tokens], d)?;
        let seg_t = Tensor::<R>::from_slice(&seg_ids, &[total_tokens], d)?;

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
}
