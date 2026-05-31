//! Inference-path varlen (packed) forward methods for `Encoder`.
//!
//! `encode_inference_varlen` / `embed_inference_varlen` run the encoder over a
//! ragged batch concatenated into a single flat buffer (no padding). The caller
//! pre-builds `cu_seqlens`, `position_ids`, and `seg_ids` on the host. For the
//! padded inference path see `encode_inference` / `embed_inference` in `mod.rs`.

use super::layer::VarlenCtx;
use super::{Encoder, EncoderClient, Pooling};
use crate::error::{Error, Result};
use crate::model::encoder::config::ArchFamily;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, ScatterReduceOp, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// NomicBert packed (varlen) forward: flat token IDs `[total_tokens]` →
    /// per-token hidden states `[total_tokens, hidden_size]`.
    ///
    /// All sequences in the batch are concatenated (no padding).  The caller
    /// must pre-build `cu_seqlens`, `position_ids`, and `seg_ids` on the host
    /// and upload them as tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_inference_varlen<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        cu_seqlens: &Tensor<R>,
        position_ids: &Tensor<R>,
        _seg_ids: &Tensor<R>,
        batch: usize,
        max_seqlen: usize,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        // Token embedding lookup: [total_tokens] → [total_tokens, hidden]
        let tok_emb = self.token_embed.forward(client, input_ids)?;

        // Gemma: multiply token embeddings by sqrt(hidden_size) immediately after lookup.
        // Mirrors the same step in encode_inference_with_pos (padded path).
        // BERT/NomicBert: embed_scale is false, this is a no-op.
        let tok_emb = if self.config.embed_scale {
            let scale = (self.config.hidden_size as f64).sqrt();
            Var::new(
                client
                    .mul_scalar(tok_emb.tensor(), scale)
                    .map_err(Error::Numr)?,
                false,
            )
        } else {
            tok_emb
        };

        // Learned absolute position embedding add for BERT / XLM-RoBERTa.
        //
        // NomicBert and GemmaEmbedding use RoPE applied inside each layer;
        // they skip this block.  BERT/XLM-R have no RoPE: their position
        // information comes solely from the learned position embedding table
        // which must be added here, in the same place the padded path adds it.
        //
        // `position_ids` is `[total_tokens]` I64 with values already offset
        // for XLM-RoBERTa (built by `embed_one_varlen_batch` in pipeline.rs).
        // `position_embed.forward` does an embedding lookup → `[total_tokens, hidden]`.
        let tok_emb = match self.config.arch_family {
            ArchFamily::Bert | ArchFamily::XlmRoberta => {
                let pos_emb = self.position_embed.forward(client, position_ids)?;
                var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?
            }
            ArchFamily::NomicBert | ArchFamily::GemmaEmbedding => tok_emb,
        };

        // NomicBert: broadcast-add token-type row 0.
        // BERT/XLM-R: token_type_embed is None, this is a no-op.
        let tok_emb = if let Some(tte) = &self.token_type_embed {
            let hidden_size = self.config.hidden_size;
            // Reshape to [1, hidden] so it broadcasts over [total_tokens, hidden].
            let tte_2d = tte.reshape(&[1, hidden_size]).map_err(Error::Numr)?;
            let tte_var = Var::new(tte_2d, false);
            var_add(&tok_emb, &tte_var, client).map_err(Error::Numr)?
        } else {
            tok_emb
        };

        let normed = self.embed_norm.forward(client, &tok_emb)?;
        let mut hidden = normed.detach();

        let ctx = VarlenCtx {
            cu_seqlens,
            position_ids,
            batch,
            max_seqlen,
        };

        for layer in &self.layers {
            let layer_input = Var::new(hidden.tensor().clone(), false);
            let out = layer.forward(client, &layer_input, None, Some(&ctx))?;
            hidden = out.detach();
        }

        // No output_norm for NomicBert.
        let hidden_tensor = if let Some(on) = &self.output_norm {
            let hidden_var = Var::new(hidden.tensor().clone(), false);
            on.forward(client, &hidden_var)?.tensor().clone()
        } else {
            hidden.tensor().clone()
        };

        let output =
            if self.config.compute_dtype == DType::F16 && hidden_tensor.dtype() == DType::F16 {
                client
                    .cast(&hidden_tensor, DType::F32)
                    .map_err(Error::Numr)?
            } else {
                hidden_tensor
            };

        Ok(output)
    }

    /// NomicBert packed (varlen) pooled embedding: flat token IDs `[total_tokens]`
    /// → `[batch, hidden_size]`.
    ///
    /// Pooling uses `scatter_reduce` with `Mean` reduction over `seg_ids`, which
    /// correctly handles variable-length sequences without any padded position
    /// contaminating the mean.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_inference_varlen<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        cu_seqlens: &Tensor<R>,
        position_ids: &Tensor<R>,
        seg_ids: &Tensor<R>,
        batch: usize,
        max_seqlen: usize,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden_out = self.encode_inference_varlen(
            client,
            input_ids,
            cu_seqlens,
            position_ids,
            seg_ids,
            batch,
            max_seqlen,
        )?;

        let total_tokens = hidden_out.shape()[0];
        let hidden_size = self.config.hidden_size;
        let device = hidden_out.device().clone();

        match self.pooling {
            Pooling::Mean => {
                // seg_ids: [total_tokens] I32  →  index: [total_tokens, hidden_size] I32
                // by reshaping to [total_tokens, 1] and broadcasting to [total_tokens, hidden_size].
                let seg_2d = seg_ids.reshape(&[total_tokens, 1]).map_err(Error::Numr)?;
                let idx = seg_2d
                    .broadcast_to(&[total_tokens, hidden_size])
                    .map_err(Error::Numr)?
                    .contiguous()
                    .map_err(Error::Numr)?;

                // dst: zero-initialised [batch, hidden_size].
                let dst = Tensor::<R>::from_slice(
                    &vec![0.0f32; batch * hidden_size],
                    &[batch, hidden_size],
                    &device,
                );

                let pooled = client
                    .scatter_reduce(&dst, 0, &idx, &hidden_out, ScatterReduceOp::Mean, false)
                    .map_err(Error::Numr)?;
                Ok(pooled)
            }
            Pooling::Cls => {
                // CLS pooling in varlen: CLS token is the first token of each sequence
                // (cu_seqlens[b] for sequence b).  We use gather via cu_seqlens.
                // For simplicity fall back to first token of the flat buffer per batch.
                // NomicBert with CLS pooling is uncommon; this is correct for well-formed
                // packed inputs where CLS is always token 0 of each sequence.
                let cu_q: Vec<i32> = cu_seqlens.to_vec();
                let cls_indices: Vec<i64> = (0..batch).map(|b| cu_q[b] as i64).collect();
                let idx_t = Tensor::<R>::from_slice(&cls_indices, &[batch], &device);
                let cls_out = client
                    .embedding_lookup(&hidden_out, &idx_t)
                    .map_err(Error::Numr)?;
                Ok(cls_out)
            }
        }
    }
}
