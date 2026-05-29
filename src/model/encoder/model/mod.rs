//! BERT-style transformer encoder for embedding generation.
//!
//! Architecture: token_embed + position_embed + embed_norm → N × (SelfAttention + FFN + LayerNorm) → pool
//!
//! Supports two position-id strategies:
//! - BERT: position_ids = [0, 1, ..., S-1]
//! - XLM-RoBERTa: position_ids = cumsum(input_ids != pad_id) + pad_id
//!   (padding positions get position_id = pad_id, real tokens are numbered
//!   from pad_id+1 upward)
//!
//! Used by sentence embedding models (all-MiniLM, BGE, nomic-embed, etc.)
//! and cross-encoder rerankers (bge-reranker-v2-m3, jina-reranker, etc.)
//! for producing fixed-size vector representations from token sequences.

mod build;
#[cfg(feature = "cuda")]
pub(crate) mod graph_cache;
mod layer;

use layer::EncoderLayer;

use super::config::{ArchFamily, EncoderConfig};
use crate::error::{Error, Result};
use crate::nn::{Embedding, LayerNorm};
use crate::quant::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pooling strategy for producing a single vector from encoder outputs.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    /// Average all token hidden states (most common for sentence embeddings).
    Mean,
    /// Use the [CLS] token's hidden state (position 0).
    Cls,
}

/// BERT-style transformer encoder for producing embeddings.
///
/// Takes token IDs and returns either:
/// - `encode()`: per-token hidden states `[B, S, hidden_size]`
/// - `embed()`: pooled embeddings `[B, hidden_size]`
pub struct Encoder<R: Runtime> {
    pub(super) config: EncoderConfig,
    pub(super) token_embed: Embedding<R>,
    pub(super) position_embed: Embedding<R>,
    /// Post-embedding LayerNorm (applied after token + position embedding sum).
    /// Both BERT and XLM-RoBERTa apply this norm before the first transformer layer.
    pub(super) embed_norm: LayerNorm<R>,
    pub(super) layers: Vec<EncoderLayer<R>>,
    pub(super) pooling: Pooling,
    /// CUDA graph capture cache. Compiled only when the `cuda` feature is active.
    /// Non-CUDA runtimes never allocate or touch this field.
    #[cfg(feature = "cuda")]
    pub(super) forward_cache: std::sync::Arc<graph_cache::EncoderForwardCache>,
}

/// Client trait bounds needed by encoder forward passes.
pub trait EncoderClient<R: Runtime>:
    RuntimeClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + BinaryOps<R>
    + ReduceOps<R>
    + ShapeOps<R>
    + IndexingOps<R>
    + ActivationOps<R>
    + UnaryOps<R>
    + NormalizationOps<R>
    + QuantMatmulOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> EncoderClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + ActivationOps<R>
        + UnaryOps<R>
        + NormalizationOps<R>
        + QuantMatmulOps<R>
        + TypeConversionOps<R>,
{
}

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Compute position IDs from input token IDs on the host.
    ///
    /// Returns a flat `Vec<i64>` of length `batch * seq_len`.
    /// Called before graph capture so the D2H read for XLM-RoBERTa runs
    /// outside the captured region.
    pub(crate) fn compute_position_ids_host(
        &self,
        flat_input_ids: &[i64],
        batch: usize,
        seq_len: usize,
    ) -> Vec<i64> {
        match self.config.arch_family {
            ArchFamily::Bert => (0..seq_len as i64).cycle().take(batch * seq_len).collect(),
            ArchFamily::XlmRoberta => {
                let pad_id = self.config.padding_token_id;
                let mut pos_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
                for b in 0..batch {
                    let mut count: i64 = 0;
                    for s in 0..seq_len {
                        let tok = flat_input_ids[b * seq_len + s];
                        if tok == pad_id {
                            pos_flat.push(pad_id);
                        } else {
                            count += 1;
                            pos_flat.push(pad_id + count);
                        }
                    }
                }
                pos_flat
            }
        }
    }

    /// Build the position-ID tensor for a forward pass from `input_ids`.
    ///
    /// BERT uses `[0, 1, ..., S-1]` (shape `[S]`, broadcast across the batch).
    /// XLM-RoBERTa numbers real tokens from `pad_id+1` upward and assigns
    /// `pad_id` to padding positions; the per-row counts depend on the actual
    /// token values, so the result is shaped `[B, S]` (or `[S]` for 1-D input).
    ///
    /// The XLM-RoBERTa branch delegates to [`Self::compute_position_ids_host`]
    /// so the host-side computation lives in exactly one place (the CUDA graph
    /// capture path also calls it directly, before capture begins).
    fn position_ids_tensor(
        &self,
        input_ids: &Tensor<R>,
        shape: &[usize],
        seq_len: usize,
    ) -> Tensor<R> {
        let device = input_ids.device();
        match self.config.arch_family {
            ArchFamily::Bert => {
                let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
                Tensor::<R>::from_slice(&pos_ids, &[seq_len], device)
            }
            ArchFamily::XlmRoberta => {
                let batch = if shape.len() == 2 { shape[0] } else { 1 };
                let flat_ids: Vec<i64> = input_ids.to_vec();
                let pos_flat = self.compute_position_ids_host(&flat_ids, batch, seq_len);
                if shape.len() == 2 {
                    Tensor::<R>::from_slice(&pos_flat, &[batch, seq_len], device)
                } else {
                    Tensor::<R>::from_slice(&pos_flat, &[seq_len], device)
                }
            }
        }
    }

    /// Inference-only forward with pre-computed position IDs.
    ///
    /// Used by the CUDA graph capture path so host-side position ID computation
    /// (which may require a D2H read for XLM-RoBERTa) runs before capture begins.
    /// Non-graph callers go through `encode_inference` which computes IDs internally.
    pub fn encode_inference_with_pos<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        pos_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let tok_emb = self.token_embed.forward(client, input_ids)?;
        let pos_emb = self.position_embed.forward(client, pos_ids)?;

        let combined = var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?;
        let normed = self.embed_norm.forward(client, &combined)?;

        // When compute_dtype == F16, the embedding weights are already F16 so
        // `normed` is F16.  We keep them as-is and the transformer layers run F16.
        // When compute_dtype == F32 (default), everything stays F32 — no cast needed.
        let mut hidden = normed.detach();

        for layer in &self.layers {
            let layer_input = Var::new(hidden.tensor().clone(), false);
            let out = layer.forward(client, &layer_input, attention_mask)?;
            hidden = out.detach();
        }

        // F16 path: cast final hidden states back to F32 so pooling, the CUDA graph
        // output buffer, and the classifier head all receive F32 tensors unchanged.
        let output =
            if self.config.compute_dtype == DType::F16 && hidden.tensor().dtype() == DType::F16 {
                client
                    .cast(hidden.tensor(), DType::F32)
                    .map_err(Error::Numr)?
            } else {
                hidden.tensor().clone()
            };

        Ok(output)
    }

    /// Inference-only forward pass: token IDs → per-token hidden states `[B, S, hidden_size]`.
    ///
    /// Identical to `encode` but breaks the autograd graph between each transformer
    /// layer by calling `.detach()` on the layer output before passing it to the next
    /// layer. This allows the runtime to reclaim intermediate activations (Q, K, V,
    /// attention scores, FFN intermediates) as each layer completes rather than
    /// pinning them all until the final output is dropped.
    ///
    /// Use this for all inference paths. `encode` is preserved for training.
    pub fn encode_inference<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let shape = input_ids.shape().to_vec();
        let seq_len = *shape.last().ok_or_else(|| Error::ModelError {
            reason: "input_ids must have at least 1 dimension".into(),
        })?;

        let pos_tensor = self.position_ids_tensor(input_ids, &shape, seq_len);

        self.encode_inference_with_pos(client, input_ids, &pos_tensor, attention_mask)
    }

    /// Inference-only pooled embedding: token IDs → `[B, hidden_size]`.
    ///
    /// On CUDA, uses a graph capture cache keyed by `(batch_size, seq_len)`.
    /// The first call for a given shape captures the encoder forward into a
    /// `CapturedGraph`; subsequent calls replay it via a single `cuGraphLaunch`,
    /// collapsing ~192 kernel dispatches into one driver call. Non-CUDA runtimes
    /// always run the standard forward pass.
    pub fn embed_inference<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        #[cfg(feature = "cuda")]
        if let Some(result) = cuda_graph::try_graph_embed(self, client, input_ids, attention_mask) {
            return result;
        }
        self.embed_inference_standard(client, input_ids, attention_mask)
    }

    /// Standard (non-graph) pooled embedding forward.
    ///
    /// Always executes the full forward pass through the CPU/GPU kernels without
    /// consulting the CUDA graph cache. Used directly in tests to compare the
    /// graph-captured path against the authoritative non-graph result.
    pub fn embed_inference_standard<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden_var = {
            let t = self.encode_inference(client, input_ids, attention_mask)?;
            Var::new(t, false)
        };

        let pooled = match self.pooling {
            Pooling::Mean => {
                if let Some(mask) = attention_mask {
                    let mask_shape = mask.shape().to_vec();
                    let batch = mask_shape[0];
                    let seq_len = mask_shape[1];
                    let hidden_size = self.config.hidden_size;

                    let mask_3d = mask.reshape(&[batch, seq_len, 1]).map_err(Error::Numr)?;
                    let masked = client
                        .mul(hidden_var.tensor(), &mask_3d)
                        .map_err(Error::Numr)?;
                    let summed = client.sum(&masked, &[1], false).map_err(Error::Numr)?;
                    let token_counts = client.sum(mask, &[1], true).map_err(Error::Numr)?;
                    let token_counts = client
                        .maximum(
                            &token_counts,
                            &Tensor::from_slice(&[1.0f32], &[1], mask.device()),
                        )
                        .map_err(Error::Numr)?;
                    let _ = hidden_size;
                    client.div(&summed, &token_counts).map_err(Error::Numr)?
                } else {
                    client
                        .mean(hidden_var.tensor(), &[1], false)
                        .map_err(Error::Numr)?
                }
            }
            Pooling::Cls => {
                let cls = var_narrow(&hidden_var, 1, 0, 1).map_err(Error::Numr)?;
                let cls = Var::new(cls.tensor().contiguous()?, false);
                let shape = cls.shape().to_vec();
                let batch = shape[0];
                let hidden_dim = shape[2];
                var_reshape(&cls, &[batch, hidden_dim])
                    .map_err(Error::Numr)?
                    .tensor()
                    .clone()
            }
        };

        Ok(pooled)
    }

    /// Forward pass: token IDs → per-token hidden states `[B, S, hidden_size]`.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding.  When `None`, no masking is applied — all positions are
    /// treated as valid, preserving the pre-existing behaviour for callers that
    /// do not have variable-length input.
    pub fn encode<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let shape = input_ids.shape().to_vec();
        let seq_len = *shape.last().ok_or_else(|| Error::ModelError {
            reason: "input_ids must have at least 1 dimension".into(),
        })?;

        let tok_emb = self.token_embed.forward(client, input_ids)?;

        let pos_tensor = self.position_ids_tensor(input_ids, &shape, seq_len);
        let pos_emb = self.position_embed.forward(client, &pos_tensor)?;

        let combined = var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?;
        let mut hidden = self.embed_norm.forward(client, &combined)?;

        for layer in &self.layers {
            hidden = layer.forward(client, &hidden, attention_mask)?;
        }

        Ok(hidden)
    }

    /// Forward pass: token IDs → pooled embedding `[B, hidden_size]`.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding.  Pass `None` for single-sequence inference with no padding.
    pub fn embed<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden = self.encode(client, input_ids, attention_mask)?;

        match self.pooling {
            Pooling::Mean => {
                let pooled = if let Some(mask) = attention_mask {
                    let mask_shape = mask.shape().to_vec();
                    let batch = mask_shape[0];
                    let seq_len = mask_shape[1];
                    let hidden_size = self.config.hidden_size;

                    let mask_3d = mask.reshape(&[batch, seq_len, 1]).map_err(Error::Numr)?;
                    let masked = client.mul(hidden.tensor(), &mask_3d).map_err(Error::Numr)?;
                    let summed = client.sum(&masked, &[1], false).map_err(Error::Numr)?;
                    let token_counts = client.sum(mask, &[1], true).map_err(Error::Numr)?;
                    let token_counts = client
                        .maximum(
                            &token_counts,
                            &Tensor::from_slice(&[1.0f32], &[1], mask.device()),
                        )
                        .map_err(Error::Numr)?;
                    let _ = hidden_size;
                    let pooled_t = client.div(&summed, &token_counts).map_err(Error::Numr)?;
                    Var::new(pooled_t, false)
                } else {
                    Var::new(
                        client
                            .mean(hidden.tensor(), &[1], false)
                            .map_err(Error::Numr)?,
                        false,
                    )
                };
                Ok(pooled)
            }
            Pooling::Cls => {
                let cls = var_narrow(&hidden, 1, 0, 1).map_err(Error::Numr)?;
                let cls = Var::new(cls.tensor().contiguous()?, false);
                let shape = cls.shape().to_vec();
                let batch = shape[0];
                let hidden_dim = shape[2];
                var_reshape(&cls, &[batch, hidden_dim]).map_err(Error::Numr)
            }
        }
    }

    /// Returns the encoder's configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Returns the pooling strategy used by this encoder.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Returns the number of distinct `(batch, seq_len)` shapes that have been
    /// captured into CUDA graphs since this encoder was constructed.
    ///
    /// Only available when the `cuda` feature is active; the count is always 0
    /// for non-CUDA runtimes (the field is not compiled in).
    #[cfg(feature = "cuda")]
    pub fn graph_capture_count(&self) -> usize {
        self.forward_cache.capture_count()
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda_graph;

#[cfg(test)]
mod tests;
