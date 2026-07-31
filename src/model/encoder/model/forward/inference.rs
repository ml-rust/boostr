//! Inference forward passes over the padded layout.

use crate::error::{Error, Result};
use crate::model::encoder::model::layer::SpanMasks;
use crate::model::encoder::model::pooling::pool_padded;
use crate::model::encoder::model::{Encoder, EncoderClient};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Inference-only forward with pre-computed position IDs and span masks.
    ///
    /// Both are passed in rather than derived here because the CUDA graph path
    /// calls this inside a stream-capture region, where a host-to-device copy
    /// would bake a host pointer into the graph.
    pub fn encode_inference_with_pos<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        pos_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
        span_masks: &SpanMasks<R>,
    ) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let tok_emb = self.token_embed.forward(client, input_ids)?;

        // Gemma: multiply token embeddings by sqrt(hidden_size) after lookup.
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

        // RoPE and ALiBi families encode position inside the attention
        // computation and carry no learned absolute position table.
        let tok_emb = if !self.config.arch_family.uses_learned_positions() {
            tok_emb
        } else {
            let pos_emb = self.position_embed.forward(client, pos_ids)?;
            var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?
        };

        // NomicBert: broadcast-add token_type row 0 (single-segment inference).
        let tok_emb = if let Some(tte) = &self.token_type_embed {
            let hidden_size = self.config.hidden_size;
            let tte_3d = tte.reshape(&[1, 1, hidden_size]).map_err(Error::Numr)?;
            var_add(&tok_emb, &Var::new(tte_3d, false), client).map_err(Error::Numr)?
        } else {
            tok_emb
        };

        let normed = match &self.embed_norm {
            Some(norm) => norm.forward(client, &tok_emb)?,
            None => tok_emb,
        };
        let mut hidden = normed.detach();

        for layer in &self.layers {
            let layer_input = Var::new(hidden.tensor().clone(), false);
            let span = span_masks.for_spec(layer.attn);
            let out = layer.forward(client, &layer_input, attention_mask, span, None)?;
            hidden = out.detach();
        }

        // Gemma/Qwen3: final output_norm over all hidden states before pooling.
        let hidden_tensor = if let Some(on) = &self.output_norm {
            let hidden_var = Var::new(hidden.tensor().clone(), false);
            on.forward(client, &hidden_var)?.tensor().clone()
        } else {
            hidden.tensor().clone()
        };

        // F16 path: cast back to F32 so pooling, the CUDA graph output buffer
        // and any classifier head all receive F32 unchanged.
        if self.config.compute_dtype == DType::F16 && hidden_tensor.dtype() == DType::F16 {
            client.cast(&hidden_tensor, DType::F32).map_err(Error::Numr)
        } else {
            Ok(hidden_tensor)
        }
    }

    /// Inference-only forward pass: token IDs → per-token hidden states
    /// `[B, S, hidden_size]`.
    ///
    /// Detaches between layers so intermediate activations are reclaimed as each
    /// layer completes rather than pinned until the final output is dropped.
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
        let span_masks = SpanMasks::build(&self.config, seq_len, input_ids.device());

        self.encode_inference_with_pos(client, input_ids, &pos_tensor, attention_mask, &span_masks)
    }

    /// Inference-only pooled embedding: token IDs → `[B, hidden_size]`.
    ///
    /// On CUDA, uses a graph capture cache keyed by `(batch_size, seq_len)`.
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
        if let Some(result) = crate::model::encoder::model::cuda_graph::try_graph_embed(
            self,
            client,
            input_ids,
            attention_mask,
        ) {
            return result;
        }
        self.embed_inference_standard(client, input_ids, attention_mask)
    }

    /// Standard (non-graph) pooled embedding forward.
    ///
    /// Always executes the full forward without consulting the CUDA graph cache.
    /// Used directly in tests to compare the captured path against the
    /// authoritative non-graph result.
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
        let hidden = self.encode_inference(client, input_ids, attention_mask)?;
        pool_padded(client, &hidden, attention_mask, self.pooling, None)
    }
}
