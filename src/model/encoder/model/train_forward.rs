//! Training-path forward methods for `Encoder`: `encode` and `embed`.
//!
//! These keep the full autograd graph across layers (used for training).
//! For inference see `encode_inference` / `embed_inference` in `mod.rs`.

use super::layer::SpanMasks;
use super::pooling::pool_padded;
use super::{Encoder, EncoderClient};
use crate::error::{Error, Result};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Forward pass: token IDs → per-token hidden states `[B, S, hidden_size]`.
    ///
    /// Preserves the full autograd graph across transformer layers (training path).
    /// For inference, use `encode_inference` which detaches between layers to free
    /// intermediate activations.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding. When `None`, no masking is applied.
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

        // Gemma: multiply token embeddings by sqrt(hidden_size) immediately after lookup.
        // Not a tensor — pure scalar multiply. Required for Gemma correctness.
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
        // computation; skip the learned absolute position embedding add.
        let tok_emb = if !self.config.arch_family.uses_learned_positions() {
            tok_emb
        } else {
            let pos_tensor = self.position_ids_tensor(input_ids, &shape, seq_len)?;
            let pos_emb = self.position_embed.forward(client, &pos_tensor)?;
            var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?
        };

        // NomicBert token-type row 0 (single-segment inference).
        let tok_emb = if let Some(tte) = &self.token_type_embed {
            let t_shape = tok_emb.shape().to_vec();
            let hidden_size = *t_shape.last().ok_or_else(|| Error::ModelError {
                reason: "tok_emb has no dimensions".into(),
            })?;
            let tte_3d = tte.reshape(&[1, 1, hidden_size]).map_err(Error::Numr)?;
            let tte_var = Var::new(tte_3d, false);
            var_add(&tok_emb, &tte_var, client).map_err(Error::Numr)?
        } else {
            tok_emb
        };

        let mut hidden = match &self.embed_norm {
            Some(norm) => norm.forward(client, &tok_emb)?,
            None => tok_emb,
        };

        let span_masks = SpanMasks::build(&self.config, seq_len, input_ids.device())?;

        for layer in &self.layers {
            // Training path is padded only; varlen is inference-only.
            let span = span_masks.for_spec(layer.attn);
            hidden = layer.forward(client, &hidden, attention_mask, span, None)?;
        }

        // Gemma: apply final output_norm (RMSNorm) to all hidden states before pooling.
        let hidden = if let Some(on) = &self.output_norm {
            on.forward(client, &hidden)?
        } else {
            hidden
        };

        Ok(hidden)
    }

    /// Forward pass: token IDs → pooled embedding `[B, hidden_size]`.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding. Pass `None` for single-sequence inference with no padding.
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
        let pooled = pool_padded(client, hidden.tensor(), attention_mask, self.pooling, None)?;
        Ok(Var::new(pooled, false))
    }
}
