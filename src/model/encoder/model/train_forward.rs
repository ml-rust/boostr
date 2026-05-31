//! Training-path forward methods for `Encoder`: `encode` and `embed`.
//!
//! These keep the full autograd graph across layers (used for training).
//! For inference see `encode_inference` / `embed_inference` in `mod.rs`.

use super::{Encoder, EncoderClient, Pooling};
use crate::error::{Error, Result};
use crate::model::encoder::config::ArchFamily;
use numr::autograd::{Var, var_add, var_narrow, var_reshape};
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

        // NomicBert and Gemma use RoPE; skip learned position embedding add.
        let tok_emb = if self.config.arch_family == ArchFamily::NomicBert
            || self.config.arch_family == ArchFamily::GemmaEmbedding
        {
            tok_emb
        } else {
            let pos_tensor = self.position_ids_tensor(input_ids, &shape, seq_len);
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

        let mut hidden = self.embed_norm.forward(client, &tok_emb)?;

        for layer in &self.layers {
            // Training path is padded only; varlen is inference-only.
            hidden = layer.forward(client, &hidden, attention_mask, None)?;
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
}
