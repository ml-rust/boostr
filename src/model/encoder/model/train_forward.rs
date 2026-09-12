//! Training-path forward methods for `Encoder`: `encode` and `embed`.
//!
//! These keep the full autograd graph across layers (used for training).
//! For inference see `encode_inference` / `embed_inference` in `mod.rs`.

use super::layer::SpanMasks;
use super::pooling::pool_padded;
use super::{Encoder, EncoderClient};
use crate::error::{Error, Result};
use crate::quant::traits::DequantOps;
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
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
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
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        let hidden = self.encode(client, input_ids, attention_mask)?;
        let pooled = pool_padded(client, hidden.tensor(), attention_mask, self.pooling, None)?;
        Ok(Var::new(pooled, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::encoder::config::{EncoderConfig, FfnVariant};
    use crate::model::encoder::model::Pooling;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn make_test_encoder() -> (
        Encoder<CpuRuntime>,
        numr::runtime::cpu::CpuClient,
        numr::runtime::cpu::CpuDevice,
    ) {
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            ffn_variant: FfnVariant::Standard,
            ..Default::default()
        };

        let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
            "embeddings.word_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.1f32; 10 * 8], &[10, 8], &device).unwrap())
            }
            "embeddings.position_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.01f32; 32 * 8], &[32, 8], &device).unwrap())
            }
            "embeddings.layer_norm.weight" => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device).unwrap())
            }
            "embeddings.layer_norm.bias" => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("query.weight")
                || n.ends_with("key.weight")
                || n.ends_with("value.weight") =>
            {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device).unwrap())
            }
            n if n.ends_with("query.bias")
                || n.ends_with("key.bias")
                || n.ends_with("value.bias") =>
            {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("attention.output.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device).unwrap())
            }
            n if n.ends_with("attention.output.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("output.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], &device).unwrap())
            }
            n if n.ends_with("output.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("LayerNorm.weight") => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("LayerNorm.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device).unwrap())
            }
            n if n.ends_with("intermediate.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], &device).unwrap())
            }
            n if n.ends_with("intermediate.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 16], &[16], &device).unwrap())
            }
            _ => Err(Error::ModelError {
                reason: format!("unknown weight: {name}"),
            }),
        })
        .unwrap();

        (encoder, client, device)
    }

    fn make_test_encoder_cls() -> (
        Encoder<CpuRuntime>,
        numr::runtime::cpu::CpuClient,
        numr::runtime::cpu::CpuDevice,
    ) {
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            ffn_variant: FfnVariant::Standard,
            ..Default::default()
        };

        let device_ref = &device;
        let encoder = Encoder::from_weights(config, Pooling::Cls, |name| match name {
            "embeddings.word_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.1f32; 10 * 8], &[10, 8], device_ref).unwrap())
            }
            "embeddings.position_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.01f32; 32 * 8], &[32, 8], device_ref).unwrap())
            }
            "embeddings.layer_norm.weight" => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref).unwrap())
            }
            "embeddings.layer_norm.bias" => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref).unwrap())
            }
            n if n.ends_with("query.weight")
                || n.ends_with("key.weight")
                || n.ends_with("value.weight")
                || n.ends_with("attention.output.dense.weight") =>
            {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], device_ref).unwrap())
            }
            n if n.ends_with("query.bias")
                || n.ends_with("key.bias")
                || n.ends_with("value.bias")
                || n.ends_with("attention.output.dense.bias")
                || n.ends_with("output.dense.bias") =>
            {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref).unwrap())
            }
            n if n.ends_with("LayerNorm.weight") => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref).unwrap())
            }
            n if n.ends_with("LayerNorm.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref).unwrap())
            }
            n if n.ends_with("intermediate.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], device_ref).unwrap())
            }
            n if n.ends_with("intermediate.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 16], &[16], device_ref).unwrap())
            }
            n if n.ends_with("output.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], device_ref).unwrap())
            }
            _ => Err(Error::ModelError {
                reason: format!("unknown weight: {name}"),
            }),
        })
        .unwrap();

        (encoder, client, device)
    }

    #[test]
    fn test_encode_output_shape() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device).unwrap();
        let hidden = encoder.encode(&client, &input_ids, None).unwrap();
        assert_eq!(hidden.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_embed_mean_pool() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device).unwrap();
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[1, 8]);
    }

    #[test]
    fn test_embed_batched() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device).unwrap();
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[2, 8]);
    }

    #[test]
    fn test_encode_with_none_mask_matches_no_mask() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device).unwrap();
        let h1 = encoder.encode(&client, &input_ids, None).unwrap();
        let h2 = encoder.encode(&client, &input_ids, None).unwrap();
        let v1: Vec<f32> = h1.tensor().to_vec();
        let v2: Vec<f32> = h2.tensor().to_vec();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_mask_wrong_shape_returns_error() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device).unwrap();
        let bad_mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device).unwrap();
        let result = encoder.encode(&client, &input_ids, Some(&bad_mask));
        assert!(result.is_err());
    }

    #[test]
    fn test_cls_pooling_batched_produces_correct_shape() {
        let (encoder, client, device) = make_test_encoder_cls();
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device).unwrap();
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[2, 8]);
    }
}
