//! [`ClapTextEncoder`]: the RoBERTa text tower, boostr's [`Encoder`] in
//! its XLM-RoBERTa position numbering, read out at the first token through
//! transformers' `ClapTextPooler` (a dense layer and `tanh`).

use super::config::ClapConfig;
use super::linear::{dense, linear};
use crate::error::{Error, Result};
use crate::model::encoder::config::{ArchFamily, EncoderConfig};
use crate::model::encoder::model::SpanMasks;
use crate::model::encoder::pipeline::batch::pad_rows;
use crate::model::encoder::{Encoder, EncoderClient, Pooling};
use crate::nn::{Linear, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The RoBERTa tower and its pooler.
pub struct ClapTextEncoder<R: Runtime> {
    encoder: Encoder<R>,
    pooler: Linear<R>,
    pad_id: i64,
}

/// The checkpoint name of the encoder weight `name`: the encoder reads its
/// embedding norm as `layer_norm`, the checkpoint stores `LayerNorm`.
fn checkpoint_name(name: &str) -> String {
    match name.strip_prefix("embeddings.layer_norm.") {
        Some(rest) => format!("embeddings.LayerNorm.{rest}"),
        None => name.to_string(),
    }
}

impl<R: Runtime<DType = DType>> ClapTextEncoder<R> {
    /// Reads `embeddings.*`, `encoder.layer.{i}.*` and `pooler.dense.*`
    /// under `vb` (transformers' `text_model`).
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &ClapConfig) -> Result<Self> {
        let t = &config.text;
        let encoder_config = EncoderConfig {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            intermediate_size: t.intermediate_size,
            max_position_embeddings: t.max_position_embeddings,
            layer_norm_eps: t.layer_norm_eps,
            hidden_act: config.text_act,
            type_vocab_size: t.type_vocab_size,
            arch_family: ArchFamily::XlmRoberta,
            padding_token_id: t.pad_token_id,
            ..EncoderConfig::default()
        };
        let encoder = Encoder::from_weights(encoder_config, Pooling::Cls, |name| {
            vb.take_tensor(&checkpoint_name(name))
        })?;
        let mut p = vb.pp("pooler");
        let pooler = linear(&mut p, "dense", true)?;
        drop(p);
        Ok(Self {
            encoder,
            pooler,
            pad_id: t.pad_token_id,
        })
    }

    /// The padding id rows are filled with.
    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }

    /// The pooled text state, `[batch, hidden]`, of `ids`: one row of token
    /// ids per text. Rows pad with [`Self::pad_id`] to the longest, masked.
    ///
    /// Returns [`Error::InvalidArgument`] for no rows or an empty row.
    pub fn forward<C>(&self, client: &C, ids: &[Vec<u32>], device: &R::Device) -> Result<Tensor<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        let batch = ids.len();
        let seq = ids.iter().map(Vec::len).max().unwrap_or(0);
        if batch == 0 || ids.iter().any(Vec::is_empty) {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!("{batch} rows, each of at least one token"),
            });
        }
        let (flat, mask) = pad_rows(ids, seq, self.pad_id);
        let positions = self.encoder.compute_position_ids_host(&flat, batch, seq);
        let input = Tensor::<R>::from_slice(&flat, &[batch, seq], device)?;
        let pos = Tensor::<R>::from_slice(&positions, &[batch, seq], device)?;
        let mask = Tensor::<R>::from_slice(&mask, &[batch, seq], device)?;
        let spans = SpanMasks::build(self.encoder.config(), seq, device)?;
        let hidden =
            self.encoder
                .encode_inference_with_pos(client, &input, &pos, Some(&mask), &spans)?;
        let first = hidden
            .narrow(1, 0, 1)?
            .contiguous()?
            .reshape(&[batch, hidden.shape()[2]])?;
        Ok(client.tanh(&dense(client, &self.pooler, &first)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_embedding_norm_is_renamed() {
        assert_eq!(
            checkpoint_name("embeddings.layer_norm.weight"),
            "embeddings.LayerNorm.weight"
        );
        assert_eq!(
            checkpoint_name("encoder.layer.0.output.LayerNorm.bias"),
            "encoder.layer.0.output.LayerNorm.bias"
        );
    }
}
