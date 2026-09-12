//! Position-id generation, which differs by architecture family.

use crate::model::encoder::config::ArchFamily;
use crate::model::encoder::model::Encoder;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Compute position IDs from input token IDs on the host.
    ///
    /// Returns a flat `Vec<i64>` of length `batch * seq_len`. Called before CUDA
    /// graph capture so the D2H read for XLM-RoBERTa runs outside the captured
    /// region.
    pub(crate) fn compute_position_ids_host(
        &self,
        flat_input_ids: &[i64],
        batch: usize,
        seq_len: usize,
    ) -> Vec<i64> {
        if self.config.arch_family != ArchFamily::XlmRoberta {
            return (0..seq_len as i64).cycle().take(batch * seq_len).collect();
        }

        // XLM-RoBERTa reserves position `pad_id` for padding and numbers real
        // tokens from `pad_id + 1` upward, so the ids depend on token values.
        // `position_row` then re-bases them onto whatever the weight producer
        // left in the table — a converted GGUF has the dead leading rows
        // already chopped, a HuggingFace checkpoint does not.
        let pad_id = self.config.padding_token_id;
        let pad_row = self.config.padding_position_row();
        let mut pos_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for b in 0..batch {
            let mut rank: i64 = 0;
            for s in 0..seq_len {
                let tok = flat_input_ids[b * seq_len + s];
                if tok == pad_id {
                    pos_flat.push(pad_row);
                } else {
                    pos_flat.push(self.config.position_row(rank));
                    rank += 1;
                }
            }
        }
        pos_flat
    }

    /// Build the position-ID tensor for a forward pass from `input_ids`.
    ///
    /// BERT-style families use `[0, 1, ..., S-1]` shaped `[S]` and broadcast
    /// across the batch; XLM-RoBERTa needs per-row ids and is shaped `[B, S]`.
    pub(in crate::model::encoder) fn position_ids_tensor(
        &self,
        input_ids: &Tensor<R>,
        shape: &[usize],
        seq_len: usize,
    ) -> crate::error::Result<Tensor<R>> {
        let device = input_ids.device();
        if self.config.arch_family != ArchFamily::XlmRoberta {
            let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
            return Ok(Tensor::<R>::from_slice(&pos_ids, &[seq_len], device)?);
        }

        let batch = if shape.len() == 2 { shape[0] } else { 1 };
        let flat_ids: Vec<i64> = input_ids.to_vec();
        let pos_flat = self.compute_position_ids_host(&flat_ids, batch, seq_len);
        if shape.len() == 2 {
            Ok(Tensor::<R>::from_slice(
                &pos_flat,
                &[batch, seq_len],
                device,
            )?)
        } else {
            Ok(Tensor::<R>::from_slice(&pos_flat, &[seq_len], device)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::model::encoder::config::{EncoderConfig, FfnVariant};
    use crate::model::encoder::model::Pooling;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_xlm_roberta_position_ids() {
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            ffn_variant: FfnVariant::Standard,
            arch_family: ArchFamily::XlmRoberta,
            padding_token_id: 1,
            ..Default::default()
        };

        let device_ref = &device;
        let encoder =
            Encoder::<CpuRuntime>::from_weights(config, Pooling::Mean, |name| match name {
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

        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 4, 7, 1, 1], &[1, 5], &device).unwrap();
        let result = encoder.embed(&client, &input_ids, None);
        assert!(
            result.is_ok(),
            "xlm-roberta forward should succeed: {result:?}"
        );
        assert_eq!(result.unwrap().shape(), &[1, 8]);
    }
}
