use crate::error::{Error, Result};
use crate::nn::Embedding;
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, NormalizationOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Geometry for one ALBERT instance. Defaults match Kokoro-82M / ALBERT-base.
#[derive(Debug, Clone, Copy)]
pub struct AlbertConfig {
    pub hidden_size: usize,
    pub embedding_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl AlbertConfig {
    /// Per-head width = `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }
}

/// ALBERT input embeddings: word + position + token-type → LayerNorm.
pub struct AlbertEmbeddings<R: Runtime> {
    word_embeddings: Embedding<R>,
    position_embeddings: Embedding<R>,
    token_type_embeddings: Embedding<R>,
    ln_weight: Tensor<R>,
    ln_bias: Tensor<R>,
    eps: f32,
    max_positions: usize,
}

impl<R: Runtime> AlbertEmbeddings<R> {
    pub fn new(
        word_embeddings: Embedding<R>,
        position_embeddings: Embedding<R>,
        token_type_embeddings: Embedding<R>,
        ln_weight: Tensor<R>,
        ln_bias: Tensor<R>,
        eps: f32,
        max_positions: usize,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            ln_weight,
            ln_bias,
            eps,
            max_positions,
        }
    }

    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + NormalizationOps<R>
            + BinaryOps<R>
            + UtilityOps<R>
            + TensorOps<R>,
        R::Client: IndexingOps<R>,
    {
        let shape = token_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: format!("expected [B, T], got {shape:?}"),
            });
        }
        let (b, t) = (shape[0], shape[1]);
        if t > self.max_positions {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: format!(
                    "sequence length {t} exceeds ALBERT's max_position_embeddings {}",
                    self.max_positions
                ),
            });
        }

        let word = self.word_embeddings.forward(client, token_ids)?;
        // Positions: 0, 1, …, T-1 broadcast across batch.
        let positions_1d = client
            .arange(0.0, t as f64, 1.0, DType::I64)
            .map_err(Error::Numr)?;
        let positions = positions_1d
            .reshape(&[1, t])
            .map_err(Error::Numr)?
            .broadcast_to(&[b, t])
            .map_err(Error::Numr)?
            .contiguous()?;
        let pos_emb = self.position_embeddings.forward(client, &positions)?;
        // Token types: all zeros.
        let type_ids = client.fill(&[b, t], 0.0, DType::I64).map_err(Error::Numr)?;
        let type_emb = self.token_type_embeddings.forward(client, &type_ids)?;

        // Sum the three embeddings, then LayerNorm over the last axis.
        let sum1 = client
            .add(word.tensor(), pos_emb.tensor())
            .map_err(Error::Numr)?;
        let summed = client.add(&sum1, type_emb.tensor()).map_err(Error::Numr)?;
        client
            .layer_norm(&summed, &self.ln_weight, &self.ln_bias, self.eps)
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::bert::test_support::{build_embeddings, tiny_config};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn embeddings_reject_oversized_sequence() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let emb = build_embeddings(&cfg, &device);
        // max_positions is 16 in tiny_config; give 20.
        let ids_data = [0i64; 20];
        let ids = Tensor::<CpuRuntime>::from_slice(&ids_data, &[1, 20], &device).unwrap();
        assert!(emb.forward(&client, &ids).is_err());
    }
}
