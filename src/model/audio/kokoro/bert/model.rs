use super::embeddings::{AlbertConfig, AlbertEmbeddings};
use super::layer::{AlbertLayer, linear};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Full ALBERT model: embeddings → projection → 12× shared layer → returns
/// `[B, T, hidden_size]`. Pooler output intentionally not exposed — Kokoro
/// doesn't use it downstream.
pub struct AlbertModel<R: Runtime> {
    pub embeddings: AlbertEmbeddings<R>,
    /// Linear(embedding_size, hidden_size). Stored as raw tensors since we
    /// don't need autograd here.
    pub embedding_projection_weight: Tensor<R>,
    pub embedding_projection_bias: Tensor<R>,
    pub shared_layer: AlbertLayer<R>,
    pub config: AlbertConfig,
}

impl<R: Runtime> AlbertModel<R> {
    /// Forward: `token_ids [B, T]` → `[B, T, hidden_size]`.
    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        let emb = self.embeddings.forward(client, token_ids)?; // [B, T, embedding_size]
        let shape = emb.shape();
        let (b, t) = (shape[0], shape[1]);
        let emb_flat = emb
            .reshape(&[b * t, self.config.embedding_size])
            .map_err(Error::Numr)?;
        let projected = linear(
            client,
            &emb_flat,
            &self.embedding_projection_weight,
            &self.embedding_projection_bias,
        )?;
        let mut x = projected
            .reshape(&[b, t, self.config.hidden_size])
            .map_err(Error::Numr)?;

        for _ in 0..self.config.num_hidden_layers {
            x = self.shared_layer.forward(client, &x, &self.config)?;
        }
        Ok(x)
    }
}

/// Kokoro's full text backbone: AlbertModel + `bert_encoder` Linear to the
/// main decoder hidden size.
pub struct BertEncoder<R: Runtime> {
    pub albert: AlbertModel<R>,
    /// `Linear(albert.hidden_size, kokoro_hidden_dim)`.
    pub projection_weight: Tensor<R>,
    pub projection_bias: Tensor<R>,
    pub out_dim: usize,
}

impl<R: Runtime> BertEncoder<R> {
    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        let albert_out = self.albert.forward(client, token_ids)?; // [B, T, H]
        let shape = albert_out.shape();
        let (b, t, h) = (shape[0], shape[1], shape[2]);
        let flat = albert_out.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let projected = linear(
            client,
            &flat,
            &self.projection_weight,
            &self.projection_bias,
        )?;
        projected
            .reshape(&[b, t, self.out_dim])
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::bert::test_support::{
        build_embeddings, build_layer, tiny_config, zeros,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn albert_model_output_shape_is_b_t_hidden() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let model = AlbertModel {
            embeddings: build_embeddings(&cfg, &device),
            embedding_projection_weight: zeros(&[cfg.hidden_size, cfg.embedding_size], &device),
            embedding_projection_bias: zeros(&[cfg.hidden_size], &device),
            shared_layer: build_layer(&cfg, &device),
            config: cfg,
        };
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap();
        let out = model.forward(&client, &ids).unwrap();
        assert_eq!(out.shape(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn bert_encoder_projects_to_out_dim() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let albert = AlbertModel {
            embeddings: build_embeddings(&cfg, &device),
            embedding_projection_weight: zeros(&[cfg.hidden_size, cfg.embedding_size], &device),
            embedding_projection_bias: zeros(&[cfg.hidden_size], &device),
            shared_layer: build_layer(&cfg, &device),
            config: cfg,
        };
        let out_dim = 8;
        let encoder = BertEncoder {
            albert,
            projection_weight: zeros(&[out_dim, cfg.hidden_size], &device),
            projection_bias: zeros(&[out_dim], &device),
            out_dim,
        };
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device).unwrap();
        let out = encoder.forward(&client, &ids).unwrap();
        assert_eq!(out.shape(), &[1, 3, out_dim]);
    }
}
