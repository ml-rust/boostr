//! Embedding layer — lookup table for token embeddings

use crate::error::{Error, Result};
use crate::nn::module::Module;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Token embedding: maps integer indices to dense vectors.
///
/// weight: `[vocab_size, embed_dim]`
pub struct Embedding<R: Runtime> {
    weight: Var<R>,
}

impl<R: Runtime> Embedding<R> {
    pub fn new(weight: Tensor<R>, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
        }
    }

    /// Create from a tensor while preserving its stable autograd ID.
    pub fn with_id(weight: Tensor<R>, weight_id: TensorId, trainable: bool) -> Self {
        Self {
            weight: Var::with_id(weight, weight_id, trainable),
        }
    }

    /// Alias for [`with_id`](Self::with_id), matching multi-parameter modules.
    pub fn with_ids(weight: Tensor<R>, weight_id: TensorId, trainable: bool) -> Self {
        Self::with_id(weight, weight_id, trainable)
    }

    /// Forward: lookup rows from embedding table.
    ///
    /// indices: `[...]` integer tensor, output: `[..., embed_dim]`
    ///
    /// Uses `embedding_lookup` which passes all parameters as kernel arguments
    /// (no device-side shape/stride arrays). This is critical for CUDA graph
    /// capture compatibility — the previous `gather`-based approach copied
    /// shape/strides to device via H2D transfers that become stale on graph replay.
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + IndexingOps<R>,
        R::Client: IndexingOps<R>,
    {
        let out = client
            .embedding_lookup(self.weight.tensor(), indices)
            .map_err(Error::Numr)?;
        Ok(Var::new(out, false))
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        vec![(self.weight.id(), &self.weight)]
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }
}

impl<R: Runtime> Module<R> for Embedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        vec![self.weight()]
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        vec![("weight".to_string(), self.weight())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_embedding_basic() {
        let (client, device) = cpu_setup();
        // vocab=3, dim=4
        #[rustfmt::skip]
        let weight = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0,   // token 0
                5.0, 6.0, 7.0, 8.0,       // token 1
                9.0, 10.0, 11.0, 12.0,    // token 2
            ],
            &[3, 4],
            &device,
        );
        let emb = Embedding::new(weight, false);

        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1], &[3], &device);
        let out = emb.forward(&client, &indices).unwrap();
        assert_eq!(out.shape(), &[3, 4]);

        let data: Vec<f32> = out.tensor().to_vec();
        assert_eq!(
            data,
            vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                9.0, 10.0, 11.0, 12.0, // token 2
                5.0, 6.0, 7.0, 8.0, // token 1
            ]
        );
    }

    #[test]
    fn test_embedding_batched() {
        let (client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2], &device);
        let emb = Embedding::new(weight, false);

        // [2, 3] indices
        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, 1, 0, 1], &[2, 3], &device);
        let out = emb.forward(&client, &indices).unwrap();
        assert_eq!(out.shape(), &[2, 3, 2]);
    }
}
