//! Embedding layer — lookup table for token embeddings

use crate::error::{Error, Result};
use numr::autograd::{Var, var_gather};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

    /// Forward: lookup rows from embedding table.
    ///
    /// indices: `[...]` integer tensor, output: `[..., embed_dim]`
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + IndexingOps<R>,
        R::Client: IndexingOps<R>,
    {
        // Flatten indices to 1D, gather from dim 0, then reshape
        let idx_shape = indices.shape().to_vec();
        let embed_dim = self.weight.shape()[1];

        // gather along dim=0: each index selects a row
        // We need indices to be [N, 1] for gather dim=0, then reshape output
        let n: usize = idx_shape.iter().product();
        let flat_idx = indices.reshape(&[n]).map_err(Error::Numr)?;

        // Expand indices to [N, embed_dim] for gather
        let expanded = flat_idx.unsqueeze(1).map_err(Error::Numr)?;
        let expanded = expanded
            .broadcast_to(&[n, embed_dim])
            .map_err(Error::Numr)?;

        let gathered = var_gather(&self.weight, 0, &expanded, client).map_err(Error::Numr)?;

        // Reshape to [..., embed_dim]
        let mut out_shape = idx_shape;
        out_shape.push(embed_dim);
        let result = numr::autograd::var_reshape(&gathered, &out_shape).map_err(Error::Numr)?;
        Ok(result)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
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
