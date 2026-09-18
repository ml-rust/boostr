//! An [`Embedding`] whose stored table needs an inverse Hadamard rotation
//! after lookup — the PrismML llama.cpp fork's `inverse_weight_names`
//! contract (see `crate::format::gguf::prism_hadamard`), e.g. a GGUF
//! `token_embd.weight` stored in the rotated basis.
//!
//! Kept as its own wrapper rather than a field on [`Embedding`] itself:
//! `Embedding::forward`'s bounds are shared by every caller in the crate
//! (`src/model/**`), and widening them with `FwhtOps`/`BinaryOps` for an
//! optional rotation forces every existing call site to prove those bounds
//! too.

use crate::error::{Error, Result};
use crate::nn::embedding::Embedding;
use crate::nn::hadamard::HadamardRotation;
use crate::nn::module::Module;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, IndexingOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// A token embedding stored in a Hadamard-rotated basis: lookup, then
/// [`HadamardRotation::inverse`].
pub struct RotatedEmbedding<R: Runtime> {
    inner: Embedding<R>,
    rotation: HadamardRotation<R>,
}

impl<R: Runtime<DType = DType>> RotatedEmbedding<R> {
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `rotation` carries a sign
    /// vector whose width does not match `inner`'s embedding dim (`weight`'s
    /// last dim).
    pub fn new(inner: Embedding<R>, rotation: HadamardRotation<R>) -> Result<Self> {
        let embed_dim = embedding_dim(&inner);
        if let Some(width) = rotation.width()
            && width != embed_dim
        {
            return Err(Error::InvalidArgument {
                arg: "rotation",
                reason: format!(
                    "rotation sign width {width} does not match the embedding's embed_dim \
                     {embed_dim}"
                ),
            });
        }
        Ok(Self { inner, rotation })
    }

    /// Forward: lookup, then the inverse Hadamard rotation.
    ///
    /// `fwht` carries no autograd, so the result is always a detached leaf
    /// `Var` — mirrors how a quantized forward detaches in
    /// [`crate::nn::linear::MaybeQuantLinear`].
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + IndexingOps<R> + FwhtOps<R> + BinaryOps<R>,
        R::Client: IndexingOps<R>,
    {
        let looked_up = self.inner.forward(client, indices)?;
        let rotated = self.rotation.inverse(client, looked_up.tensor())?;
        Ok(Var::new(rotated, false))
    }

    pub fn weight(&self) -> &Var<R> {
        self.inner.weight()
    }

    pub fn embedding_dim(&self) -> usize {
        embedding_dim(&self.inner)
    }

    pub fn num_embeddings(&self) -> usize {
        self.inner.weight().shape()[0]
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.trainable_parameters()
    }
}

fn embedding_dim<R: Runtime>(embedding: &Embedding<R>) -> usize {
    let shape = embedding.weight().shape();
    shape[shape.len() - 1]
}

impl<R: Runtime<DType = DType>> Module<R> for RotatedEmbedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Module::parameters(&self.inner)
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        Module::named_parameters(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// A `RotatedEmbedding` over a Hadamard-ROTATED table, with the matching
    /// inverse rotation attached, must return the original rows.
    #[test]
    fn inverse_rotation_recovers_original_rows() {
        let (client, device) = cpu_setup();
        let vocab = 6;
        let dim = 16;
        let block = 8;

        let table_data: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let table = Tensor::<CpuRuntime>::from_slice(&table_data, &[vocab, dim], &device).unwrap();

        let signs: Vec<i8> = (0..dim).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(block, Some(&signs), DType::F32, &device).unwrap();

        // Rotate every row in one call — `fwht` applies per `block_size`
        // segment of the last axis regardless of leading (vocab) dims.
        let rotated_table = rotation.forward(&client, &table).unwrap();

        let rotation_for_embedding =
            HadamardRotation::<CpuRuntime>::new(block, Some(&signs), DType::F32, &device).unwrap();
        let emb =
            RotatedEmbedding::new(Embedding::new(rotated_table, false), rotation_for_embedding)
                .unwrap();

        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 3, 5], &[3], &device).unwrap();
        let out = emb.forward(&client, &indices).unwrap();
        let got: Vec<f32> = out.tensor().to_vec();

        let mut expected = Vec::with_capacity(3 * dim);
        for row in [0usize, 3, 5] {
            expected.extend_from_slice(&table_data[row * dim..(row + 1) * dim]);
        }

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
    }

    #[test]
    fn width_mismatch_errors() {
        let (_client, device) = cpu_setup();
        let table = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6 * 16], &[6, 16], &device).unwrap();
        let signs: Vec<i8> = vec![1; 8];
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs), DType::F32, &device).unwrap();

        assert!(RotatedEmbedding::new(Embedding::new(table, false), rotation).is_err());
    }
}
