//! An embedding whose stored table needs an inverse Hadamard rotation
//! after lookup — llama.cpp's `inverse_weight_names`
//! contract (see `crate::format::gguf::hadamard_contract`), e.g. a GGUF
//! `token_embd.weight` stored in the rotated basis.
//!
//! The inner table is a [`MaybeQuantEmbedding`]: Bonsai ships
//! `token_embd.weight` block-quantized (PQ2_0) AND rotated, so the wrapper
//! must accept a packed table, not only a dense one.
//!
//! Kept as its own wrapper rather than a field on [`Embedding`](crate::nn::Embedding) itself:
//! `Embedding::forward`'s bounds are shared by every caller in the crate
//! (`src/model/**`), and widening them with `FwhtOps`/`BinaryOps` for an
//! optional rotation forces every existing call site to prove those bounds
//! too.
//!
//! [`MaybeRotatedEmbedding`] is the `Plain`/`Rotated` dispatch enum, the
//! embedding analogue of [`crate::nn::linear::MaybeRotatedLinear`].

use crate::error::{Error, Result};
use crate::nn::hadamard::HadamardRotation;
use crate::nn::module::Module;
use crate::nn::quant_embedding::MaybeQuantEmbedding;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, IndexingOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// A token embedding stored in a Hadamard-rotated basis: lookup, then
/// [`HadamardRotation::inverse`].
pub struct RotatedEmbedding<R: Runtime> {
    inner: MaybeQuantEmbedding<R>,
    rotation: HadamardRotation<R>,
}

impl<R: Runtime<DType = DType>> RotatedEmbedding<R> {
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `rotation` carries a sign
    /// vector whose width does not match `inner`'s embedding dim (the
    /// table's last dim).
    pub fn new(inner: MaybeQuantEmbedding<R>, rotation: HadamardRotation<R>) -> Result<Self> {
        let embed_dim = table_shape(&inner)[1];
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
        C: RuntimeClient<R> + IndexingOps<R> + FwhtOps<R> + BinaryOps<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        let looked_up = self.inner.forward(client, indices)?;
        let rotated = self.rotation.inverse(client, looked_up.tensor())?;
        Ok(Var::new(rotated, false))
    }

    /// The wrapped table, rotation aside.
    pub fn base(&self) -> &MaybeQuantEmbedding<R> {
        &self.inner
    }

    /// The dense weight. `None` when the table is block-quantized.
    pub fn weight(&self) -> Option<&Var<R>> {
        match &self.inner {
            MaybeQuantEmbedding::Standard(emb) => Some(emb.weight()),
            MaybeQuantEmbedding::Quantized(_) => None,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        table_shape(&self.inner)[1]
    }

    pub fn num_embeddings(&self) -> usize {
        table_shape(&self.inner)[0]
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.trainable_parameters()
    }
}

/// Logical `[num_embeddings, embed_dim]` of either table variant.
fn table_shape<R: Runtime<DType = DType>>(embedding: &MaybeQuantEmbedding<R>) -> [usize; 2] {
    let shape: &[usize] = match embedding {
        MaybeQuantEmbedding::Standard(emb) => emb.weight().shape(),
        MaybeQuantEmbedding::Quantized(qemb) => qemb.weight().shape(),
    };
    let rows = shape.first().copied().unwrap_or(0);
    let cols = shape.last().copied().unwrap_or(0);
    [rows, cols]
}

impl<R: Runtime<DType = DType>> Module<R> for RotatedEmbedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Module::parameters(&self.inner)
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        Module::named_parameters(&self.inner)
    }
}

/// A token embedding that is either plain or Hadamard-rotated, mirroring
/// [`crate::nn::linear::MaybeRotatedLinear`]'s `Plain`/`Rotated` dispatch.
pub enum MaybeRotatedEmbedding<R: Runtime> {
    Plain(MaybeQuantEmbedding<R>),
    Rotated(RotatedEmbedding<R>),
}

impl<R: Runtime<DType = DType>> MaybeRotatedEmbedding<R> {
    /// Forward pass: plain lookup, or lookup-then-inverse-rotate.
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + IndexingOps<R> + FwhtOps<R> + BinaryOps<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        match self {
            Self::Plain(base) => base.forward(client, indices),
            Self::Rotated(rotated) => rotated.forward(client, indices),
        }
    }

    /// The underlying table, rotation aside.
    pub fn base(&self) -> &MaybeQuantEmbedding<R> {
        match self {
            Self::Plain(base) => base,
            Self::Rotated(rotated) => rotated.base(),
        }
    }

    /// The dense weight. `None` when the table is block-quantized.
    pub fn weight(&self) -> Option<&Var<R>> {
        match self.base() {
            MaybeQuantEmbedding::Standard(emb) => Some(emb.weight()),
            MaybeQuantEmbedding::Quantized(_) => None,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        table_shape(self.base())[1]
    }

    pub fn num_embeddings(&self) -> usize {
        table_shape(self.base())[0]
    }

    /// `true` when a Hadamard rotation is attached.
    pub fn is_rotated(&self) -> bool {
        matches!(self, Self::Rotated(_))
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().trainable_parameters()
    }

    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        self.base().named_parameters()
    }
}

impl<R: Runtime<DType = DType>> From<MaybeQuantEmbedding<R>> for MaybeRotatedEmbedding<R> {
    fn from(base: MaybeQuantEmbedding<R>) -> Self {
        Self::Plain(base)
    }
}

impl<R: Runtime<DType = DType>> From<RotatedEmbedding<R>> for MaybeRotatedEmbedding<R> {
    fn from(rotated: RotatedEmbedding<R>) -> Self {
        Self::Rotated(rotated)
    }
}

impl<R: Runtime<DType = DType>> Module<R> for MaybeRotatedEmbedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeRotatedEmbedding::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeRotatedEmbedding::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedEmbedding::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedEmbedding::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Embedding;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn standard(table: Tensor<CpuRuntime>) -> MaybeQuantEmbedding<CpuRuntime> {
        MaybeQuantEmbedding::Standard(Embedding::new(table, false))
    }

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
        let emb = RotatedEmbedding::new(standard(rotated_table), rotation_for_embedding).unwrap();
        assert_eq!(emb.num_embeddings(), vocab);
        assert_eq!(emb.embedding_dim(), dim);

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

        assert!(RotatedEmbedding::new(standard(table), rotation).is_err());
    }

    #[test]
    fn maybe_rotated_plain_matches_bare_lookup() {
        let (client, device) = cpu_setup();
        let table_data: Vec<f32> = (0..4 * 8).map(|i| i as f32).collect();
        let table = Tensor::<CpuRuntime>::from_slice(&table_data, &[4, 8], &device).unwrap();
        let plain = MaybeRotatedEmbedding::Plain(standard(table));
        assert!(!plain.is_rotated());
        assert_eq!(plain.num_embeddings(), 4);
        assert_eq!(plain.embedding_dim(), 8);

        let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 1], &[2], &device).unwrap();
        let got: Vec<f32> = plain.forward(&client, &indices).unwrap().tensor().to_vec();
        let mut expected = table_data[16..24].to_vec();
        expected.extend_from_slice(&table_data[8..16]);
        assert_eq!(got, expected);
    }
}
