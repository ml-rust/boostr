//! [`SwinStage`]: the blocks of one HTSAT stage, then, on every stage but
//! the last, patch merging: each 2×2 group of tokens concatenated,
//! normalized and projected to twice the width. Port of transformers'
//! `ClapAudioStage` and `ClapAudioPatchMerging`.

use super::linear::{dense, layer_norm, linear, norm};
use super::swin_block::{BlockShape, SwinBlock};
use crate::error::Result;
use crate::nn::{Activation, LayerNorm, Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ScalarOps, ShapeOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// The patch merging that ends a stage.
struct PatchMerging<R: Runtime> {
    norm: LayerNorm<R>,
    reduction: Linear<R>,
}

/// One stage: its blocks over a `height × width` grid.
pub struct SwinStage<R: Runtime> {
    blocks: Vec<SwinBlock<R>>,
    merge: Option<PatchMerging<R>>,
    dim: usize,
    height: usize,
    width: usize,
}

impl<R: Runtime<DType = DType>> SwinStage<R> {
    /// Reads `blocks.{i}.*` and, when `merges`, `downsample.*` under `vb`.
    /// `first` is the shape of block 0; odd blocks shift.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        first: BlockShape,
        depth: usize,
        merges: bool,
        act: Activation,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + IndexingOps<R>,
    {
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let mut b = vb.pp(&format!("blocks.{i}"));
            let shape = BlockShape {
                shifted: i % 2 == 1,
                ..first
            };
            blocks.push(SwinBlock::from_varbuilder(client, &mut b, shape, act)?);
        }
        let merge = if merges {
            let mut d = vb.pp("downsample");
            let norm = layer_norm(&mut d, "norm", 1e-5)?;
            let reduction = linear(&mut d, "reduction", false)?;
            Some(PatchMerging { norm, reduction })
        } else {
            None
        };
        Ok(Self {
            blocks,
            merge,
            dim: first.dim,
            height: first.height,
            width: first.width,
        })
    }

    /// The grid after this stage: halved, rounded up, when it merges.
    pub fn output_grid(&self) -> (usize, usize) {
        match self.merge {
            Some(_) => (self.height.div_ceil(2), self.width.div_ceil(2)),
            None => (self.height, self.width),
        }
    }

    /// `x`: `[batch, height · width, dim]`. Out: `[batch, tokens, dim']`
    /// on the [`Self::output_grid`].
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + ActivationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(client, &h)?;
        }
        let Some(merge) = &self.merge else {
            return Ok(h);
        };
        let (batch, c) = (h.shape()[0], self.dim);
        let mut grid = h.reshape(&[batch, self.height, self.width, c])?;
        let (hp, wp) = (self.height.div_ceil(2) * 2, self.width.div_ceil(2) * 2);
        if hp != self.height || wp != self.width {
            grid = client.pad(&grid, &[0, 0, 0, wp - self.width, 0, hp - self.height], 0.0)?;
        }
        // transformers concatenates (row 0, col 0), (row 1, col 0),
        // (row 0, col 1), (row 1, col 1): column offset major, row offset
        // minor.
        let merged = grid
            .reshape(&[batch, hp / 2, 2, wp / 2, 2, c])?
            .permute(&[0, 1, 3, 4, 2, 5])?
            .contiguous()?
            .reshape(&[batch, (hp / 2) * (wp / 2), 4 * c])?;
        let merged = norm(client, &merge.norm, &merged)?;
        dense(client, &merge.reduction, &merged)
    }
}
