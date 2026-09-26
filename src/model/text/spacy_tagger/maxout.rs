//! thinc `Maxout(normalize=True)`: affine map to `nO * nP` units, the
//! maximum over each group of `nP` pieces, then `LayerNorm`.

use super::weights::MaxoutWeights;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A Maxout layer with its `LayerNorm`, weights laid out for `x @ W`.
pub(super) struct MaxoutNorm<R: Runtime> {
    /// `W` as `[nI, nO * nP]`: thinc's `[nO, nP, nI]` flattened and
    /// transposed.
    w_t: Tensor<R>,
    /// `b` as `[nO * nP]`.
    b: Tensor<R>,
    norm_g: Tensor<R>,
    norm_b: Tensor<R>,
    n_out: usize,
    n_pieces: usize,
    eps: f32,
}

impl<R: Runtime<DType = DType>> MaxoutNorm<R> {
    /// Checks `weights` against `[n_out, n_pieces, n_in]` and lays `W` out
    /// for the forward matmul.
    pub(super) fn new(
        name: &str,
        weights: MaxoutWeights<R>,
        (n_out, n_pieces, n_in): (usize, usize, usize),
        eps: f32,
    ) -> Result<Self> {
        check_shape(&format!("{name}.W"), &weights.w, &[n_out, n_pieces, n_in])?;
        check_shape(&format!("{name}.b"), &weights.b, &[n_out, n_pieces])?;
        check_shape(&format!("{name}.norm.G"), &weights.norm_g, &[n_out])?;
        check_shape(&format!("{name}.norm.b"), &weights.norm_b, &[n_out])?;
        let w_t = weights
            .w
            .reshape(&[n_out * n_pieces, n_in])?
            .transpose(0, 1)?
            .contiguous()?;
        Ok(Self {
            w_t,
            b: weights.b.reshape(&[n_out * n_pieces])?,
            norm_g: weights.norm_g,
            norm_b: weights.norm_b,
            n_out,
            n_pieces,
            eps,
        })
    }

    /// `[n, nI]` -> `[n, nO]`: `max_p (x @ W + b)`, then `LayerNorm`.
    pub(super) fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let rows = x.shape().first().copied().unwrap_or(0);
        let y = client.matmul_bias(x, &self.w_t, &self.b)?;
        let y = y.reshape(&[rows, self.n_out, self.n_pieces])?;
        let best = client.max(&y, &[2], false)?;
        Ok(client.layer_norm(&best, &self.norm_g, &self.norm_b, self.eps)?)
    }
}

/// Refuses a tensor whose shape is not `want`, naming it.
pub(super) fn check_shape<R: Runtime>(name: &str, t: &Tensor<R>, want: &[usize]) -> Result<()> {
    if t.shape() == want {
        return Ok(());
    }
    Err(Error::ModelError {
        reason: format!(
            "spaCy tagger tensor {name} has shape {:?}, expected {want:?}",
            t.shape()
        ),
    })
}
