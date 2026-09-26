//! `MaxoutWindowEncoder.v2`: `with_array(clone(residual(expand_window >>
//! Maxout >> LayerNorm), depth), pad)`.
//!
//! `with_array` puts `pad` zero rows before and after the document; they run
//! through every block like tokens and are cut off at the end.
//! `expand_window` (thinc `seq2col`) reads zeros past both ends of that
//! padded array.

use super::config::SpacyTaggerConfig;
use super::maxout::MaxoutNorm;
use super::weights::MaxoutWeights;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// The CNN encoder stage of `tok2vec`.
pub(super) struct WindowEncoder<R: Runtime> {
    blocks: Vec<MaxoutNorm<R>>,
    window: usize,
    pad: usize,
}

impl<R: Runtime<DType = DType>> WindowEncoder<R> {
    pub(super) fn new(config: &SpacyTaggerConfig, blocks: Vec<MaxoutWeights<R>>) -> Result<Self> {
        if blocks.len() != config.depth {
            return Err(Error::ModelError {
                reason: format!(
                    "spaCy tagger has {} encoder blocks, config depth is {}",
                    blocks.len(),
                    config.depth
                ),
            });
        }
        let geometry = (config.width, config.encode_pieces, config.encode_in());
        let blocks = blocks
            .into_iter()
            .enumerate()
            .map(|(d, w)| {
                MaxoutNorm::new(
                    &format!("encode.{d}.maxout"),
                    w,
                    geometry,
                    config.layer_norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            window: config.window_size,
            pad: config.pad,
        })
    }

    /// thinc `seq2col`: row `i` of the output is rows `i - window ..= i +
    /// window` of `x` side by side, zeros past either end.
    fn expand_window<C>(&self, client: &C, x: &Tensor<R>, rows: usize) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        if self.window == 0 {
            return Ok(x.clone());
        }
        let padded = client.pad(x, &[0, 0, self.window, self.window], 0.0)?;
        let shifts = (0..=2 * self.window)
            .map(|k| padded.narrow(0, k, rows))
            .collect::<numr::error::Result<Vec<_>>>()?;
        let refs: Vec<&Tensor<R>> = shifts.iter().collect();
        Ok(client.cat(&refs, 1)?)
    }

    /// `[n, width]` -> `[n, width]` for one document of `n > 0` tokens.
    pub(super) fn forward<C>(&self, client: &C, x: &Tensor<R>, n_tokens: usize) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let rows = n_tokens + 2 * self.pad;
        let mut h = client.pad(x, &[0, 0, self.pad, self.pad], 0.0)?;
        for block in &self.blocks {
            let window = self.expand_window(client, &h, rows)?;
            let y = block.forward(client, &window)?;
            h = client.add(&h, &y)?;
        }
        Ok(h.narrow(0, self.pad, n_tokens)?.contiguous()?)
    }
}
