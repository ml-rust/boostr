//! [`SwinBlock`]: one HTSAT block. LayerNorm, attention within windows
//! (shifted by half a window on every other block), a residual, then
//! LayerNorm, an MLP and a residual. Port of transformers'
//! `ClapAudioLayer` at inference.

use super::linear::{dense, layer_norm, linear, norm};
use super::window_attention::WindowAttention;
use crate::error::{Error, Result};
use crate::nn::{Activation, LayerNorm, Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ScalarOps, ShapeOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// The shape of one block: its width, heads, input grid and window.
#[derive(Debug, Clone, Copy)]
pub struct BlockShape {
    pub dim: usize,
    pub heads: usize,
    pub height: usize,
    pub width: usize,
    /// The configured window side.
    pub window: usize,
    /// Whether this is a shifted (odd) block.
    pub shifted: bool,
    pub mlp_ratio: f64,
    pub qkv_bias: bool,
    pub eps: f64,
}

/// One HTSAT block.
pub struct SwinBlock<R: Runtime> {
    norm_before: LayerNorm<R>,
    attention: WindowAttention<R>,
    norm_after: LayerNorm<R>,
    mlp_in: Linear<R>,
    mlp_out: Linear<R>,
    act: Activation,
    shape: BlockShape,
    shift: usize,
    /// `[windows, N, N]` of `0` and `-100` for a shifted block.
    mask: Option<Tensor<R>>,
}

/// Rows padded up to a whole number of windows.
fn padded(n: usize, window: usize) -> usize {
    n.div_ceil(window) * window
}

/// The region id of each position of a `height × width` grid shifted by
/// `shift`, in windows: transformers' `img_mask` after `window_partition`,
/// then the pairwise `0` / `-100` mask.
fn shift_mask(height: usize, width: usize, window: usize, shift: usize) -> Vec<f32> {
    let region = |p: usize, n: usize| {
        if p < n - window {
            0
        } else if p < n - shift {
            1
        } else {
            2
        }
    };
    let (rows, cols) = (height / window, width / window);
    let tokens = window * window;
    let mut out = Vec::with_capacity(rows * cols * tokens * tokens);
    for wr in 0..rows {
        for wc in 0..cols {
            let ids: Vec<usize> = (0..tokens)
                .map(|t| {
                    let (h, w) = (wr * window + t / window, wc * window + t % window);
                    region(h, height) * 3 + region(w, width)
                })
                .collect();
            for &a in &ids {
                for &b in &ids {
                    out.push(if a == b { 0.0 } else { -100.0 });
                }
            }
        }
    }
    out
}

impl<R: Runtime<DType = DType>> SwinBlock<R> {
    /// Reads `layernorm_before`, `attention`, `layernorm_after`,
    /// `intermediate.dense` and `output.dense` under `vb`.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        shape: BlockShape,
        act: Activation,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + IndexingOps<R>,
    {
        // transformers: a grid no larger than the window is one window,
        // never shifted.
        let mut window = shape.window;
        let mut shift = if shape.shifted { window / 2 } else { 0 };
        if shape.height.min(shape.width) <= window {
            shift = 0;
            window = shape.height.min(shape.width);
        }
        if window != shape.window {
            return Err(Error::ModelError {
                reason: format!(
                    "CLAP grid {}x{} is smaller than the window {}; the bias table does not fit",
                    shape.height, shape.width, shape.window
                ),
            });
        }
        let norm_before = layer_norm(vb, "layernorm_before", shape.eps)?;
        let mut a = vb.pp("attention");
        let attention = WindowAttention::from_varbuilder(
            client,
            &mut a,
            shape.dim,
            shape.heads,
            window,
            shape.qkv_bias,
        )?;
        drop(a);
        let norm_after = layer_norm(vb, "layernorm_after", shape.eps)?;
        let mut i = vb.pp("intermediate");
        let mlp_in = linear(&mut i, "dense", true)?;
        drop(i);
        let mut o = vb.pp("output");
        let mlp_out = linear(&mut o, "dense", true)?;
        drop(o);
        let mask = if shift > 0 {
            let (hp, wp) = (padded(shape.height, window), padded(shape.width, window));
            let data = shift_mask(hp, wp, window, shift);
            let windows = (hp / window) * (wp / window);
            let tokens = window * window;
            Some(Tensor::<R>::from_slice(
                &data,
                &[windows, tokens, tokens],
                vb.device(),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm_before,
            attention,
            norm_after,
            mlp_in,
            mlp_out,
            act,
            shape,
            shift,
            mask,
        })
    }

    /// `x`: `[batch, height · width, dim]`, the same shape out.
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
        let s = self.shape;
        let (batch, c, ws) = (x.shape()[0], s.dim, s.window);
        let (hp, wp) = (padded(s.height, ws), padded(s.width, ws));
        let mut h = norm(client, &self.norm_before, x)?.reshape(&[batch, s.height, s.width, c])?;
        if hp != s.height || wp != s.width {
            h = client.pad(&h, &[0, 0, 0, wp - s.width, 0, hp - s.height], 0.0)?;
        }
        let shift = self.shift as isize;
        if shift > 0 {
            h = client.roll(&client.roll(&h, -shift, 1)?, -shift, 2)?;
        }
        let (rows, cols) = (hp / ws, wp / ws);
        let windows = h
            .reshape(&[batch, rows, ws, cols, ws, c])?
            .permute(&[0, 1, 3, 2, 4, 5])?
            .contiguous()?
            .reshape(&[batch * rows * cols, ws * ws, c])?;
        let attended = self
            .attention
            .forward(client, &windows, self.mask.as_ref())?;
        let mut h = attended
            .reshape(&[batch, rows, cols, ws, ws, c])?
            .permute(&[0, 1, 3, 2, 4, 5])?
            .contiguous()?
            .reshape(&[batch, hp, wp, c])?;
        if shift > 0 {
            h = client.roll(&client.roll(&h, shift, 1)?, shift, 2)?;
        }
        if hp != s.height || wp != s.width {
            h = h
                .narrow(1, 0, s.height)?
                .narrow(2, 0, s.width)?
                .contiguous()?;
        }
        let h = client.add(x, &h.reshape(&[batch, s.height * s.width, c])?)?;
        let y = norm(client, &self.norm_after, &h)?;
        let y = self
            .act
            .forward(client, &dense(client, &self.mlp_in, &y)?)?;
        Ok(client.add(&h, &dense(client, &self.mlp_out, &y)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_mask_separates_wrapped_regions() {
        // A 4x4 grid in 2x2 windows, shifted by 1: the last window holds
        // tokens from four regions; the first holds one.
        let m = shift_mask(4, 4, 2, 1);
        assert_eq!(m.len(), 4 * 4 * 4);
        assert!(m[..16].iter().all(|&v| v == 0.0));
        let last = &m[48..64];
        assert_eq!(last.iter().filter(|&&v| v == 0.0).count(), 4);
    }

    #[test]
    fn padding_rounds_up_to_windows() {
        assert_eq!(padded(64, 8), 64);
        assert_eq!(padded(9, 8), 16);
    }
}
