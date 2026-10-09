//! [`WindowAttention`]: the self-attention of one HTSAT (Swin) block, over
//! the tokens of each window, with a learned bias per relative offset and
//! head. Port of transformers' `ClapAudioSelfAttention` and
//! `ClapAudioSelfOutput`.

use super::linear::{dense, linear};
use crate::error::{Error, Result};
use crate::nn::{Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Attention within windows of `window × window` tokens.
pub struct WindowAttention<R: Runtime> {
    query: Linear<R>,
    key: Linear<R>,
    value: Linear<R>,
    output: Linear<R>,
    /// `[1, heads, N, N]`, `N = window²`: the bias of each token pair.
    bias: Tensor<R>,
    heads: usize,
    head_dim: usize,
    tokens: usize,
}

/// Row of the bias table for the pair `(a, b)` of a `window × window`
/// window: transformers' `relative_position_index`.
fn relative_index(window: usize) -> Vec<i64> {
    let n = window * window;
    let span = 2 * window as i64 - 1;
    let mut out = Vec::with_capacity(n * n);
    for a in 0..n {
        for b in 0..n {
            let dh = (a / window) as i64 - (b / window) as i64 + window as i64 - 1;
            let dw = (a % window) as i64 - (b % window) as i64 + window as i64 - 1;
            out.push(dh * span + dw);
        }
    }
    out
}

impl<R: Runtime<DType = DType>> WindowAttention<R> {
    /// Reads `self.{query,key,value}.*`, `self.relative_position_bias_table`
    /// and `output.dense.*` under `vb`.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        dim: usize,
        heads: usize,
        window: usize,
        qkv_bias: bool,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + IndexingOps<R>,
    {
        if heads == 0 || !dim.is_multiple_of(heads) {
            return Err(Error::ModelError {
                reason: format!("CLAP block width {dim} does not split into {heads} heads"),
            });
        }
        let mut s = vb.pp("self");
        let query = linear(&mut s, "query", qkv_bias)?;
        let key = linear(&mut s, "key", qkv_bias)?;
        let value = linear(&mut s, "value", qkv_bias)?;
        let table = s.take_tensor("relative_position_bias_table")?;
        drop(s);
        let mut o = vb.pp("output");
        let output = linear(&mut o, "dense", true)?;
        drop(o);

        let tokens = window * window;
        let index = relative_index(window);
        let index = Tensor::<R>::from_slice(&index, &[index.len()], table.device())?;
        let bias = client
            .index_select(&table, 0, &index)?
            .reshape(&[tokens, tokens, heads])?
            .permute(&[2, 0, 1])?
            .contiguous()?
            .reshape(&[1, heads, tokens, tokens])?;
        Ok(Self {
            query,
            key,
            value,
            output,
            bias,
            heads,
            head_dim: dim / heads,
            tokens,
        })
    }

    /// `[windows, N, dim] → [windows, heads, N, head_dim]`.
    fn split_heads(&self, x: &Tensor<R>, windows: usize) -> Result<Tensor<R>> {
        Ok(
            x.reshape(&[windows, self.tokens, self.heads, self.head_dim])?
                .transpose(1, 2)?
                .contiguous()?,
        )
    }

    /// `x`: `[batch · windows, N, dim]`. `mask`, for a shifted block:
    /// `[windows, N, N]` of `0` and `-100`, the same for every batch item.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + BinaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>,
    {
        let windows = x.shape()[0];
        let dim = self.heads * self.head_dim;
        let q = self.split_heads(&dense(client, &self.query, x)?, windows)?;
        let k = self.split_heads(&dense(client, &self.key, x)?, windows)?;
        let v = self.split_heads(&dense(client, &self.value, x)?, windows)?;
        let k_t = k.transpose(-2, -1)?.contiguous()?;
        let scores = client.matmul(&q, &k_t)?;
        let scores = client.div_scalar(&scores, (self.head_dim as f64).sqrt())?;
        let mut scores = client.add(&scores, &self.bias)?;
        if let Some(mask) = mask {
            let per_image = mask.shape()[0];
            let shape = [
                windows / per_image,
                per_image,
                self.heads,
                self.tokens,
                self.tokens,
            ];
            let masked = client.add(
                &scores.reshape(&shape)?,
                &mask.reshape(&[1, per_image, 1, self.tokens, self.tokens])?,
            )?;
            scores = masked.reshape(&[windows, self.heads, self.tokens, self.tokens])?;
        }
        let probs = client.softmax(&scores, -1)?;
        let context = client
            .matmul(&probs, &v)?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[windows, self.tokens, dim])?;
        dense(client, &self.output, &context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_index_matches_transformers() {
        // window 2: offsets (dh, dw) in {-1, 0, 1}², table side 3.
        let idx = relative_index(2);
        assert_eq!(idx.len(), 16);
        // Same token: offset (0, 0) -> (1, 1) -> 1 * 3 + 1.
        assert_eq!(idx[0], 4);
        // Token 0 against token 3 (row 1, col 1): offset (-1, -1) -> 0.
        assert_eq!(idx[3], 0);
        // Token 3 against token 0: offset (1, 1) -> (2, 2) -> 8.
        assert_eq!(idx[12], 8);
    }
}
