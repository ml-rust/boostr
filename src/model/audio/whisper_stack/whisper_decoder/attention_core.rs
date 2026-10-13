//! Shared attention core for the Whisper decoder: scaled matmul -> optional
//! additive mask -> softmax -> matmul -> merge heads. Every self/cross
//! attention path (cached and uncached) calls this so the four paths run
//! identical floating-point op order and differ only in where Q/K/V and the
//! mask come from.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ScalarOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `softmax((Q @ K^T) / sqrt(D) [+ mask]) @ V`, then merge heads back to
/// `[B, T, H*D]`.
///
/// - `q`/`k`/`v`: `[B, H, T(_kv), D]` (already reshaped to heads).
/// - `mask`: added to the scaled scores before softmax when given — the
///   caller builds it (e.g. a causal mask), already broadcast to the scores'
///   shape.
/// - `capture`: when given, receives a clone of the post-softmax attention
///   probabilities `[B, H, T, T_kv]`.
pub(super) fn attend<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    mask: Option<&Tensor<R>>,
    head_dim: usize,
    capture: Option<&mut Vec<Tensor<R>>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R> + ScalarOps<R> + ActivationOps<R>,
{
    let q_shape = q.shape();
    let batch = q_shape[0];
    let num_heads = q_shape[1];
    let tgt_len = q_shape[2];

    // k_t stays a view so the matmul reads k transposed without a copy.
    let k_t = k.transpose(-2, -1)?;
    let scale = (head_dim as f32).sqrt();
    let scores = client.matmul(q, &k_t).map_err(Error::Numr)?;
    let mut scores = client
        .div_scalar(&scores, scale as f64)
        .map_err(Error::Numr)?;

    if let Some(m) = mask {
        scores = client.add(&scores, m).map_err(Error::Numr)?;
    }

    let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
    if let Some(sink) = capture {
        sink.push(attn.clone());
    }
    let out = client.matmul(&attn, v).map_err(Error::Numr)?;

    out.transpose(1, 2)?
        .contiguous()?
        .reshape(&[batch, tgt_len, num_heads * head_dim])
        .map_err(Error::Numr)
}
