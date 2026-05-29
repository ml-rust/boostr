//! Shared loading and attention helpers for the Whisper decoder.

use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn load_layernorm<R: Runtime>(
    vb: &mut VarBuilder<'_, R>,
    name: &str,
) -> Result<LayerNorm<R>> {
    let mut ln_vb = vb.pp(name);
    let w = ln_vb.take_tensor("weight")?;
    let b = ln_vb.take_tensor("bias")?;
    Ok(LayerNorm::new(w, b, 1e-5, false))
}

type QkvOutLinears<R> = (Linear<R>, Linear<R>, Linear<R>, Linear<R>);

pub(super) fn load_attn<R: Runtime>(
    vb: &mut VarBuilder<'_, R>,
    prefix: &str,
) -> Result<QkvOutLinears<R>> {
    let mut attn_vb = vb.pp(prefix);
    let q = Linear::new(
        attn_vb.take_tensor("q_proj.weight")?,
        attn_vb.take_tensor_optional("q_proj.bias")?,
        false,
    );
    // Whisper's k_proj has no bias; use optional and accept either form.
    let k = Linear::new(
        attn_vb.take_tensor("k_proj.weight")?,
        attn_vb.take_tensor_optional("k_proj.bias")?,
        false,
    );
    let v = Linear::new(
        attn_vb.take_tensor("v_proj.weight")?,
        attn_vb.take_tensor_optional("v_proj.bias")?,
        false,
    );
    let out = Linear::new(
        attn_vb.take_tensor("out_proj.weight")?,
        attn_vb.take_tensor_optional("out_proj.bias")?,
        false,
    );
    Ok((q, k, v, out))
}

pub(super) fn reshape_heads<R: Runtime>(
    x: &Tensor<R>,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor<R>> {
    x.reshape(&[batch, seq, num_heads, head_dim])
        .and_then(|t| t.transpose(1, 2))
        .map(|t| t.contiguous())
        .map_err(Error::Numr)
}

/// Add a causal (upper-triangular) mask of -inf to attention scores.
///
/// `scores`: `[B, H, T, T]`. We build a broadcasted mask `[1, 1, T, T]` with
/// `f32::NEG_INFINITY` above the diagonal and 0 on/below it, then add.
pub(super) fn apply_causal_mask<R, C>(
    client: &C,
    scores: Tensor<R>,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R>,
{
    let device = scores.device();
    // Build mask on CPU then upload; runs once per decoder block per step but
    // construction cost is O(T^2) which is tiny vs attention matmul.
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    // Upload as f32; cast to scores' dtype is handled by runtime on add-broadcast.
    let mask_tensor = Tensor::<R>::from_slice(&mask, &[1, 1, seq_len, seq_len], device);
    let mask_b = mask_tensor
        .broadcast_to(&[batch, num_heads, seq_len, seq_len])
        .map_err(Error::Numr)?
        .contiguous();
    client.add(&scores, &mask_b).map_err(Error::Numr)
}
