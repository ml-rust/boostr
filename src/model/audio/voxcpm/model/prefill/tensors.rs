//! Tensor helpers of the prefill body: one row's `audio_feat`, the last-row
//! slice and the mask upload.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{TensorOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One row's `audio_feat`, `[1, S_b, patch_size, feat_dim]`.
///
/// Reference mode: `z1 ++ ref_feat ++ z1 ++ zeros(text_length)`, one leading
/// zero patch and `1 + text_length` trailing ones. No-reference mode:
/// `zeros(text_length)` alone. Both bookends go with the delimiters they
/// back — see `SequenceLayout`.
#[allow(clippy::too_many_arguments)]
pub(super) fn row_audio_feat<R, C>(
    client: &C,
    ref_feat: Option<&Tensor<R>>,
    t_ref: usize,
    text_length: usize,
    patch_size: usize,
    feat_dim: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R> + TypeConversionOps<R>,
{
    let Some(ref_feat) = ref_feat else {
        return Ok(Var::new(
            Tensor::<R>::zeros(&[1, text_length, patch_size, feat_dim], dtype, device)?,
            false,
        ));
    };
    let ref_patches = Var::new(
        ref_feat
            .to_dtype(dtype)?
            .reshape(&[1, t_ref, patch_size, feat_dim])?,
        false,
    );
    let head = Var::new(
        Tensor::<R>::zeros(&[1, 1, patch_size, feat_dim], dtype, device)?,
        false,
    );
    let tail = Var::new(
        Tensor::<R>::zeros(&[1, 1 + text_length, patch_size, feat_dim], dtype, device)?,
        false,
    );
    Ok(var_cat(&[&head, &ref_patches, &tail], 1, client)?)
}

/// `[B, S, hidden]` -> `[B, hidden]`, taking row `seq_len - 1`.
///
/// `narrow` yields a strided view; `reshape` needs it materialized first.
pub(super) fn last_row<R: Runtime<DType = DType>>(x: &Var<R>, seq_len: usize) -> Result<Var<R>>
where
    R::Client: TensorOps<R>,
{
    let shape = x.shape().to_vec();
    if shape.len() != 3 || shape[1] != seq_len {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [batch, {seq_len}, hidden], got {shape:?}"),
        });
    }
    let row = var_contiguous(&var_narrow(x, 1, seq_len - 1, 1)?)?;
    Ok(var_reshape(&row, &[shape[0], shape[2]])?)
}

/// Upload a per-position mask as `[B, S, 1]` in `dtype`, ready to broadcast
/// against `[B, S, hidden]`.
pub(super) fn mask_var<R: Runtime<DType = DType>>(
    mask: &[f32],
    batch: usize,
    seq_len: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Var<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let tensor = Tensor::<R>::from_slice(mask, &[batch, seq_len, 1], device)?;
    Ok(Var::new(tensor.to_dtype(dtype)?, false))
}
