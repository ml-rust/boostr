//! Tensor helpers of the prefill body: the batch's `audio_feat`, each
//! row's continuation decode context, the last-row and last-patch slices and
//! the mask upload.

use super::rows::PaddedBatch;
use super::state::PrefillRow;
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::config::STREAMING_PREFIX_LEN;
use crate::model::audio::voxcpm::model::sequence::SequenceLayout;
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{TensorOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The batch's `audio_feat` and each row's continuation decode context.
pub(super) type BatchAudioFeat<R> = (Var<R>, Vec<Vec<Var<R>>>);

/// The batch's `audio_feat`, `[B, S_max, patch_size, feat_dim]` in `dtype`,
/// each row's pad in front, plus each row's continuation decode context
/// (see [`continuation_context`]). Rows are stacked only when there is more
/// than one.
pub(super) fn batch_audio_feat<R, C>(
    client: &C,
    rows: &[PrefillRow<'_, R>],
    padded: &PaddedBatch,
    patch_size: usize,
    feat_dim: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<BatchAudioFeat<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R> + TypeConversionOps<R>,
{
    let mut per_row = Vec::with_capacity(rows.len());
    let mut context = Vec::with_capacity(rows.len());
    for (b, (row, layout)) in rows.iter().zip(&padded.layouts).enumerate() {
        let feat = row_audio_feat(client, row, layout, patch_size, feat_dim, dtype, device)?;
        context.push(continuation_context(&feat, layout, patch_size, feat_dim)?);
        let pad = padded.pad(b);
        per_row.push(if pad == 0 {
            feat
        } else {
            let front = Var::new(
                Tensor::<R>::zeros(&[1, pad, patch_size, feat_dim], dtype, device)?,
                false,
            );
            var_cat(&[&front, &feat], 1, client)?
        });
    }
    let audio_feat = if per_row.len() == 1 {
        per_row.swap_remove(0)
    } else {
        let refs: Vec<&Var<R>> = per_row.iter().collect();
        var_cat(&refs, 0, client)?
    };
    Ok((audio_feat, context))
}

/// One row's `audio_feat`, `[1, S_b, patch_size, feat_dim]`.
///
/// Reference prefix: `z1 ++ ref_feat ++ z1 ++ zeros(text_length)`, one
/// leading zero patch and `1 + text_length` trailing ones. No reference:
/// `zeros(text_length)` alone. Both bookends go with the delimiters they
/// back — see `SequenceLayout`. A continuation tail appends `prompt_feat`
/// after either form.
fn row_audio_feat<R, C>(
    client: &C,
    row: &PrefillRow<'_, R>,
    layout: &SequenceLayout,
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
    let zeros = |len: usize| -> Result<Var<R>> {
        Ok(Var::new(
            Tensor::<R>::zeros(&[1, len, patch_size, feat_dim], dtype, device)?,
            false,
        ))
    };
    let patches = |feat: &Tensor<R>, len: usize| -> Result<Var<R>> {
        Ok(Var::new(
            feat.to_dtype(dtype)?
                .reshape(&[1, len, patch_size, feat_dim])?,
            false,
        ))
    };
    let mut parts = Vec::with_capacity(4);
    match row.ref_feat {
        Some(ref_feat) => {
            parts.push(zeros(1)?);
            parts.push(patches(ref_feat, layout.t_ref)?);
            parts.push(zeros(1 + layout.text_length)?);
        }
        None => parts.push(zeros(layout.text_length)?),
    }
    if let Some(prompt_feat) = row.prompt_feat {
        parts.push(patches(prompt_feat, layout.prompt_len)?);
    }
    if parts.len() == 1 {
        return Ok(parts.swap_remove(0));
    }
    let refs: Vec<&Var<R>> = parts.iter().collect();
    Ok(var_cat(&refs, 1, client)?)
}

/// The decode context of one row: the last `min(STREAMING_PREFIX_LEN - 1,
/// audio positions)` audio patches of `feat` (`[1, S_b, patch_size,
/// feat_dim]`), each `[1, patch_size, feat_dim]`, in sequence order.
///
/// Empty unless the row ends in a continuation tail, which is the
/// reference's `feat_mask[-1] == 1` test. The count runs over EVERY audio
/// position, the reference span included, so a tail shorter than the
/// context reaches back into the reference audio exactly as the reference
/// does. Positions come from the host mask, and each patch is a device-side
/// slice of `feat`.
fn continuation_context<R: Runtime<DType = DType>>(
    feat: &Var<R>,
    layout: &SequenceLayout,
    patch_size: usize,
    feat_dim: usize,
) -> Result<Vec<Var<R>>> {
    if layout.prompt_len == 0 {
        return Ok(Vec::new());
    }
    let audio: Vec<usize> = (0..layout.seq_len())
        .filter(|&i| layout.audio_mask[i] == 1.0)
        .collect();
    let len = (STREAMING_PREFIX_LEN - 1).min(audio.len());
    audio[audio.len() - len..]
        .iter()
        .map(|&position| -> Result<Var<R>> {
            let patch = feat
                .tensor()
                .narrow(1, position, 1)?
                .contiguous()?
                .reshape(&[1, patch_size, feat_dim])?;
            Ok(Var::new(patch, false))
        })
        .collect()
}

/// `[B, S, patch_size, feat_dim]` -> `[B, patch_size, feat_dim]`, taking
/// position `seq_len - 1`: each row's last raw `audio_feat` patch.
pub(super) fn last_patch<R: Runtime<DType = DType>>(
    audio_feat: &Var<R>,
    seq_len: usize,
) -> Result<Var<R>>
where
    R::Client: TensorOps<R>,
{
    let shape = audio_feat.shape().to_vec();
    if shape.len() != 4 || shape[1] != seq_len || seq_len == 0 {
        return Err(Error::InvalidArgument {
            arg: "audio_feat",
            reason: format!(
                "expected [batch, {seq_len}, patch_size, feat_dim] with {seq_len} >= 1, \
                 got {shape:?}"
            ),
        });
    }
    let patch = var_contiguous(&var_narrow(audio_feat, 1, seq_len - 1, 1)?)?;
    Ok(var_reshape(&patch, &[shape[0], shape[2], shape[3]])?)
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
