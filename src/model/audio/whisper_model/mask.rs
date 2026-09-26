//! Shared masking and argmax helpers for greedy Whisper decoding.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Build an additive `[1, 1, vocab]` mask holding `0` at every id in `ids` and
/// `-inf` elsewhere, so an argmax over the masked logits can only land on one
/// of `ids`. The inverse of [`suppression_mask`].
pub(super) fn allow_only_mask<R>(
    ids: &[u32],
    vocab_size: usize,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    if ids.is_empty() {
        return Err(Error::ModelError {
            reason: "allow_only_mask needs at least one candidate id".into(),
        });
    }
    let mut mask = vec![f32::NEG_INFINITY; vocab_size];
    let mut allowed = 0usize;
    for &id in ids {
        if let Some(slot) = mask.get_mut(id as usize) {
            *slot = 0.0;
            allowed += 1;
        }
    }
    if allowed == 0 {
        return Err(Error::ModelError {
            reason: format!(
                "none of the {} candidate ids fall inside the {vocab_size}-entry vocabulary",
                ids.len()
            ),
        });
    }
    let tensor = Tensor::<R>::from_slice(&mask, &[1, 1, vocab_size], device)?;
    Ok(tensor)
}

/// Build an additive `[1, 1, vocab]` suppression mask holding `-inf` at every
/// id in `ids` and `0` elsewhere. Returns `None` when `ids` is empty, so the
/// caller can skip the add entirely.
pub(super) fn suppression_mask<R>(
    ids: &[u32],
    vocab_size: usize,
    device: &R::Device,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
{
    if ids.is_empty() {
        return Ok(None);
    }
    let mut mask = vec![0.0f32; vocab_size];
    for &id in ids {
        if (id as usize) < vocab_size {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    let tensor = Tensor::<R>::from_slice(&mask, &[1, 1, vocab_size], device)?;
    Ok(Some(tensor))
}

/// Greedy-pick the argmax over the vocab dimension at the **last** time step of
/// a logits tensor `[B, T, vocab]` (`B==1` here).
///
/// `mask` is an additive `-inf` suppression mask from [`suppression_mask`], or
/// `None` when nothing is suppressed. The argmax runs on-device; only the
/// resulting index crosses back to the host.
pub(super) fn greedy_pick_last<R, C>(
    client: &C,
    logits: &Tensor<R>,
    mask: Option<&Tensor<R>>,
) -> Result<u32>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + IndexingOps<R>,
{
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(Error::ModelError {
            reason: format!("greedy_pick_last expects [B, T, vocab] logits, got {shape:?}"),
        });
    }
    let t = shape[1];
    // Unreachable on today's call paths — `generate` rejects an empty prefix and
    // every later step feeds a `[1, 1]` id tensor — but `t - 1` on a `usize` is
    // an underflow, not a caught mistake, so it is refused rather than assumed.
    if t == 0 {
        return Err(Error::ModelError {
            reason: "greedy_pick_last got zero time steps".to_string(),
        });
    }
    // Slice last time-step: [1, 1, vocab]
    let last = logits.narrow(1, t - 1, 1).map_err(Error::Numr)?;

    let scored = match mask {
        Some(mask) => client.add(&last, mask).map_err(Error::Numr)?,
        None => last,
    };

    // Device-side argmax over the vocab dimension → [1, 1] of I64.
    let index = client.argmax(&scored, 2, false).map_err(Error::Numr)?;
    let host: Vec<i64> = index.try_to_vec().map_err(Error::Numr)?;
    let picked = host.first().copied().ok_or_else(|| Error::ModelError {
        reason: "argmax over the vocab dimension returned an empty index tensor".into(),
    })?;
    u32::try_from(picked).map_err(|_| Error::ModelError {
        reason: format!("argmax returned an out-of-range token index {picked}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn allow_only_mask_zeroes_candidates_and_blocks_the_rest() {
        let device = CpuDevice::new();
        let mask = allow_only_mask::<CpuRuntime>(&[1, 3, 99], 5, &device)
            .expect("in-range candidates build a mask");
        assert_eq!(mask.shape(), &[1, 1, 5]);
        let host: Vec<f32> = mask.try_to_vec().expect("host copy");
        assert_eq!(host[1], 0.0);
        assert_eq!(host[3], 0.0);
        for &i in &[0usize, 2, 4] {
            assert_eq!(host[i], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn allow_only_mask_rejects_empty_and_out_of_range_candidates() {
        let device = CpuDevice::new();
        assert!(allow_only_mask::<CpuRuntime>(&[], 5, &device).is_err());
        assert!(allow_only_mask::<CpuRuntime>(&[7, 8], 5, &device).is_err());
    }
}
