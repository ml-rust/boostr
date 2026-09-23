//! Per-row left-padding start for flash attention: the `kv_start` argument
//! of `FlashAttentionOps::flash_attention_fwd`.
//!
//! Shape and dtype checks shared by every backend, plus the host-side mask
//! the composed reference path applies.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Checks that a per-row key bound is a contiguous `[batch]` I32 tensor.
///
/// `arg` names the argument in any error. Values are not read here: a device
/// backend consumes them in-kernel, and each kernel clamps a start past the
/// key range to an empty range.
pub fn validate_row_bounds<R: Runtime<DType = DType>>(
    bounds: &Tensor<R>,
    batch: usize,
    arg: &'static str,
) -> Result<()> {
    if bounds.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected I32, got {:?}", bounds.dtype()),
        });
    }
    if bounds.shape() != [batch] {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected shape [{batch}], got {:?}", bounds.shape()),
        });
    }
    if !bounds.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg,
            reason: "must be contiguous".into(),
        });
    }
    Ok(())
}

/// Checks that `kv_start` is a contiguous `[batch]` I32 tensor.
pub fn validate_kv_start<R: Runtime<DType = DType>>(
    kv_start: &Tensor<R>,
    batch: usize,
) -> Result<()> {
    validate_row_bounds(kv_start, batch, "kv_start")
}

/// Clamped per-row key ends, checked against each row's start and the key
/// axis length `seq_len_k`.
///
/// Row `b` attends over `[kv_start[b], ends[b])`, so an end below its start
/// or past the buffer is rejected rather than clamped. A missing `starts`
/// means every row starts at `0`; a negative start is read as `0`, the way
/// `kv_start_masked` reads it.
pub fn clamped_kv_ends(
    ends: &[i32],
    starts: Option<&[i32]>,
    seq_len_k: usize,
) -> Result<Vec<usize>> {
    ends.iter()
        .enumerate()
        .map(|(b, &end)| {
            let start = starts.map_or(0, |s| s[b].max(0) as i64);
            let end = i64::from(end);
            if end < start || end > seq_len_k as i64 {
                return Err(Error::InvalidArgument {
                    arg: "kv_seq_len",
                    reason: format!(
                        "row {b}: end {end} is outside [start {start}, key axis {seq_len_k}]; \
                         pass an end at or above the row's kv_start and at or below the key \
                         tensor's dim-2"
                    ),
                });
            }
            Ok(end as usize)
        })
        .collect()
}

/// Error for a backend whose kernel does not accept `KvSeqLen::PerRow`.
///
/// `backend` names the flash path in the message (e.g. `"CUDA"`, `"WebGPU"`).
pub fn per_row_kv_seq_len_unsupported(backend: &str) -> Error {
    Error::InvalidArgument {
        arg: "kv_seq_len",
        reason: format!(
            "the {backend} flash path does not yet take a per-row key length; pass \
             KvSeqLen::Shared(n) with one length for the whole batch, or run this batch \
             on the CPU client"
        ),
    }
}

/// Key mask for one batch row: key `j` is valid iff `j >= start`, on top
/// of the causal and window rules of `build_attention_mask`.
///
/// A negative start is clamped to `0`; a start past `seq_len_k` masks
/// every key.
pub fn kv_start_masked(j: usize, start: i32) -> bool {
    let start = usize::try_from(start.max(0)).unwrap_or(0);
    j < start
}

/// Additive masks for a left-padded batch, built on the host.
///
/// Returns `(scores_mask [B, 1, S_q, S_k], row_alive [B, 1, S_q, 1],
/// lse_mask [B, 1, S_q])`. `scores_mask` is `0` where a key is valid and
/// `-inf` where it is not. A query row with no valid key gets a zero mask
/// row instead (so its softmax is finite), `row_alive = 0` to zero its
/// output, and `lse_mask = -inf` for its logsumexp; every other row has
/// `row_alive = 1` and `lse_mask = 0`.
pub fn build_kv_start_masks<R: Runtime<DType = DType>>(
    kv_start: &[i32],
    seq_len_q: usize,
    seq_len_k: usize,
    causal: bool,
    window_size: usize,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)> {
    let batch = kv_start.len();
    let key_offset = seq_len_k.saturating_sub(seq_len_q);
    let mut scores = vec![0.0f32; batch * seq_len_q * seq_len_k];
    let mut alive = vec![1.0f32; batch * seq_len_q];
    let mut lse = vec![0.0f32; batch * seq_len_q];
    for (b, &start) in kv_start.iter().enumerate() {
        for i in 0..seq_len_q {
            let q_pos = key_offset + i;
            let row = (b * seq_len_q + i) * seq_len_k;
            let mut any_valid = false;
            for j in 0..seq_len_k {
                let masked = (causal && j > q_pos)
                    || (window_size > 0 && (j + window_size) <= q_pos)
                    || kv_start_masked(j, start);
                if masked {
                    scores[row + j] = f32::NEG_INFINITY;
                } else {
                    any_valid = true;
                }
            }
            if !any_valid {
                scores[row..row + seq_len_k].fill(0.0);
                alive[b * seq_len_q + i] = 0.0;
                lse[b * seq_len_q + i] = f32::NEG_INFINITY;
            }
        }
    }
    Ok((
        Tensor::<R>::from_slice(&scores, &[batch, 1, seq_len_q, seq_len_k], device)?,
        Tensor::<R>::from_slice(&alive, &[batch, 1, seq_len_q, 1], device)?,
        Tensor::<R>::from_slice(&lse, &[batch, 1, seq_len_q], device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn validate_accepts_i32_batch_vector() {
        let device = CpuDevice::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[0i32, 3, 7], &[3], &device).expect("tensor");
        validate_kv_start(&t, 3).expect("valid");
    }

    #[test]
    fn validate_rejects_wrong_shape_and_dtype() {
        let device = CpuDevice::new();
        let wrong_len =
            Tensor::<CpuRuntime>::from_slice(&[0i32, 3], &[2], &device).expect("tensor");
        assert!(validate_kv_start(&wrong_len, 3).is_err());
        let wrong_dtype =
            Tensor::<CpuRuntime>::from_slice(&[0f32, 3.0, 7.0], &[3], &device).expect("tensor");
        assert!(validate_kv_start(&wrong_dtype, 3).is_err());
    }

    #[test]
    fn validate_row_bounds_rejects_non_contiguous() {
        let device = CpuDevice::new();
        // Every other element of a 4-element buffer: shape [2], stride 2.
        let t = Tensor::<CpuRuntime>::from_slice(&[0i32, 1, 2, 3], &[2, 2], &device)
            .expect("tensor")
            .narrow(1, 0, 1)
            .expect("narrow")
            .squeeze(Some(1));
        assert_eq!(t.shape(), &[2]);
        assert!(!t.is_contiguous());
        assert!(validate_row_bounds(&t, 2, "kv_seq_len").is_err());
    }

    #[test]
    fn clamped_ends_accept_starts_and_buffer_bounds() {
        let ends = clamped_kv_ends(&[0i32, 5, 8], Some(&[0i32, 5, 3]), 8).expect("valid");
        assert_eq!(ends, vec![0, 5, 8]);
    }

    #[test]
    fn clamped_ends_reject_end_below_start() {
        let err = clamped_kv_ends(&[6i32, 2], Some(&[0i32, 4]), 8);
        assert!(err.is_err());
    }

    #[test]
    fn clamped_ends_reject_end_past_buffer() {
        assert!(clamped_kv_ends(&[9i32], None, 8).is_err());
        assert!(clamped_kv_ends(&[-1i32], None, 8).is_err());
    }

    #[test]
    fn mask_clamps_negative_start() {
        assert!(!kv_start_masked(0, -4));
        assert!(kv_start_masked(2, 3));
        assert!(!kv_start_masked(3, 3));
    }
}
