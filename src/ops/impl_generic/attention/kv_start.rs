//! Per-row left-padding start for flash attention: the `kv_start` argument
//! of `FlashAttentionOps::flash_attention_fwd`.
//!
//! Shape and dtype checks shared by every backend, plus the host-side mask
//! the composed reference path applies.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Checks that `kv_start` is a contiguous `[batch]` I32 tensor.
///
/// Values are not read here: a device backend consumes them in-kernel, and
/// each kernel clamps a start past the key range to an empty range.
pub fn validate_kv_start<R: Runtime<DType = DType>>(
    kv_start: &Tensor<R>,
    batch: usize,
) -> Result<()> {
    if kv_start.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "kv_start",
            reason: format!("expected I32, got {:?}", kv_start.dtype()),
        });
    }
    if kv_start.shape() != [batch] {
        return Err(Error::InvalidArgument {
            arg: "kv_start",
            reason: format!("expected shape [{batch}], got {:?}", kv_start.shape()),
        });
    }
    if !kv_start.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "kv_start",
            reason: "must be contiguous".into(),
        });
    }
    Ok(())
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
    fn mask_clamps_negative_start() {
        assert!(!kv_start_masked(0, -4));
        assert!(kv_start_masked(2, 3));
        assert!(!kv_start_masked(3, 3));
    }
}
