//! Shared logits validation for `SamplingOps` — every backend method.

use crate::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Shape a `SamplingOps` method expects its logits tensor in.
///
/// `apply_sampling_penalties` and `sample_token` take logits already
/// narrowed to one sequence position; `logits_to_token` takes the
/// unnarrowed `[1, seq_len, vocab_size]` buffer and computes the last
/// position's offset itself.
#[derive(Clone, Copy)]
pub enum SamplingLogitsShape {
    /// `[vocab_size]`
    Vocab,
    /// `[1, seq_len, vocab_size]`
    BatchSeqVocab,
}

/// Checks that a logits tensor is contiguous and has the rank a
/// `SamplingOps` method assumes.
///
/// Every backend kernel for `SamplingOps` reads `logits` through a raw
/// pointer with an offset computed from its shape, not through its
/// strides. A non-contiguous or wrong-rank tensor is silently read at the
/// wrong offsets rather than rejected, so every method validates its
/// input here before touching the pointer.
///
/// `op` names the calling method in any error.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` naming `op`, the tensor's shape and
/// strides, if `logits` is not contiguous or does not match `shape`.
pub fn validate_sampling_logits<R: Runtime>(
    op: &'static str,
    logits: &Tensor<R>,
    shape: SamplingLogitsShape,
) -> Result<()> {
    let logits_shape = logits.shape();
    let shape_ok = match shape {
        SamplingLogitsShape::Vocab => logits_shape.len() == 1,
        SamplingLogitsShape::BatchSeqVocab => logits_shape.len() >= 3,
    };
    if !shape_ok {
        let expected = match shape {
            SamplingLogitsShape::Vocab => "[vocab_size]",
            SamplingLogitsShape::BatchSeqVocab => "rank >= 3 [1, seq_len, vocab_size]",
        };
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "{op}: expected logits shape {expected}, got {logits_shape:?} with strides {:?} — pass a contiguous logits tensor with the shape this op expects",
                logits.strides()
            ),
        });
    }
    validate_contiguous_arg(op, "logits", logits)
}

/// Checks that `tensor` is contiguous, independent of shape.
///
/// `apply_sampling_penalties` and `logits_to_token` read their `token_ids`
/// and `token_counts` arguments through `Tensor::to_vec`, which panics on a
/// non-contiguous tensor. Every method that reads such an argument
/// validates it here first.
///
/// `op` names the calling method, `arg` names the tensor argument, in any
/// error.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` naming `op` and `arg` if `tensor` is not
/// contiguous.
pub fn validate_contiguous_arg<R: Runtime>(
    op: &'static str,
    arg: &'static str,
    tensor: &Tensor<R>,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {arg} tensor is not contiguous (shape {:?}, strides {:?}) — pass a contiguous {arg} tensor",
                tensor.shape(),
                tensor.strides()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn accepts_contiguous_vocab_shape() {
        let (_client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device).expect("t");
        validate_sampling_logits("op", &logits, SamplingLogitsShape::Vocab).expect("valid");
    }

    #[test]
    fn rejects_wrong_rank_for_vocab_shape() {
        let (_client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[1, 2], &device).expect("t");
        let err = validate_sampling_logits("op", &logits, SamplingLogitsShape::Vocab)
            .expect_err("wrong rank must be rejected");
        assert!(format!("{err}").contains("op"));
    }

    #[test]
    fn rejects_non_contiguous_vocab_shape() {
        let (_client, device) = cpu_setup();
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[4, 2], &device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert!(!view.is_contiguous());
        let err = validate_sampling_logits("op", &view, SamplingLogitsShape::Vocab)
            .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("op"));
    }
}
