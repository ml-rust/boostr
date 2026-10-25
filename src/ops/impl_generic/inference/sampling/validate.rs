//! Shared logits validation for `SamplingOps` — every backend method.

use crate::error::{Error, Result};
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
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
    // Every backend reads the last position of a non-empty vocab row.
    let empty = match shape {
        SamplingLogitsShape::Vocab => logits_shape[0] == 0,
        SamplingLogitsShape::BatchSeqVocab => logits_shape[1] == 0 || logits_shape[2] == 0,
    };
    if empty {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "{op}: logits shape {logits_shape:?} has no position or no vocab entry — pass \
                 logits of at least one position and one token"
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

/// Checks that every scratch buffer has the shape and dtype of
/// [`SamplingScratch::new`] for `vocab_size`.
///
/// The `_into` sampling methods write into `scratch.probs`,
/// `scratch.output` and `scratch.random` instead of allocating. The fields
/// are public. A scratch built by hand or for another vocab size reads or
/// writes past its buffer without this check.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` naming `op` and the field, if
/// `probs` is not a contiguous `[vocab_size]` F32 tensor, or `output` not a
/// contiguous `[1]` I64 tensor, or `random` not a contiguous `[1]` F32 tensor.
pub fn validate_scratch_vocab<R: Runtime<DType = DType>>(
    op: &'static str,
    scratch: &SamplingScratch<R>,
    vocab_size: usize,
) -> Result<()> {
    let fields = [
        ("probs", &scratch.probs, vocab_size, DType::F32),
        ("output", &scratch.output, 1, DType::I64),
        ("random", &scratch.random, 1, DType::F32),
    ];
    for (name, t, len, dtype) in fields {
        if t.shape() != [len] || t.dtype() != dtype || !t.is_contiguous() {
            return Err(Error::InvalidArgument {
                arg: "scratch",
                reason: format!(
                    "{op}: scratch.{name} must be a contiguous [{len}] {dtype:?} tensor, got \
                     {:?} of shape {:?} — allocate the scratch with \
                     SamplingScratch::new(vocab_size, device) sized for this call's logits",
                    t.dtype(),
                    t.shape()
                ),
            });
        }
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
    fn rejects_logits_with_no_position_or_no_vocab() {
        let (_client, device) = cpu_setup();
        let empty = Tensor::<CpuRuntime>::zeros(&[0], DType::F32, &device).expect("t");
        assert!(validate_sampling_logits("op", &empty, SamplingLogitsShape::Vocab).is_err());
        for shape in [[1, 0, 4], [1, 2, 0]] {
            let t = Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device).expect("t");
            let err = validate_sampling_logits("op", &t, SamplingLogitsShape::BatchSeqVocab)
                .expect_err("an empty position or vocab must be rejected");
            assert!(
                format!("{err}").contains("no position or no vocab"),
                "{err}"
            );
        }
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

    #[test]
    fn accepts_a_scratch_sized_for_the_call() {
        let (_client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        validate_scratch_vocab("op", &scratch, 4).expect("sized scratch must be accepted");
    }

    #[test]
    fn rejects_a_scratch_sized_for_a_different_vocab() {
        let (_client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        let err = validate_scratch_vocab("op", &scratch, 8)
            .expect_err("a mismatched scratch must be rejected");
        assert!(format!("{err}").contains("op"));
    }

    #[test]
    fn rejects_a_hand_built_scratch_with_a_wrong_output_or_draw_buffer() {
        let (_client, device) = cpu_setup();
        let mut scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        scratch.output = Tensor::<CpuRuntime>::zeros(&[1], DType::I32, &device).expect("output");
        let err = validate_scratch_vocab("op", &scratch, 4).expect_err("I32 output");
        assert!(format!("{err}").contains("scratch.output"), "{err}");
        let mut scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        scratch.random = Tensor::<CpuRuntime>::zeros(&[0], DType::F32, &device).expect("random");
        let err = validate_scratch_vocab("op", &scratch, 4).expect_err("empty draw buffer");
        assert!(format!("{err}").contains("scratch.random"), "{err}");
    }
}
