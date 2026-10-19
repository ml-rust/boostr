//! Argument checks for `speculative_verify`, shared by every backend.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The largest vocab `speculative_verify` accepts: ids stay exact in F32.
pub const MAX_VERIFY_VOCAB: usize = 1 << 24;

/// The checked sizes of one `speculative_verify` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyShape {
    /// Rows in the batch.
    pub batch: usize,
    /// Draft tokens per row.
    pub drafts: usize,
    /// Vocab size.
    pub vocab: usize,
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument {
        arg,
        reason: format!("speculative_verify: {reason}"),
    }
}

fn expect_dtype<R: Runtime<DType = DType>>(
    arg: &'static str,
    t: &Tensor<R>,
    dtype: DType,
) -> Result<()> {
    if t.dtype() == dtype {
        return Ok(());
    }
    Err(invalid(
        arg,
        format!("expected {dtype:?}, got {:?} — cast it first", t.dtype()),
    ))
}

fn expect_shape(arg: &'static str, got: &[usize], want: &[usize], dims: &str) -> Result<()> {
    if got == want {
        return Ok(());
    }
    Err(invalid(
        arg,
        format!("expected shape {want:?} ({dims}), got {got:?}"),
    ))
}

/// Checks the ranks, shapes and dtypes of a `speculative_verify` call.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` naming the first argument that fails.
pub fn validate_verify_args<R: Runtime<DType = DType>>(
    target_probs: &Tensor<R>,
    draft_probs: Option<&Tensor<R>>,
    draft_tokens: &Tensor<R>,
    uniforms: &Tensor<R>,
) -> Result<VerifyShape> {
    let ts = target_probs.shape();
    if ts.len() != 3 || ts[1] == 0 || ts[2] == 0 {
        return Err(invalid(
            "target_probs",
            format!("expected [B, K+1, V] with K+1 and V above 0, got {ts:?}"),
        ));
    }
    expect_dtype("target_probs", target_probs, DType::F32)?;
    let (batch, drafts, vocab) = (ts[0], ts[1] - 1, ts[2]);
    if vocab > MAX_VERIFY_VOCAB {
        return Err(invalid(
            "target_probs",
            format!("vocab {vocab} exceeds the limit of {MAX_VERIFY_VOCAB}"),
        ));
    }
    if i32::try_from(drafts + 1).is_err() {
        return Err(invalid(
            "target_probs",
            format!("{} positions exceed the limit of {}", drafts + 1, i32::MAX),
        ));
    }
    if let Some(q) = draft_probs {
        expect_shape("draft_probs", q.shape(), &[batch, drafts, vocab], "B, K, V")?;
        expect_dtype("draft_probs", q, DType::F32)?;
    }
    expect_shape(
        "draft_tokens",
        draft_tokens.shape(),
        &[batch, drafts],
        "B, K",
    )?;
    expect_dtype("draft_tokens", draft_tokens, DType::I32)?;
    expect_shape("uniforms", uniforms.shape(), &[batch, drafts + 1], "B, K+1")?;
    expect_dtype("uniforms", uniforms, DType::F32)?;
    Ok(VerifyShape {
        batch,
        drafts,
        vocab,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    type T = Tensor<CpuRuntime>;

    fn zeros(shape: &[usize], dtype: DType) -> T {
        let (_, device) = cpu_setup();
        T::zeros(shape, dtype, &device).expect("zeros")
    }

    fn arg_of(r: Result<VerifyShape>) -> &'static str {
        match r {
            Err(Error::InvalidArgument { arg, .. }) => arg,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_call_reports_its_sizes() {
        let p = zeros(&[2, 4, 8], DType::F32);
        let q = zeros(&[2, 3, 8], DType::F32);
        let d = zeros(&[2, 3], DType::I32);
        let u = zeros(&[2, 4], DType::F32);
        let shape = validate_verify_args(&p, Some(&q), &d, &u).expect("valid");
        assert_eq!(
            shape,
            VerifyShape {
                batch: 2,
                drafts: 3,
                vocab: 8
            }
        );
        let d0 = zeros(&[2, 0], DType::I32);
        let u0 = zeros(&[2, 1], DType::F32);
        let p0 = zeros(&[2, 1, 8], DType::F32);
        assert_eq!(
            validate_verify_args(&p0, None, &d0, &u0)
                .expect("K = 0")
                .drafts,
            0
        );
    }

    #[test]
    fn each_bad_argument_is_named() {
        let p = zeros(&[2, 4, 8], DType::F32);
        let q = zeros(&[2, 3, 8], DType::F32);
        let d = zeros(&[2, 3], DType::I32);
        let u = zeros(&[2, 4], DType::F32);
        let p_bad = zeros(&[2, 8], DType::F32);
        assert_eq!(
            arg_of(validate_verify_args(&p_bad, None, &d, &u)),
            "target_probs"
        );
        let p_i32 = zeros(&[2, 4, 8], DType::I32);
        assert_eq!(
            arg_of(validate_verify_args(&p_i32, None, &d, &u)),
            "target_probs"
        );
        let q_bad = zeros(&[2, 4, 8], DType::F32);
        assert_eq!(
            arg_of(validate_verify_args(&p, Some(&q_bad), &d, &u)),
            "draft_probs"
        );
        let q_i32 = zeros(&[2, 3, 8], DType::I32);
        assert_eq!(
            arg_of(validate_verify_args(&p, Some(&q_i32), &d, &u)),
            "draft_probs"
        );
        let d_i64 = zeros(&[2, 3], DType::I64);
        assert_eq!(
            arg_of(validate_verify_args(&p, Some(&q), &d_i64, &u)),
            "draft_tokens"
        );
        let message = validate_verify_args(&p, Some(&q), &d_i64, &u)
            .expect_err("I64 draft ids")
            .to_string();
        assert!(
            message.contains("draft_tokens") && message.contains("expected I32"),
            "{message}"
        );
        let d_bad = zeros(&[2, 2], DType::I32);
        assert_eq!(
            arg_of(validate_verify_args(&p, Some(&q), &d_bad, &u)),
            "draft_tokens"
        );
        let u_bad = zeros(&[2, 3], DType::F32);
        assert_eq!(
            arg_of(validate_verify_args(&p, Some(&q), &d, &u_bad)),
            "uniforms"
        );
    }
}
