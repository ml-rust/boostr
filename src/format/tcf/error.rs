//! `TcfError` to [`crate::error::Error`] mapping.
//!
//! Every TCF failure becomes an [`Error::ModelError`] carrying the spec's own
//! `E_*` code text plus the context the caller needs to act. Nothing here
//! panics: a malformed file is a value, never an abort.

use crate::error::Error;
use tcf_core::TcfError;

/// Wrap a `TcfError` raised while doing `context`.
///
/// `context` names the operation, e.g. `"open /models/m.tcf"`.
pub fn tcf_error(context: &str, source: TcfError) -> Error {
    Error::ModelError {
        reason: format!("TCF: {context}: {source}"),
    }
}

/// Wrap a `TcfError` raised while handling one named tensor.
///
/// The tensor name is provenance, not identity (Section 6), so it is reported
/// alongside the spec code rather than in place of it.
pub fn tcf_tensor_error(tensor: &str, context: &str, source: TcfError) -> Error {
    Error::ModelError {
        reason: format!("TCF tensor '{tensor}': {context}: {source}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_and_spec_code_both_survive() {
        let err = tcf_error("open model.tcf", TcfError::BadMagic);
        let text = err.to_string();
        assert!(text.contains("open model.tcf"), "{text}");
        assert!(text.contains("E_BAD_MAGIC"), "{text}");
    }

    #[test]
    fn tensor_name_and_spec_code_both_survive() {
        let err = tcf_tensor_error(
            "blk.0.attn_q.weight",
            "verify",
            TcfError::PayloadDigestMismatch { tensor_id: 7 },
        );
        let text = err.to_string();
        assert!(text.contains("blk.0.attn_q.weight"), "{text}");
        assert!(text.contains("E_PAYLOAD_DIGEST_MISMATCH"), "{text}");
    }
}
