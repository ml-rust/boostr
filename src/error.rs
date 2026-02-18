//! boostr error types

use numr::dtype::DType;

/// boostr result type
pub type Result<T> = std::result::Result<T, Error>;

/// boostr errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error from numr operations
    #[error("numr error: {0}")]
    Numr(#[from] numr::error::Error),

    /// Unsupported quantization format
    #[error("unsupported quantization format: {format}")]
    UnsupportedQuantFormat {
        /// The format name
        format: String,
    },

    /// Quantization error
    #[error("quantization error: {reason}")]
    QuantError {
        /// Description of what went wrong
        reason: String,
    },

    /// Model loading error
    #[error("model error: {reason}")]
    ModelError {
        /// Description of what went wrong
        reason: String,
    },

    /// DType mismatch for quantized operations
    #[error("dtype mismatch: expected {expected}, got {got}")]
    DTypeMismatch {
        /// Expected dtype
        expected: DType,
        /// Actual dtype
        got: DType,
    },
}
