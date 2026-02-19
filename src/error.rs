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

    /// Invalid argument to an operation
    #[error("invalid argument '{arg}': {reason}")]
    InvalidArgument {
        /// Argument name
        arg: &'static str,
        /// Why it's invalid
        reason: String,
    },

    /// Inference infrastructure error
    #[error("inference error: {reason}")]
    InferenceError {
        /// Description of what went wrong
        reason: String,
    },

    /// Scheduler error
    #[error("scheduler error: {reason}")]
    SchedulerError {
        /// Description of what went wrong
        reason: String,
    },

    /// Training/optimizer error
    #[error("training error: {reason}")]
    TrainingError {
        /// Description of what went wrong
        reason: String,
    },

    /// Distributed communication error
    #[error("distributed error: {reason}")]
    DistributedError {
        /// Description of what went wrong
        reason: String,
    },
}
