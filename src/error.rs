//! boostr error types

use numr::dtype::DType;

use crate::quant::contract::ActivationContractMismatchDetail;

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

    /// A selected kernel does not compute what the weight's declared
    /// activation contract requires.
    ///
    /// TCF Section 9 resolves dispatch on
    /// `(weight_encoding, contract_digest, execution_role)` and defines no
    /// float fallback, so this is a refusal: the operation stops rather than
    /// running on a kernel whose arithmetic the producer never declared.
    /// Only a weight whose source format can express a contract raises it —
    /// a GGUF weight carries none and never reaches this variant.
    #[error("{0}")]
    ActivationContractMismatch(Box<ActivationContractMismatchDetail>),

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

    /// CUDA kernel error
    #[error("kernel error: {reason}")]
    KernelError {
        /// Description of what went wrong
        reason: String,
    },

    /// Data loading / IO error
    #[error("data error: {reason}")]
    DataError {
        /// Description of what went wrong
        reason: String,
    },

    /// IO error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
