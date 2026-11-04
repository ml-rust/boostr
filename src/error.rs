//! boostr error types

use numr::dtype::DType;

use crate::quant::contract::ActivationContractMismatchDetail;

/// boostr result type
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Returns `true` when numr reports an allocation that fails for lack of
    /// memory. A caller can then name the next action, such as a smaller input.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::Numr(source) if source.is_out_of_memory())
    }

    /// The error of a failed numr call inside a model.
    ///
    /// `reason` names the op and its input. An out-of-memory `source` stays
    /// [`Error::Numr`], so [`Error::is_out_of_memory`] still finds it.
    /// Any other `source` becomes [`Error::ModelError`] with `reason`.
    pub fn model_op(reason: String, source: numr::error::Error) -> Self {
        if source.is_out_of_memory() {
            Self::Numr(source)
        } else {
            Self::ModelError { reason }
        }
    }
}

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
    /// TCF Section 9 resolves dispatch on `weight_encoding`, `execution_role`,
    /// and the contract's typed semantic fields (never `contract_digest`,
    /// an integrity check) and defines no float fallback, so this is a
    /// refusal: the operation stops rather than running on a kernel whose
    /// arithmetic the producer never declared.
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

    /// A caller cancelled the work through its cancel token. Nothing after
    /// `at` ran, and no partial result is returned.
    #[error("cancelled {at}")]
    Cancelled {
        /// The point the work stopped at, such as "before pass 2 of 5".
        at: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_op_keeps_out_of_memory_visible() {
        let source = numr::error::Error::AllocFailed {
            shape: vec![1, 2],
            dtype: "f16".to_string(),
            device: "cuda:0".to_string(),
            source: Box::new(numr::error::Error::OutOfMemory { size: 4 }),
        };
        let error = Error::model_op("conv 'body.0'".to_string(), source);
        assert!(error.is_out_of_memory(), "{error}");
    }

    #[test]
    fn model_op_keeps_the_reason_for_other_errors() {
        let source = numr::error::Error::Backend("capture".to_string());
        let error = Error::model_op("conv 'body.0'".to_string(), source);
        assert!(!error.is_out_of_memory(), "{error}");
        assert!(error.to_string().contains("conv 'body.0'"), "{error}");
    }
}
