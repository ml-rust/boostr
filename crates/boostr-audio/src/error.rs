//! boostr-audio error types.
//!
//! Model and kernel errors arrive from boostr and pass through unchanged in
//! [`Error::Boostr`]; the variants declared here cover what this crate adds:
//! argument checks on sample buffers, files that fail to load, and decoded
//! data that does not fit the pipeline.

/// boostr-audio result type.
pub type Result<T> = std::result::Result<T, Error>;

/// boostr-audio errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid argument to an operation.
    #[error("invalid argument '{arg}': {reason}")]
    InvalidArgument {
        /// Argument name.
        arg: &'static str,
        /// Why it is invalid.
        reason: String,
    },

    /// A model asset (voice pack, tokenizer, checkpoint) failed to load.
    #[error("model error: {reason}")]
    ModelError {
        /// Description of what went wrong.
        reason: String,
    },

    /// Decoded or generated data does not fit the pipeline.
    #[error("data error: {reason}")]
    DataError {
        /// Description of what went wrong.
        reason: String,
    },

    /// Error from a boostr model, loader, or kernel.
    #[error(transparent)]
    Boostr(#[from] boostr::error::Error),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<numr::error::Error> for Error {
    /// A numr error reaching this crate came through a boostr tensor call, so
    /// it reports under the same variant as the rest of boostr's errors.
    fn from(err: numr::error::Error) -> Self {
        Self::Boostr(boostr::error::Error::from(err))
    }
}
