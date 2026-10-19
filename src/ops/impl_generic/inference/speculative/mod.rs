//! Backend-independent `speculative_verify`, composed from numr ops.

pub mod accept;
pub mod draw;
pub mod residual;
pub mod validate;
pub mod verify;

pub use validate::{MAX_VERIFY_VOCAB, VerifyShape, validate_verify_args};
pub use verify::speculative_verify_impl;
