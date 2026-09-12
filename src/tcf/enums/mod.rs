//! Enumerations decoded from TCF v1 record fields.

#[macro_use]
mod macros;

pub mod contract;
pub mod metadata;
pub mod module;
pub mod tensor;

pub use contract::*;
pub use metadata::*;
pub use module::*;
pub use tensor::*;
