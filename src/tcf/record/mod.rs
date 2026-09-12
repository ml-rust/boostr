//! TCF v1 records: the `Header` singleton (Section 5) and the six array records
//! (Section 7, Section 8, Section 9, Section 10, Section 10.5, Section 11), with their byte-level codec.

#[macro_use]
pub mod macros;

pub mod calibration;
pub mod contract;
pub mod field;
pub mod header;
pub mod module;
pub mod relation;
pub mod tensor;
pub mod traits;
pub mod workload;

#[cfg(test)]
mod testkit;

pub use calibration::CalibrationRecord;
pub use contract::ContractRecord;
pub use field::StringRef;
pub use header::{HEADER_DIGEST_RANGE, Header};
pub use module::ModuleRecord;
pub use relation::RelationRecord;
pub use tensor::TensorRecord;
pub use traits::Record;
pub use workload::WorkloadProfileRecord;
