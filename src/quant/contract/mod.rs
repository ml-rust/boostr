//! Activation contracts: what a weight declares, what a kernel satisfies, and
//! the check between them.

mod carrier;
mod declared;
mod kernel;
mod mismatch;

pub use declared::{
    ActivationContract, dot_accumulator_name, input_representation_name, output_dtype_name,
    role_name,
};
pub use kernel::{DYNAMIC_INT8_GROUP, DYNAMIC_INT8_RANGE, KernelContract};
pub use mismatch::ActivationContractMismatchDetail;
