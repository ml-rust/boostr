mod compare;
mod lcg;
mod loss;
mod packed;
mod segments;
mod setup;

pub(crate) use compare::{bits, max_abs_diff, max_abs_diff_f64, tensor_max_abs_diff, to_f64};
pub(crate) use lcg::Lcg;
pub(crate) use loss::weighted_loss;
pub(crate) use packed::{
    check_no_leak, check_no_leak_tol, check_packed_vs_separate, check_packed_vs_separate_tol,
    check_single_document_equals_none, check_single_document_equals_none_tol, t64,
};
pub(crate) use segments::ids_for;
pub(crate) use setup::cpu_setup;
#[cfg(feature = "neucodec")]
pub(crate) use setup::neucodec_checkpoint;
#[cfg(feature = "cuda")]
pub(crate) use setup::{CudaTest, cuda_setup};
